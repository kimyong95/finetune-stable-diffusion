from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
from accelerate.utils import set_seed
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import torch
import einops
import wandb
from torch.utils.data import TensorDataset, DataLoader
from functools import partial
from tqdm import tqdm
from typing import List, Union
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from torch.utils.data import Dataset, DataLoader, Sampler
from rewards import AestheticScorer, PickScoreScorer
from pipeline_extensions import ExtendPipeline, FlowMatchEulerDiscreteSdeScheduler
from reward_normalizer import RewardNormalizer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config.py", "Configuration.")

class DistributedSubsampleDataset(Dataset):
    
    def __init__(self, dataset_dir, split, m, k, base_seed=0):
        self.dataset_path = os.path.join(dataset_dir, f'{split}.txt')
        with open(self.dataset_path, 'r') as f:
            self.all_data = [line.strip() for line in f.readlines()]
        self.N = len(self.all_data)
        self.m = m if m > 0 else self.N  # if m is -1, use the full dataset
        self.k = k
        self.base_seed = base_seed
        self.subsample_indices = None

    # randomly sample a subset of m number of items from the full dataset
    # repeat each item k times
    # in total, B = m * k number of items for each epoch
    def subsample(self, epoch: int):
        rng = random.Random(self.base_seed + epoch)
        chosen = rng.sample(range(self.N), self.m)             # pick m
        repeated = [i for idx in chosen for i in [idx]*self.k] # repeat each k
        rng.shuffle(repeated)
        self.subsample_indices = repeated

    def __len__(self): return self.m * self.k
    def __getitem__(self, i): return self.subsample_indices[i]
    def indices_to_data(self, indices): return [self.all_data[i] for i in indices]

def batches_dict(data, batch_size):
    n = len(next(iter(data.values())))
    for i in range(0, n, batch_size):
        yield {k: v[i:i+batch_size] for k, v in data.items()}

def concat(data: Union[List[torch.Tensor], List[List]]):
    if isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)
    elif isinstance(data[0], list):
        return sum(data, [])
    else:
        raise ValueError(f"Unsupported data type: {type(data[0])}")

def main(_):

    config = FLAGS.config

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="fp16",
    )

    accelerator.gradient_accumulation_steps = config.sample.diffusion_steps * (config.train.effective_batch_size // (config.train.batch_size_per_device * accelerator.num_processes))

    accelerator.init_trackers(
        project_name="finetune-stable-diffusion",
        config=config,
        init_kwargs={"wandb": {"name": config.run_name}}
    )
    set_seed(config.seed, device_specific=True)
    Pipeline = type('ExtendPipeline', (ExtendPipeline, StableDiffusion3Pipeline), {})

    pipeline = Pipeline.from_pretrained(
        config.diffusion.model,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteSdeScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(accelerator.device)
    pipeline.vae.enable_slicing()

    [ module.requires_grad_(False) for module in pipeline.text_encoder.modules() if isinstance(module, torch.nn.Module) ]

    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
    ]
    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    if config.train.gradient_checkpointing:
        pipeline.transformer.enable_gradient_checkpointing()
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_fn = PickScoreScorer()
    reward_fn.to(accelerator.device)

    train_dataset = DistributedSubsampleDataset(dataset_dir=config.dataset_dir, split='train', m=config.sample.m, k=config.sample.k, base_seed=config.seed)
    test_dataset = DistributedSubsampleDataset(dataset_dir=config.dataset_dir, split='test', m=-1, k=1, base_seed=config.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=config.sample.batch_size_per_device, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size_per_device, shuffle=False)

    reward_normalizer = RewardNormalizer()

    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)


    for epoch in range(config.max_epochs):
        pipeline.transformer.eval()
        train_dataset.subsample(epoch)
        training_data = []
        
        # ------------------ Sampling ------------------ #
        for prompt_ids in train_dataloader:
            
            batch_size = len(prompt_ids)
            prompts = train_dataset.indices_to_data(prompt_ids)
            
            with accelerator.autocast():
                with torch.no_grad():
                    images = pipeline(
                        prompts,
                        guidance_scale=config.diffusion.guidance_scale,
                        num_inference_steps=config.sample.diffusion_steps,
                    ).images
            trajectory_data = pipeline.scheduler.collect_trajectory_data()
            trajectory_data = {k: torch.stack([d[k] for d in trajectory_data], dim=1) for k in trajectory_data[0]} 
            
            rewards = reward_fn(images, prompts)

            training_data.append({
                    "prompt_ids": prompt_ids,
                    "sample": trajectory_data["sample"],
                    "prev_sample": trajectory_data["prev_sample"],
                    "prev_sample_mean": trajectory_data["prev_sample_mean"],
                    "rewards": rewards,
                    "images": images,
            })
            del trajectory_data
        
        training_data = {
            key: concat([batch[key] for batch in training_data])
            for key in training_data[0].keys()
        }

        gathered_prompt_ids = accelerator.gather(training_data["prompt_ids"])
        gathered_rewards = accelerator.gather(training_data["rewards"])
        reward_normalizer.fit(gathered_prompt_ids, gathered_rewards)
        gathered_advantages = reward_normalizer.transform(gathered_prompt_ids, gathered_rewards)

        # un-gather
        training_data["advantages"] = einops.rearrange(gathered_advantages,'(process batch) -> process batch',process=accelerator.num_processes)[accelerator.process_index]

        # ------------------ Training ------------------ #
        pipeline.transformer.train()
        timesteps, _ = retrieve_timesteps(pipeline.scheduler, num_inference_steps=config.sample.diffusion_steps, device=accelerator.device)
        for training_batch in batches_dict(training_data, config.train.batch_size_per_device):
            prompts = train_dataset.indices_to_data(training_batch["prompt_ids"])
            for i, timestep in enumerate(timesteps):
                with accelerator.accumulate(transformer), accelerator.autocast():

                    with torch.enable_grad():
                        prev_sample_mean = pipeline.sample_one_step_mean(prompts, training_batch["sample"][:,i], timestep, guidance_scale=config.diffusion.guidance_scale)

                    with accelerator.unwrap_model(transformer).disable_adapter(), torch.no_grad():
                        prev_sample_mean_ref = pipeline.sample_one_step_mean(prompts, training_batch["sample"][:,i], timestep, guidance_scale=config.diffusion.guidance_scale)
                    
                    # p_θ(x_{t-1}|x_t) in the paper
                    curr_log_prob = pipeline.scheduler.calculate_log_prob(prev_sample_mean=prev_sample_mean, prev_sample=training_batch["prev_sample"][:,i], timestep=timestep)
                    
                    # p_{θ_old}(x_{t-1}|x_t) in the paper
                    old_log_prob = pipeline.scheduler.calculate_log_prob(prev_sample_mean=training_batch["prev_sample_mean"][:,i], prev_sample=training_batch["prev_sample"][:,i], timestep=timestep)

                    advantages = torch.clamp(
                        training_batch["advantages"],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    ratio = torch.exp(curr_log_prob - old_log_prob)
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - config.train.clip_range,
                        1.0 + config.train.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    std_dev_t = pipeline.scheduler.get_coeficients(timestep)["std_dev_t"]
                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                    kl_loss = torch.mean(kl_loss)

                    loss = policy_loss + config.train.beta * kl_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

        # ------------------ Logging ------------------ #
        accelerator.log({
            "epoch": epoch,
            "train/rewards": training_data["rewards"].mean().item(),
        }, step=epoch)

        wandb_images = []
        for idx, (image, reward) in enumerate(zip(training_data["images"], training_data["rewards"])):
            wandb_images.append(wandb.Image(image,caption=f"i={idx},f={reward.item():.4f}",file_type="jpeg"))
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.log({"images":wandb_images}, step=epoch)

    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
