from collections import defaultdict
import contextlib
import os
import time
from concurrent import futures
import itertools
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
import sys
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from torch.utils.data import Dataset, DataLoader, Sampler
from rewards import REWARDS_CLS
from pipeline_extensions import ExtendPipeline, FlowMatchEulerDiscreteSdeScheduler, batch_cache
from reward_normalizer import RewardNormalizer

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config.py", "Configuration.")


@contextlib.contextmanager
def swap_attrs(obj, **attrs):
    old = {k: getattr(obj, k) for k in attrs}; [setattr(obj, k, v) for k, v in attrs.items()]
    try: 
        yield obj
    finally:
        [setattr(obj, k, v) for k, v in old.items()]
class DistributedSubsampleDataset(Dataset):
    
    def __init__(self, m, k, base_seed=0, dataset_dir=None, split=None, one_prompt=None):
        assert (dataset_dir is not None) ^ (one_prompt is not None), "Exactly one of 'dataset_dir' or 'one_prompt' must be defined"
        
        if dataset_dir is not None:
            with open(os.path.join(dataset_dir, f'{split}.txt'), 'r') as f:
                self.all_data = [line.strip() for line in f.readlines()]
        elif one_prompt is not None:
            self.all_data = [one_prompt] * m
        
        self.N = len(self.all_data)
        self.m = self.N if m == -1 else m  # if m is -1, use the full dataset
        self.k = k
        self.base_seed = base_seed
        self.subsample_indices = [i for i in range(self.N)] if m == -1 else None

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

class Trainer:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator(
            log_with="wandb",
            mixed_precision="fp16",
        )
        self.accelerator.gradient_accumulation_steps = self.config.sample.diffusion_steps * (self.config.train.effective_batch_size // (self.config.train.batch_size_per_device * self.accelerator.num_processes))

        self.accelerator.init_trackers(
            project_name="finetune-stable-diffusion",
            config=self.config,
            init_kwargs={"wandb": {"name": self.config.run_name, "config": self.config.to_dict()}}
        )
        set_seed(self.config.seed, device_specific=True)
        Pipeline = type('ExtendPipeline', (ExtendPipeline, StableDiffusion3Pipeline), {})

        self.pipeline = Pipeline.from_pretrained(
            self.config.diffusion.model,
            torch_dtype=torch.float16,
        )
        self.original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = FlowMatchEulerDiscreteSdeScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to(self.accelerator.device)
        self.pipeline.vae.enable_slicing()
        self.pipeline.encode_prompt = batch_cache(max_size=32)(self.pipeline.encode_prompt)
        [ comp.requires_grad_(False) for comp in self.pipeline.components.values() if isinstance(comp, torch.nn.Module) ] # disable all gradients
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
        self.pipeline.transformer = get_peft_model(self.pipeline.transformer, transformer_lora_config)
        self.transformer = self.pipeline.transformer
        
        if self.config.train.gradient_checkpointing:
            self.pipeline.transformer.enable_gradient_checkpointing()
        transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))

        self.optimizer = torch.optim.AdamW(
            transformer_trainable_parameters,
            lr=self.config.train.learning_rate,
            betas=(self.config.train.adam_beta1, self.config.train.adam_beta2),
            weight_decay=self.config.train.adam_weight_decay,
            eps=self.config.train.adam_epsilon,
        )

        RewardCls = REWARDS_CLS[self.config.reward]
        self.reward_fn = RewardCls()
        self.reward_fn.to(self.accelerator.device)

        self.train_dataset = DistributedSubsampleDataset(dataset_dir=self.config.dataset_dir, split="train", one_prompt=self.config.dataset_one_prompt, m=self.config.sample.m, k=self.config.sample.k, base_seed=self.config.seed)
        self.eval_dataset = DistributedSubsampleDataset(dataset_dir=self.config.dataset_dir, split="eval", one_prompt=self.config.dataset_one_prompt, m=-1, k=1, base_seed=self.config.seed)

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.sample.batch_size_per_device)
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.config.eval.batch_size_per_device)

        self.reward_normalizer = RewardNormalizer()

        self.transformer, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(self.transformer, self.optimizer, train_dataloader, eval_dataloader)

    def run(self):
        if self.config.eval.frequency > 0:
            self.evaluation_step(epoch=0)
        
        for epoch in tqdm(range(1, self.config.max_epochs+1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            
            training_data = self.sampling_step(epoch)
            self.training_step(epoch=epoch, training_data=training_data)

            if self.config.eval.frequency > 0 and epoch % self.config.eval.frequency == 0:
                self.evaluation_step(epoch=epoch)

        self.accelerator.end_training()

    def evaluation_step(self, epoch):
        
        self.pipeline.transformer.eval()
        all_rewards = []
        log_images = []
        for prompt_ids in tqdm(self.eval_dataloader, desc="Evaluation", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            prompts = self.train_dataset.indices_to_data(prompt_ids)
            
            with swap_attrs(self.pipeline, scheduler=self.original_scheduler), torch.no_grad(), self.accelerator.autocast():
                images = self.pipeline(
                    prompts,
                    guidance_scale=self.config.diffusion.guidance_scale,
                    num_inference_steps=self.config.eval.diffusion_steps,
                    height=self.config.diffusion.resolution,
                    width=self.config.diffusion.resolution,
                ).images
            
            rewards = self.reward_fn(images, prompts)
            rewards = rewards.to(device=self.accelerator.device, dtype=torch.float32)
            all_rewards.append(rewards)
            log_images.extend(images)
        
        all_rewards = torch.cat(all_rewards, dim=0)
        gathered_rewards = self.accelerator.gather(all_rewards)
        self.log_rewards(epoch=epoch, rewards=gathered_rewards, stage="eval")

        # assume single device generated images is enough for logging
        log_images = log_images[:self.config.eval.log_images]
        log_images_rewards = all_rewards[:self.config.eval.log_images]
        self.log_images(epoch=epoch, rewards=log_images_rewards, images=log_images, stage="eval")


    def sampling_step(self, epoch):
        self.pipeline.transformer.eval()
        training_data = []
        self.train_dataset.subsample(epoch)
        for prompt_ids in tqdm(self.train_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            
            prompts = self.train_dataset.indices_to_data(prompt_ids)
            
            with torch.no_grad(), self.accelerator.autocast():
                images = self.pipeline(
                    prompts,
                    guidance_scale=self.config.diffusion.guidance_scale,
                    num_inference_steps=self.config.sample.diffusion_steps,
                    height=self.config.diffusion.resolution,
                    width=self.config.diffusion.resolution,
                ).images
            trajectory_data = self.pipeline.scheduler.collect_trajectory_data()
            trajectory_data = {k: torch.stack([d[k] for d in trajectory_data], dim=1) for k in trajectory_data[0]} 
            
            rewards = self.reward_fn(images, prompts)
            rewards = rewards.to(device=self.accelerator.device, dtype=torch.float32)

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

        gathered_prompt_ids = self.accelerator.gather(training_data["prompt_ids"])
        gathered_rewards = self.accelerator.gather(training_data["rewards"])
        self.reward_normalizer.fit(gathered_prompt_ids, gathered_rewards)
        gathered_advantages = self.reward_normalizer.transform(gathered_prompt_ids, gathered_rewards)

        training_data["advantages"] = einops.rearrange(gathered_advantages,'(process batch) -> process batch',process=self.accelerator.num_processes)[self.accelerator.process_index]

        non_zero_mask = training_data["advantages"] != 0
        training_data = {
            key: value[non_zero_mask] if isinstance(value, torch.Tensor) else list(itertools.compress(value, non_zero_mask))
            for key, value in training_data.items()
        }

        self.log_rewards(epoch=epoch, rewards=gathered_rewards, stage="sampling")

        return training_data

    def training_step(self, epoch, training_data):
        self.pipeline.transformer.train()
        timesteps, _ = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps=self.config.sample.diffusion_steps, device=self.accelerator.device)
        
        training_batches = list(batches_dict(training_data, self.config.train.batch_size_per_device))
        
        for training_batch in tqdm(training_batches, desc="Training Batches", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = self.train_dataset.indices_to_data(training_batch["prompt_ids"])
            
            for i, timestep in tqdm(enumerate(timesteps), desc="Timesteps", position=2, leave=False, total=len(timesteps), disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(self.transformer):
                    
                    with torch.enable_grad(), self.accelerator.autocast():
                        prev_sample_mean = self.pipeline.sample_one_step_mean(prompts, training_batch["sample"][:,i], timestep, guidance_scale=self.config.diffusion.guidance_scale)

                    with self.accelerator.unwrap_model(self.transformer).disable_adapter(), torch.no_grad(), self.accelerator.autocast():
                        prev_sample_mean_ref = self.pipeline.sample_one_step_mean(prompts, training_batch["sample"][:,i], timestep, guidance_scale=self.config.diffusion.guidance_scale)

                    curr_log_prob = self.pipeline.scheduler.calculate_log_prob(prev_sample_mean=prev_sample_mean, prev_sample=training_batch["prev_sample"][:,i], timestep=timestep)
                    
                    old_log_prob = self.pipeline.scheduler.calculate_log_prob(prev_sample_mean=training_batch["prev_sample_mean"][:,i], prev_sample=training_batch["prev_sample"][:,i], timestep=timestep)

                    advantages = torch.clamp(
                        training_batch["advantages"],
                        -self.config.train.adv_clip_max,
                        self.config.train.adv_clip_max,
                    )
                    ratio = torch.exp(curr_log_prob - old_log_prob)
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - self.config.train.clip_range,
                        1.0 + self.config.train.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    std_dev_t = self.pipeline.scheduler.get_coeficients(timestep)["std_dev_t"]
                    kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                    kl_loss = torch.mean(kl_loss)

                    loss = policy_loss + self.config.train.beta * kl_loss

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.config.train.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def log_rewards(self, epoch, rewards, stage):
        self.accelerator.log({
            "epoch": epoch,
            f"{stage}/rewards": rewards.mean().item(),
        }, step=epoch)

    def log_images(self, epoch, rewards, images, stage):
        wandb_images = []
        for idx, (image, reward) in enumerate(zip(images, rewards)):
            wandb_images.append(wandb.Image(image,caption=f"i={idx},f={reward.item():.4f}",file_type="jpeg"))
        wandb_tracker = self.accelerator.get_tracker("wandb")
        wandb_tracker.log({f"{stage}/images":wandb_images}, step=epoch)

if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
