from typing import List, Union
import torch
from diffusers import StableDiffusion3Pipeline

class ExtendPipeline:

    def sample_one_step_mean(
        self: Union[StableDiffusion3Pipeline],
        prompt: List[str],
        latents: torch.FloatTensor,
        timestep: torch.FloatTensor,
        guidance_scale: float,
        max_sequence_length: int = 256,
    ):
        with torch.no_grad():
            (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = self.encode_prompt(prompt=prompt,prompt_2=prompt,prompt_3=prompt,do_classifier_free_guidance=self.do_classifier_free_guidance,device=self._execution_device,num_images_per_prompt=1,max_sequence_length=max_sequence_length)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
            latent_model_input = torch.cat([latents] * 2)

        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep.expand(latent_model_input.shape[0]),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        self.scheduler._step_index = None # reset step index of scheduler
        _ = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        step_index = self.scheduler.index_for_timestep(timestep)
        prev_sample_mean = self.scheduler.collect_trajectory_data()[step_index]["prev_sample_mean"]

        return prev_sample_mean

import torch
import math
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler, FlowMatchEulerDiscreteSchedulerOutput

class FlowMatchEulerDiscreteSdeScheduler(FlowMatchEulerDiscreteScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        noise_level: float = 0.7,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma_idx = self.step_index
        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]

        current_sigma = sigma
        next_sigma = sigma_next
        dt = sigma_next - sigma

        # ------------ modified part ------------ #
        # Warning: not multi-thread safe
        
        if not hasattr(self, "trajectory_data") or self.trajectory_data is None:
            self.trajectory_data = [None]*len(self.timesteps)

        sigma_max = self.sigmas[1].item()
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
        prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
        variance_noise = torch.randn_like(sample)
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

        self.trajectory_data[self._step_index] = {
            "sample": sample,
            "prev_sample": prev_sample,
            "prev_sample_mean": prev_sample_mean,
        }
        # --------------------------------------- #

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
    
    def collect_trajectory_data(self):
        data = self.trajectory_data
        del self.trajectory_data
        return data
    
    # prev_sample_mean: μ_θ(x_t; t) in the paper
    # prev_sample: x_{t-1} in the paper
    def calculate_log_prob(self, prev_sample_mean, prev_sample, timestep):

        coeficients = self.get_coeficients(timestep)
        std_dev_t = coeficients["std_dev_t"]
        dt = coeficients["dt"]

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
            - torch.log(std_dev_t * torch.sqrt(-1*dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return log_prob
    
    def get_coeficients(self, timestep, noise_level=0.7):
        step_index = self.index_for_timestep(timestep)
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        dt = sigma_next - sigma
        sigma_max = self.sigmas[1].item()
        std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

        return {
            "sigma": sigma,
            "sigma_next": sigma_next,
            "std_dev_t": std_dev_t,
            "dt": dt,
        }