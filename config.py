import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.max_epochs = 1000

    config.run_name = "grpo"
    config.dataset_dir = "dataset/ocr"
    config.dataset_one_prompt = None
    # config.dataset_one_prompt = "Traffic light with lights from top to bottom: green, yellow, red."

    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.model = "stabilityai/stable-diffusion-3.5-medium"
    config.diffusion.guidance_scale = 4.5
    config.diffusion.resolution = 512
    

    config.sample = ml_collections.ConfigDict()
    config.sample.batch_size_per_device = 16
    # total number of samples (across all devices) = m * k
    config.sample.m = 8     # unique prompt per epoch (across all devices)
    config.sample.k = 8     # repeats the unique prompt
    config.sample.diffusion_steps = 10

    config.reward = "ocr"

    config.train = ml_collections.ConfigDict()
    config.train.batch_size_per_device = 4
    config.train.learning_rate = 1e-4
    config.train.adv_clip_max = 5.0
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.gradient_checkpointing = False
    config.train.clip_range=1e-3
    config.train.beta = 0.04
    config.train.effective_batch_size = 32  # acheive via gradient accumulation steps

    config.test = ml_collections.ConfigDict()
    config.test.batch_size_per_device = 32
    config.test.diffusion_steps = 40

    return config
