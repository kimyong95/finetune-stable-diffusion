import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 0

    config.max_epochs = 1000

    config.run_name = "grpo"
    config.dataset_dir = "dataset/pickscore"

    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.model = "stabilityai/stable-diffusion-3.5-medium"
    config.diffusion.guidance_scale = 3.5

    config.sample = ml_collections.ConfigDict()
    config.sample.batch_size_per_device = 16
    # total number of samples (across all devices) = m * k
    config.sample.m = 2     # unique prompt per epoch (across all devices)
    config.sample.k = 32    # repeats the unique prompt
    config.sample.diffusion_steps = 10

    config.train = ml_collections.ConfigDict()
    config.train.batch_size_per_device = 8
    config.train.learning_rate = 1e-4
    config.train.adv_clip_max = 5.0
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon = 1e-8
    config.train.max_grad_norm = 1.0
    config.train.gradient_checkpointing = True
    config.train.clip_range=1e-3
    config.train.beta = 0.01
    config.train.effective_batch_size = 32  # acheive via gradient accumulation steps

    config.test = ml_collections.ConfigDict()
    config.test.batch_size_per_device = 32
    config.test.diffusion_steps = 40

    return config
