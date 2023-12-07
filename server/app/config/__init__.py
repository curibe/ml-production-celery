from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API params
    allowed_origins: str = "*"

    # Model params
    default_scheduler: str = "DDPMScheduler"
    default_pipeline: str = "StableDiffusionXLPipeline"
    default_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    max_num_images: int = 4

    # celery params
    celery_broker_url: str = "redis://redis:6379/0"
    celery_backend_url: str = "redis://redis:6379/1"
    celery_task_queue: str = "stable_diffusion"
    celery_worker_prefetch_multiplier: int = 1


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    return settings
