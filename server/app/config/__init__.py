from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    allowed_origins: str = "*"
    default_scheduler: str = "DDPMScheduler"
    default_pipeline: str = "StableDiffusionXLPipeline"
    default_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    max_num_images: int = 4


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    return settings
