from pydantic import BaseModel
from app.config import get_settings

settings = get_settings()


class Request(BaseModel):
    prompt: str = "a beautiful cat with blue eyes, artwork, fujicolor, trending on artstation"
    negative_prompt: str = "bad, low res, ugly, deformed"
    width: int = 768
    height: int = 768
    num_inference_steps: int = 30
    guidance_scale: int = 7
    num_images_per_prompt: int = 1
    generator: int = -1
    pipeline: str = settings.default_pipeline
    scheduler: str = settings.default_scheduler
    modelid: str = settings.default_model


class Response(BaseModel):
    image: str
