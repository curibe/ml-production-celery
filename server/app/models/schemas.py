from pydantic import BaseModel
from app.config import get_settings

settings = get_settings()


class GenRequest(BaseModel):
    prompt: str = "logo of The Witcher Portrait for a spiritual Premium Instagram channel,artwork, fujicolor, trending on artstation, faded forest background"
    negative_prompt: str = "bad, low res, ugly, deformed"
    width: int = 768
    height: int = 768
    num_inference_steps: int = 60
    guidance_scale: float = 5
    num_images_per_prompt: int = 1
    generator: int = 88300
    pipeline: str = settings.default_pipeline
    scheduler: str = settings.default_scheduler
    modelid: str = settings.default_model


class Response(BaseModel):
    image: str
