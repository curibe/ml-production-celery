import importlib

import torch
from loguru import logger

from app.config import get_settings
from app.models.schemas import GenRequest

settings = get_settings()


class StableDiffusionText2Image:
    def __init__(
            self,
            model_name: str = settings.default_model,
            scheduler: str = "PNDMScheduler",
            pipeline_name: str = "StableDiffusionXLPipeline",
    ):
        self.model_name = model_name
        self.scheduler = scheduler

        self.dtype = torch.float16

        if torch.cuda.is_available():
            self.generator_device = "cuda"
            self.device = "cuda"
        else:
            self.generator_device = "cpu"
            self.device = "cpu"

        self._module_import = importlib.import_module("diffusers")
        self._pipeline = getattr(self._module_import, pipeline_name)

        self.pipe = self._pipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model = self.pipe.to(self.device)

        # Set param optimization
        self.model.enable_attention_slicing()

        # import scheduler
        scheduler = getattr(self._module_import, self.scheduler)
        self.model.scheduler = scheduler.from_config(self.model.scheduler.config)

    def generate_images(self, **kwargs):
        logger.info("generating image in Text2Image pipeline")
        request: GenRequest = kwargs.get("request")
        generator = torch.Generator(self.generator_device).manual_seed(request.generator)

        images = self.model(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_images_per_prompt=request.num_images_per_prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            generator=generator,
            width=request.width,
            height=request.height,
        ).images

        if len(images) == 0:
            logger.info("Unable to generate text2img images")
            return None

        logger.info("text2img completed successfully")
        return images
