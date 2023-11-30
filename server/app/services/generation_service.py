from app.integrations.generation_interface import GenerationInterface
from app.models.schemas import Request


class GenerationService:
    def __init__(self, generator: GenerationInterface):
        self.generation = generator

    def generate_images_with_text2img(self, request: Request):
        generation = self.generation.generate_image_with_text2img(request=request)
        return generation

