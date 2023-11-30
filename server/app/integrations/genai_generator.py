from app.integrations.generation_interface import GenerationInterface
from app.models.schemas import Request
from app.operators.stable_diffusion import StableDiffusionText2Image


class GenerativeAIGenerator(GenerationInterface):
    @staticmethod
    def generate_image_with_text2img(*, request: Request) -> str:
        generator = StableDiffusionText2Image()
        generation = generator.generate_images(request=request)
        return generation
