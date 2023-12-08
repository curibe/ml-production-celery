from app.integrations.generation_interface import GenerationInterface
from app.models.schemas import GenRequest
from app.operators.stable_diffusion import StableDiffusionText2Image
from loguru import logger
from app.celery.tasks import task_custom_generate_image_with_text2img, task_generate_image_with_text2img


class GenerativeAIGenerator(GenerationInterface):
    @staticmethod
    def generate_image_with_text2img(*, request: GenRequest):
        generator = StableDiffusionText2Image(model_name=request.modelid, scheduler=request.scheduler,
                                              pipeline_name=request.pipeline)
        generation = generator.generate_images(request=request)
        return generation


class GenerativeAIGeneratorCelery(GenerationInterface):
    @staticmethod
    def generate_image_with_text2img(*, request: GenRequest) -> str:
        logger.info("generating image in Text2Image pipeline")
        taskid = task_generate_image_with_text2img.delay(request=request)
        return str(taskid)


class GenerativeAIGeneratorCeleryCustom(GenerationInterface):
    @staticmethod
    def generate_image_with_text2img(*, request: GenRequest) -> str:
        logger.info("generating image in Text2Image pipeline")
        taskid = task_custom_generate_image_with_text2img.delay(request=request)
        return str(taskid)
