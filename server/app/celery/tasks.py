from app.celery.celery_app import app
from app.models.schemas import GenRequest
from app.operators.stable_diffusion import StableDiffusionText2Image


@app.task
def task_generate_image_with_text2img(*, request: GenRequest) -> str:
    generator = StableDiffusionText2Image(model_name=request.modelid, scheduler=request.scheduler,
                                          pipeline_name=request.pipeline)
    generation = generator.generate_images(request=request)
    return generation
