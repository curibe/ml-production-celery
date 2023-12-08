import importlib

from celery import Task
from loguru import logger

from app.celery.celery_app import app
from app.config import get_settings
from app.models.schemas import GenRequest
from app.operators.stable_diffusion import StableDiffusionText2Image

settings = get_settings()


class DiffusionTask(Task):
    abstract = True

    def __init__(self, name: str = settings.default_model):
        super().__init__()
        self.model = None
        self.model_name = name
        self.module_import = importlib.import_module(self.path[0])
        self.model_obj = getattr(self.module_import, self.path[1])
        logger.info(f"Init task with model {self.model_name}")

    def __call__(self, *args, **kwargs):
        if not self.model:
            logger.info("loading model....")

            self.model = self.model_obj(model_name=self.name)
            logger.info("Model loaded")

        return self.run(*args, **kwargs)


@app.task(
    ignore_result=False,
    bind=True,
    base=DiffusionTask,
    name=f"{__name__}.task_custom_generate_image_with_text2img",
    path=("app.operators.stable_diffusion", "StableDiffusionText2Image")
)
def task_custom_generate_image_with_text2img(self, request: GenRequest) -> str:
    # Check model and scheduler from request
    model_selected = request.modelid
    scheduler_selected = request.scheduler

    logger.info(f"Current model: {self.model.model_name} - Model selected: {model_selected}")
    logger.info(f"Current scheduler: {self.model.scheduler} - Sampler selected: {scheduler_selected}")

    # If model or scheduler are different, we need to reinitialize the model
    if self.model.model_name != model_selected and self.model.scheduler != scheduler_selected:
        self.model.__init__(
            model_name=model_selected,
            scheduler=scheduler_selected,
            pipeline_name=request.pipeline,
        )

    # Generate images
    generation = self.model.generate_images(request=request)

    return generation


@app.task
def task_generate_image_with_text2img(*, request: GenRequest) -> str:
    generator = StableDiffusionText2Image(model_name=request.modelid, scheduler=request.scheduler,
                                          pipeline_name=request.pipeline)
    generation = generator.generate_images(request=request)
    return generation
