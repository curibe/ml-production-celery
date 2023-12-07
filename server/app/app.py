from io import BytesIO
from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.config.logger import InitLogger
from app.integrations.genai_generator import GenerativeAIGenerator, GenerativeAIGeneratorCelery
from app.models.schemas import GenRequest
from app.services.generation_service import GenerationService
from app.utils.images import from_image_to_bytes

app = FastAPI()

# Create service to generate an image with Diffusion models
# We inject the image generator integration dependency in the service
generator_service = GenerationService(generator=GenerativeAIGenerator())
generator_service_celery = GenerationService(generator=GenerativeAIGeneratorCelery())

# Load config to Logging system
config_path = Path("app/config").absolute() / "logging-conf.yaml"
logger = InitLogger.create_logger(config_path)

# Load settings
settings = get_settings()

ALLOWED_ORIGINS = settings.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def root():
    return {"message": "Hello World"}


@app.post('/generate')
async def generate(request: GenRequest):
    # Call the service to generate the images according to the request params
    images = generator_service.generate_images_with_text2img(request=request)
    img_bytes = from_image_to_bytes(images[0])
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")


@app.post('/generate_async')
async def generate_async(request: GenRequest):
    # Call the service to generate the images according to the request params
    taskid = generator_service_celery.generate_images_with_text2img(request=request)
    return {"taskid": taskid}


@app.get(
    "/results/{task_id}",
)
async def get_generation_result(task_id, ):
    task = AsyncResult(task_id)

    if not task.ready():
        return JSONResponse(content={"task": str(task.status)}, status_code=202)

    result = task.get()
    img_bytes = from_image_to_bytes(result[0])
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
