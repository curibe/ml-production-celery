from pydantic import BaseModel

TASK_TYPE = "CELERY_CUSTOM"

url_map = {
    "CELERY_CUSTOM": "http://server:8000/generate_async_custom",
    "CELERY": "http://server:8000/generate_async",
    "ASYNC": "http://server:8000/generate",
}

api_url = url_map.get(TASK_TYPE, "http://server:8000/generate")
result_url = "http://server:8000/results"

model_map = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 2": "stabilityai/stable-diffusion-2"
}

pipeline_map = {
    "stabilityai/stable-diffusion-xl-base-1.0": "StableDiffusionXLPipeline",
    "stabilityai/stable-diffusion-2": "StableDiffusionPipeline"
}

scheduler_map = {
    'Euler': 'EulerDiscreteScheduler',
    'Euler a': 'EulerAncestralDiscreteScheduler',
    'DPM2': 'KDPM2DiscreteScheduler',
    'LMS': 'LMSDiscreteScheduler',
    'Heun': 'HeunDiscreteScheduler',
    'DPM2 a': 'KDPM2AncestralDiscreteScheduler',
    'DDPM': 'DDPMScheduler',
    'DPM++ 2M': 'DPMSolverMultistepScheduler',
    'DDIM': 'DDIMScheduler',
    'DPM': 'DPMSolverSinglestepScheduler',
    'DEIS': 'DEISMultistepScheduler',
    'UniPC': 'UniPCMultistepScheduler',
    'PNDM': 'PNDMScheduler'
}

size_dict = {
        "512x512": (512, 512),
        "768x768": (768, 768)
    }


class GenRequest(BaseModel):
    prompt: str = ("logo of The Witcher Portrait for a spiritual Premium Instagram channel,artwork, fujicolor, "
                   "trending on artstation, faded forest background")
    negative_prompt: str = "bad, low res, ugly, deformed"
    width: int = 768
    height: int = 768
    num_inference_steps: int = 60
    guidance_scale: float = 5.0
    num_images_per_prompt: int = 1
    generator: int = 88300
    pipeline: str = pipeline_map[model_map["Stable Diffusion XL"]]
    scheduler: str = scheduler_map["PNDM"]
    modelid: str = model_map["Stable Diffusion XL"]


default_values = GenRequest()
