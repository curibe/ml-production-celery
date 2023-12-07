from celery import Celery
from kombu import Exchange, Queue

from app.config import get_settings

settings = get_settings()

# Define queue names
stable_diffusion_queue_name = settings.celery_task_queue
stable_diffusion_exchange_name = settings.celery_task_queue
stable_diffusion_routing_key = settings.celery_task_queue
worker_prefetch_multiplier = settings.celery_worker_prefetch_multiplier

# get all task modules
task_modules = ["app.celery.tasks"]

app = Celery(__name__)
app.conf.broker_url = settings.celery_broker_url
app.conf.result_backend = settings.celery_backend_url

# define exchanges
stable_diffusion_exchange = Exchange(stable_diffusion_exchange_name, type="direct")

# define queues
stable_diffusion_queue = Queue(
    stable_diffusion_queue_name,
    stable_diffusion_exchange,
    routing_key=stable_diffusion_routing_key,
)

# set the task queues
app.conf.task_queues = stable_diffusion_queue

# set the task routes
app.conf.task_routes = {
    "app.celery.tasks.*": {"queue": stable_diffusion_queue_name},
}

# serializer and accept content
app.conf.task_serializer = "pickle"
app.conf.result_serializer = "pickle"
app.conf.accept_content = ["application/json", "application/x-python-serialize"]
app.autodiscover_tasks(task_modules)

app.conf.worker_prefetch_multiplier = worker_prefetch_multiplier
