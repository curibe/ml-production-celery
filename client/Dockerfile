FROM python:3.10-slim

ENV CLIENT_ROOT "/opt/app"
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV VENV_NAME=".venv"

RUN apt-get update && apt-get install -y jq build-essential curl python3-dev libpq-dev python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR "$CLIENT_ROOT"

RUN python -m venv $VENV_NAME \
ENV PATH="/opt/$VENV_NAME/bin:${PATH}"

RUN pip install --upgrade pip
COPY requirements.txt "$CLIENT_ROOT"
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV STREAMLIT_SERVER_PORT=8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY . "$CLIENT_ROOT"

# Run app.py when the container launches
CMD ["streamlit", "run", "client.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
