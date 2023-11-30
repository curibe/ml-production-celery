
ARG BUILD_ENV="no_copy"
FROM nvidia/cuda:11.7.1-base-ubuntu22.04 as builder
LABEL authors="cesaruribe"

# Configuration defaults
ENV API_ROOT "/opt/api"


ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    VENV_NAME=".venv"


# Install system requirements
RUN apt-get update && apt-get install -y jq build-essential curl python3-dev libpq-dev python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR "$API_ROOT"

# add poetry and venv to path
ENV PATH="$POETRY_HOME/bin:/opt/$VENV_NAME/bin:${PATH}"

# Install dependencies
RUN pip install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python3 -


COPY pyproject.toml poetry.lock ./
RUN  poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR


FROM builder as build_copy
ONBUILD COPY . .

FROM builder as build_no_copy
ONBUILD RUN echo "No copy ..."




