FROM python:3.8-slim-buster as req-stage

WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.8-slim-buster
COPY --from=req-stage /tmp/requirements.txt /code/requirements.txt

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential \
        manpages-dev

RUN pip install --no-cache-dir -I -r /code/requirements.txt
COPY src/ /code/src/
COPY pyproject.toml /code/
RUN pip install -I /code/

WORKDIR /code/src/fvt_ml
