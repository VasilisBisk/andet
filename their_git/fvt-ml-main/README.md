# fvt-ml

## Models

### CPU Spike Detection

- Standard Deviation Anomaly Detection

## How to build the Docker image

`docker build -t fvt-ml .`

## How to run the application using the Docker image

`docker container run --rm fvt-ml stdev-model-job run <target-id> <start-date> <end-date> <resample_rule>`

## How to run the application locally

### Install the application

`poetry install`

### Run the application

`fvt-ml stdev-model-job run  <target-id> <start-date> <end-date> <resample_rule>`

## Environment Variables required

### ElasticSearch Raw Data Repo

- `ES_RAW_DATA_REPO_HOST`
- `ES_RAW_DATA_REPO_USERNAME`
- `ES_RAW_DATA_REPO_PASSWORD`
- `ES_RAW_DATA_REPO_INDEX`
- `ES_RAW_DATA_REPO_KEEP_ALIVE`
- `ES_RAW_DATA_REPO_SIZE`
