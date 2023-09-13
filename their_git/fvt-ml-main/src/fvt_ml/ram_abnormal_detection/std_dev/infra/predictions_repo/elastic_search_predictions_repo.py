import datetime
import json
import logging

import pandas as pd
import requests
from pydantic import BaseSettings

from fvt_ml.ram_abnormal_detection.std_dev.infra.predictions_repo.predictions_repo import (
    PredictionsRepo,
)


class ElasticSearchPredictionsRepo(PredictionsRepo):
    def __init__(self, host: str, index: str, username: str, password: str) -> None:
        if host.endswith("/"):
            raise Exception("Host URL should not end with /")
        self.uri = f"{host}/{index}/_create/"
        self._username = username
        self._password = password
        self._headers = {"Content-Type": "application/json"}
        self._logger = logging.getLogger(self.__class__.__name__)

    def put(self, predictions: pd.DataFrame) -> None:
        predictions["@timestamp"] = datetime.datetime.utcnow().isoformat()
        predictions.timestamp_id = predictions.timestamp_id.apply(lambda x: x.isoformat())

        count = 0

        for doc in predictions.to_dict("records"):
            self._logger.debug("Storing document %s...", doc)
            resp = requests.post(
                self.uri + f"{doc['timestamp_id']}",
                data=json.dumps(doc),
                headers=self._headers,
                auth=(self._username, self._password),
            )
            if not resp.ok:
                self._logger.warning("Response to request: %s", resp.text)
            else:
                count += 1

        self._logger.info("Stored %s of documents.", count)


class ElasticSearchPredictionsRepoConfig(BaseSettings):
    host: str
    username: str
    password: str
    index: str

    class Config:
        env_prefix = "ES_PREDICTIONS_REPO_"

    def get(self) -> ElasticSearchPredictionsRepo:
        return ElasticSearchPredictionsRepo(
            host=self.host, index=self.index, username=self.username, password=self.password
        )
