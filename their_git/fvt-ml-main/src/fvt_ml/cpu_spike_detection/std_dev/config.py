import datetime

from pydantic import BaseSettings, Field

from fvt_ml.cpu_spike_detection.std_dev.infra.predictions_repo.elastic_search_predictions_repo import (
    ElasticSearchPredictionsRepo,
    ElasticSearchPredictionsRepoConfig,
)
from fvt_ml.cpu_spike_detection.std_dev.infra.raw_data_repo.elastic_search_raw_data_repo import (
    ElasticSearchRawDataRepo,
    ElasticSearchRawDataRepoConfig,
)
from fvt_ml.cpu_spike_detection.std_dev.model import Model
from fvt_ml.cpu_spike_detection.std_dev.postprocessor import Postprocessor
from fvt_ml.cpu_spike_detection.std_dev.preprocessor import Preprocessor
from fvt_ml.cpu_spike_detection.std_dev.stats import Stats


class Config(BaseSettings):
    log_level: str = Field(default="INFO")
    target_id: str
    start_date: datetime.date
    end_date: datetime.date
    resample_rule: str = Field(default="30Min")
    raw_data_repo: ElasticSearchRawDataRepoConfig = Field(
        default_factory=ElasticSearchRawDataRepoConfig
    )
    predictions_repo: ElasticSearchPredictionsRepoConfig = Field(
        default_factory=ElasticSearchPredictionsRepoConfig
    )

    def get_model(self) -> Model:
        filename = f"{self.target_id}.json"
        return Model(Stats.from_config(filename))

    def get_preprocessor(self) -> Preprocessor:
        return Preprocessor(self.resample_rule)

    def get_postprocessor(self) -> Postprocessor:
        return Postprocessor(target_id=self.target_id)

    def get_raw_data_repo(self) -> ElasticSearchRawDataRepo:
        return self.raw_data_repo.get()

    def get_predictions_repo(self) -> ElasticSearchPredictionsRepo:
        return self.predictions_repo.get()
