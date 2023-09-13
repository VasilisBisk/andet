from typing import Any, Dict, List

import pandas as pd

from fvt_ml.cpu_spike_detection.std_dev.infra.predictions_repo.predictions_repo import (
    PredictionsRepo,
)


class MockPredictionsRepo(PredictionsRepo):
    def __init__(self) -> None:
        self.db: List[Dict[str, Any]] = []

    def put(self, predictions: pd.DataFrame) -> None:
        self.db.append(predictions.to_dict("records"))
