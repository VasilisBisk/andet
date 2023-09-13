from typing import Final

import pandas as pd

SPIKE: Final = 1
NOT_SPIKE: Final = 0


class Postprocessor:
    def __init__(self, target_id: str) -> None:
        self.target_id = target_id

    def process(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Process post porcessing."""
        predictions = predictions[predictions.spike == SPIKE]
        predictions["target_id"] = self.target_id
        return predictions
