import abc

import pandas as pd


class PredictionsRepo(abc.ABC):
    @abc.abstractmethod
    def put(self, predictions: pd.DataFrame) -> None:
        pass
