import datetime
import json
import logging
from typing import Any, Dict, Final, Optional

import pandas as pd

from fvt_ml.ram_abnormal_detection.std_dev.infra.predictions_repo.predictions_repo import (
    PredictionsRepo,
)
from fvt_ml.ram_abnormal_detection.std_dev.infra.raw_data_repo.raw_data_repo import RawDataRepo
from fvt_ml.ram_abnormal_detection.std_dev.model import Model
from fvt_ml.ram_abnormal_detection.std_dev.postprocessor import Postprocessor
from fvt_ml.ram_abnormal_detection.std_dev.preprocessor import Preprocessor

SPIKE: Final[int] = 1
NOT_SPIKE: Final[int] = 0


class StDevApp:
    def __init__(
        self,
        preprocessor: Preprocessor,
        raw_data_repo: RawDataRepo,
        postprocessor: Optional[Postprocessor] = None,
        predictions_repo: Optional[PredictionsRepo] = None,
        model: Optional[Model] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.raw_data_repo = raw_data_repo
        self.predictions_repo = predictions_repo
        self.model = model
        self._logger = logging.getLogger(self.__class__.__name__)

    def infer(
        self, start_date: datetime.date, end_date: datetime.date, target_id: str
    ) -> Optional[pd.DataFrame]:
        if not (self.model and self.predictions_repo and self.postprocessor):
            raise RuntimeError("Model and predictions repo must be specified...")

        raw_data_df = self.raw_data_repo.get_by_id(
            start_date=start_date, end_date=end_date, target_id=target_id
        )

        if raw_data_df is not None:

            self._logger.info("Data for timerange and target have been identified.")

            processed_data = self.preprocessor.process(raw_data_df)
            predictions = self.model.predict(processed_data)

            if predictions.shape[0] == 0:
                self._logger.warning("No predictions calculated.")
            else:
                self._logger.info("Number of predictions calculated: %s", predictions.shape[0])

                predictions = self.postprocessor.process(predictions)

                self.predictions_repo.put(predictions)
                return predictions
        else:
            self._logger.info("No data identified.")
            return None

    @staticmethod
    def _calculate_stats(df: pd.DataFrame, resampling_rule: str) -> pd.DataFrame:
        stats = df.copy()
        stats["day_of_week"] = stats.timestamp.apply(lambda x: x.day_of_week)

        stats_new = []
        for day_num in (0, 1, 2, 3, 4, 5, 6):
            temp = stats[stats.day_of_week == day_num]
            temp = temp.to_dict("records")
            if not temp:
                continue
            day = temp[0]["timestamp"].day

            temp_new = []
            for t in temp:
                t["timestamp"] = t["timestamp"].replace(day=day)
                temp_new.append(t)

            stats_new.extend(temp_new)

        stats_new_df: pd.DataFrame = pd.DataFrame(stats_new)

        stats = (
            stats_new_df.set_index("timestamp")
            .resample(rule=resampling_rule, closed="left", label="left")
            .agg(["mean", "std"])
            .fillna(0)
        )

        stats = stats.reset_index()
        stats["time"] = stats.timestamp.apply(lambda x: x.time())
        stats["time"] = stats.time.apply(lambda x: str(x))
        stats = stats.drop(["timestamp"], axis=1)
        stats.columns = ["mean", "stdev", "day_of_week", "to_drop", "time"]
        stats = stats.drop("to_drop", axis=1)
        stats.day_of_week = stats.day_of_week.astype(int)
        stats["stdev"] = stats["stdev"].apply(lambda x: 0.01 if x == 0.0 else x)
        return stats

    def calculate_stats(
        self, target_id: str, resampling_rule: str, df: pd.DataFrame
    ) -> Dict[str, Any]:
        return {
            "target_id": target_id,
            "resampling_rule": resampling_rule,
            "stats": self._calculate_stats(df, resampling_rule).to_dict("records"),
        }

    def train(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        target_id: str,
        output_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Train the algorithm.

        This menthod is responsible for calculating and storing the stats for the
        model.

        Args:
            start_date: the start date of the training dataset
            end_date: the end date of the training dataset
            target_id: the server that stats have to be calculated
            output_path: the output path of the stats, including the name of the file

        Returns:
            stats: a dictionaty that holds the stats for each timeframe

        """
        raw_data_df = self.raw_data_repo.get_by_id(
            start_date=start_date, end_date=end_date, target_id=target_id
        )

        if raw_data_df is not None:

            self._logger.info("Data for timerange and target have been identified.")

            processed_data = self.preprocessor.process(raw_data_df)
            stats = self.calculate_stats(
                target_id=target_id, resampling_rule="30Min", df=processed_data
            )

            if output_path:

                self._logger.info("Storing stats in path %s", output_path)
                with open(output_path, "w") as f:
                    json.dump(stats, f, indent=4)

            return stats
        else:
            self._logger.info("No data identified.")
            return None
