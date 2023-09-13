import pandas as pd

from fvt_ml.cpu_spike_detection.std_dev.stats import Stats


class Model:
    def __init__(self, stats: Stats):
        self.stats = stats

    @staticmethod
    def _predict(value: float, mean: float, stdev: float) -> int:
        z = (value - mean) / stdev
        return 1 if z >= 3 else 0

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        output = []
        days = data.day_of_week.unique().tolist()
        for day in days:
            stats = self.stats.get_stats_for_day_of_week(day)
            data_entries = data[data.day_of_week == day].to_dict("records")
            for data_entry in data_entries:
                for stat_entry in stats:
                    if data_entry["time"] == stat_entry["time"]:
                        temp = {
                            "host_cpu_usage": data_entry["host_cpu_usage"],
                            "time": data_entry["time"],
                            "mean": stat_entry["mean"],
                            "stdev": stat_entry["stdev"],
                            "timestamp_id": data_entry["timestamp"],
                        }
                        output.append(temp)

        df = pd.DataFrame(output)
        df["spike"] = df.apply(
            lambda x: self._predict(x["host_cpu_usage"], x["mean"], x["stdev"]), axis=1
        )
        return df[["timestamp_id", "spike", "host_cpu_usage", "mean", "stdev"]]
