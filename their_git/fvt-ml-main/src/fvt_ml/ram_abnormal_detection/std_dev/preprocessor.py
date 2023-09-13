import pandas as pd


class Preprocessor:
    def __init__(self, resample_rule: str) -> None:
        self.resample_rule = resample_rule

    def process(self, raw_data_df: pd.DataFrame) -> pd.DataFrame:
        """Process the raw data ta useful dataset for the model.

        In a more specific manner, creates buckets of 30 minutes and calculates the
        mean ram usage percentage.

        Returns a dataframe with the mean ram usage, time of the day and day of
        the week.

        """
        df = (
            raw_data_df.set_index("timestamp")
            .resample(rule=self.resample_rule, closed="left", label="left")
            .memory_usage.mean()
            .fillna(0)
            .reset_index()
        )
        df["time"] = df.timestamp.apply(lambda x: x.time())
        df["day_of_week"] = df.timestamp.apply(lambda x: x.day_of_week)
        return df
