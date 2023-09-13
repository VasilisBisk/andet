import abc
import datetime
from typing import Optional

import pandas as pd


class RawDataRepo(abc.ABC):
    @abc.abstractmethod
    def get_by_id(
        self, start_date: datetime.date, end_date: datetime.date, target_id: str
    ) -> Optional[pd.DataFrame]:
        pass
