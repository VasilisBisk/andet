import datetime
from typing import Any, Dict, Optional

import pandas as pd

from fvt_ml.cpu_spike_detection.std_dev.infra.raw_data_repo.raw_data_repo import RawDataRepo


class MockRawDataRepo(RawDataRepo):
    def __init__(self) -> None:
        self.db: Dict[str, Any] = {}

    def get_by_id(
        self, start_date: datetime.date, end_date: datetime.date, target_id: str
    ) -> Optional[pd.DataFrame]:
        return None
