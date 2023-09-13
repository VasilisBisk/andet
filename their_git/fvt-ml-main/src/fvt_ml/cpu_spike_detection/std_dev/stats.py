import datetime
import json
from importlib import resources
from typing import Any, Dict, List

from pydantic import BaseModel, validator


class StatsEntry(BaseModel):
    time: datetime.time
    mean: float
    stdev: float
    day_of_week: int

    @validator("time", pre=True)
    def parse_to_datetime_time(cls, v: str) -> datetime.time:
        return datetime.time.fromisoformat(v)


class Stats(BaseModel):
    target_id: str
    resampling_rule: str
    stats: List[StatsEntry]

    @classmethod
    def load_stats(cls, stats_dict: Dict[str, Any]) -> "Stats":
        return Stats.parse_obj(stats_dict)

    def get_stats_for_day_of_week(self, day_of_week: int) -> List[Dict[str, Any]]:
        output = []
        for s in self.stats:
            if s.day_of_week == day_of_week:
                output.append(s.dict())
        return output

    @classmethod
    def from_config(cls, filename: str) -> "Stats":
        stats = json.loads(
            resources.read_text("fvt_ml_resources.cpu_spike_detection_stdev_model", filename)
        )
        return cls(**stats)
