import datetime
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from elasticsearch import Elasticsearch
from pydantic import BaseSettings, Field

from fvt_ml.cpu_spike_detection.std_dev.infra.raw_data_repo.raw_data_repo import RawDataRepo


class ElasticSearchRawDataRepo(RawDataRepo):
    def __init__(self, client: Elasticsearch, index: str, keep_alive: str, size: int) -> None:
        self.client = client
        self.index = index
        self.size = size
        self.keep_alive = keep_alive
        self._logger = logging.getLogger(self.__class__.__name__)

    def _open_point_in_time(self) -> Dict[str, Any]:
        return self.client.open_point_in_time(index=self.index, keep_alive=self.keep_alive)

    def _close_point_in_time(self, pit: Dict[str, str]) -> None:
        self.client.close_point_in_time(pit)

    @staticmethod
    def _build_query(
        start_date: datetime.date,
        end_date: datetime.date,
        target_id: str,
        pit: Optional[Dict[str, str]] = None,
        search_after: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:

        end_date_ = end_date
        if start_date == end_date_:
            end_date_ += datetime.timedelta(days=1)

        query = {
            "_source": {"includes": ["@timestamp", "host.cpu.usage", "_shard_doc"]},
            "sort": [
                {"@timestamp": {"order": "asc", "format": "strict_date_optional_time_nanos"}},
                {"_shard_doc": "desc"},
            ],
            "query": {
                "bool": {
                    "must": [{"match": {"host.name": target_id}}],
                    "filter": {"range": {"@timestamp": {"gte": start_date, "lt": end_date_}}},
                }
            },
        }
        if pit:
            query["pit"] = pit

        if search_after:
            query["search_after"] = search_after

        return query

    def _search(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.search(body=query, size=self.size)

    @staticmethod
    def _format_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_data = []
        for d in data:
            timestamp = d.get("@timestamp")
            host_cpu_usage = d.get("host", {}).get("cpu", {}).get("usage", None)

            if timestamp and host_cpu_usage:
                formatted_data.append(
                    {
                        "timestamp": datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"),
                        "host_cpu_usage": float(host_cpu_usage),
                    }
                )

        return formatted_data

    @staticmethod
    def _extract_pit_from_resp(resp: Dict[str, Any]) -> Dict[str, str]:
        return {"id": resp["pit_id"]}

    @staticmethod
    def _extract_data_from_resp(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [d["_source"] for d in resp["hits"]["hits"]]

    @staticmethod
    def _extract_search_after_values_from_resp(resp: Dict[str, Any]) -> List[Any]:
        return resp["hits"]["hits"][-1]["sort"]

    def get_by_id(
        self, start_date: datetime.date, end_date: datetime.date, target_id: str
    ) -> Optional[pd.DataFrame]:

        self._logger.info("Opening PIT...")
        pit = self._open_point_in_time()

        query = self._build_query(
            start_date=start_date, end_date=end_date, target_id=target_id, pit=pit
        )

        data = []

        self._logger.debug("Initial query: %s", query)

        count_ = 0
        while True:
            count_ += 1
            resp = self._search(query=query)
            if resp:
                extracted_data = self._extract_data_from_resp(resp)
                if not extracted_data:
                    break
                else:
                    data.extend(extracted_data)
                pit = self._extract_pit_from_resp(resp)

                search_after = self._extract_search_after_values_from_resp(resp)
                self._logger.debug("Search after: %s", search_after)

                query = self._build_query(start_date, end_date, target_id, pit, search_after)
                self._logger.debug("Query: %s", query)
            else:
                break

        self._close_point_in_time(pit)
        self._logger.info("Closing PIT...")
        self._logger.info("Queried number of times: %s", count_)

        return pd.DataFrame(self._format_data(data)) if data else None


class ElasticSearchRawDataRepoConfig(BaseSettings):
    host: str
    username: str
    password: str
    index: str
    keep_alive: str = Field(default="1m")
    size: int = Field(default=10, le=10000)

    class Config:
        env_prefix = "ES_RAW_DATA_REPO_"

    def get(self) -> ElasticSearchRawDataRepo:
        client = Elasticsearch(hosts=self.host, http_auth=(self.username, self.password))
        return ElasticSearchRawDataRepo(
            client=client, index=self.index, keep_alive=self.keep_alive, size=self.size
        )
