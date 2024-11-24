import hashlib
import json
import logging
import os
from typing import Any, Dict

import CACHE_DIR

logger = logging.getLogger(__name__)


class APICaller:
    def __init__(self, cache_name: str = "default", max_retries: int = 3, retry_delay: int = 2):
        self.cache_dir = os.path.join(CACHE_DIR, cache_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _generate_hash(self, data: Any) -> str:
        data_str = str(data)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _cache_path(self, data_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{data_hash}.cache")

    def _is_cached(self, data_hash: str) -> bool:
        return os.path.exists(self._cache_path(data_hash))

    def _write_cache(self, data_hash: str, response: Dict[str, Any]) -> None:
        with open(self._cache_path(data_hash), "w") as cache_file:
            json.dump(response, cache_file)

    def _read_cache(self, data_hash: str) -> Dict[str, Any]:
        with open(self._cache_path(data_hash), "r") as cache_file:
            return json.load(cache_file)
