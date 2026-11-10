"""Small, polite client for the public iNaturalist v1 API.

iNaturalist asks API users to stay around one request per second, so every
call is throttled and cached on disk. Re-running a script hits the cache
instead of the network.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import requests

BASE_URL = "https://api.inaturalist.org/v1"
USER_AGENT = "FaunaVision-research/0.1 (portfolio project)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class InatClient:
    def __init__(
        self,
        cache_dir: Path,
        min_interval: float = 1.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last_request = 0.0

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        params = params or {}
        key = hashlib.sha1(
            json.dumps([path, sorted(params.items())], default=str).encode()
        ).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=60)
            if resp.status_code in RETRY_STATUS:
                time.sleep(2 ** attempt * 2)
                continue
            resp.raise_for_status()
            data = resp.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            return data
        raise RuntimeError(f"iNaturalist API kept failing for {path} {params}")

    def species_counts(
        self,
        place_id: int,
        iconic_taxa: Iterable[str],
        licenses: Iterable[str],
        min_observations: int,
        per_page: int = 500,
        max_pages: int = 40,
    ) -> list[dict]:
        """Research-grade species counts, most-observed first.

        Stops paging once counts drop below ``min_observations``.
        """
        rows: list[dict] = []
        for page in range(1, max_pages + 1):
            data = self.get(
                "/observations/species_counts",
                {
                    "place_id": place_id,
                    "iconic_taxa": ",".join(iconic_taxa),
                    "quality_grade": "research",
                    "photos": "true",
                    "photo_license": ",".join(licenses),
                    "per_page": per_page,
                    "page": page,
                },
            )
            results = data.get("results", [])
            rows.extend(results)
            if not results or results[-1]["count"] < min_observations:
                break
        return rows

    def taxa(self, ids: Iterable[int], batch_size: int = 30) -> dict[int, dict]:
        """Look up taxa by id (used to find the rank of each ancestor)."""
        ids = sorted(set(ids))
        out: dict[int, dict] = {}
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            data = self.get(
                f"/taxa/{','.join(map(str, chunk))}", {"per_page": batch_size}
            )
            for t in data.get("results", []):
                out[t["id"]] = t
        return out
