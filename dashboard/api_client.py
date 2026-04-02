from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class QueryLensAPI:
    """
    Thin wrapper around the FastAPI endpoints so dashboard views do not embed
    raw URL construction. Centralising the base_url here means switching from
    localhost to a deployed host requires one config change, not a grep through
    every view module.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def stages(self) -> List[Dict[str, Any]]:
        return self._get("/stages")

    def stage_metrics(
        self,
        stage_id: str,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        return self._get(
            f"/stages/{stage_id}/metrics",
            params={"page": page, "page_size": page_size},
        )

    def stage_anomalies(
        self,
        stage_id: str,
        detector_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if detector_type:
            params["detector_type"] = detector_type
        return self._get(f"/stages/{stage_id}/anomalies", params=params)

    def localizations(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        return self._get("/localizations", params={"page": page, "page_size": page_size})

    def localization_detail(self, hypothesis_id: str) -> Dict[str, Any]:
        return self._get(f"/localizations/{hypothesis_id}")

    def healing_actions(
        self,
        outcome: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if outcome:
            params["outcome"] = outcome
        return self._get("/healing/actions", params=params)

    def override_action(
        self,
        hypothesis_id: str,
        operator: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._post(
            f"/healing/actions/{hypothesis_id}/override",
            json={"operator": operator, "reason": reason},
        )

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        resp = self._session.get(f"{self._base}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: Optional[Dict] = None) -> Any:
        resp = self._session.post(f"{self._base}{path}", json=json, timeout=10)
        resp.raise_for_status()
        return resp.json()
