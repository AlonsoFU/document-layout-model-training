"""HTTP client for the CVAT REST API.

Handles auth (basic), project/task CRUD, data upload, COCO import/export,
and the async request-id polling pattern that replaced the deprecated
`action=import_status` endpoint in recent CVAT versions.
"""
from __future__ import annotations

import time
from typing import Any, BinaryIO

import requests


class CvatClient:
    def __init__(self, url: str, auth: tuple[str, str]) -> None:
        self.url = url.rstrip("/")
        self.auth = auth
        self._session = requests.Session()
        self._session.auth = auth

    # ---------- Projects ----------

    def get_project_by_name(self, name: str) -> dict[str, Any] | None:
        r = self._session.get(f"{self.url}/api/projects", params={"search": name})
        r.raise_for_status()
        for p in r.json().get("results", []):
            if p["name"] == name:
                return p
        return None

    def create_project(self, name: str, labels: list[str]) -> int:
        payload = {"name": name, "labels": [{"name": n} for n in labels]}
        r = self._session.post(f"{self.url}/api/projects", json=payload)
        r.raise_for_status()
        return int(r.json()["id"])

    # ---------- Tasks ----------

    def get_task_by_name(self, project_id: int, name: str) -> dict[str, Any] | None:
        r = self._session.get(
            f"{self.url}/api/tasks",
            params={"project_id": project_id, "search": name},
        )
        r.raise_for_status()
        for t in r.json().get("results", []):
            if t["name"] == name and t.get("project_id") == project_id:
                return t
        return None

    def create_task(self, name: str, project_id: int) -> int:
        r = self._session.post(
            f"{self.url}/api/tasks",
            json={"name": name, "project_id": project_id},
        )
        r.raise_for_status()
        return int(r.json()["id"])

    # ---------- Data upload ----------

    def upload_data(
        self,
        task_id: int,
        zip_stream: BinaryIO,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
    ) -> None:
        r = self._session.post(
            f"{self.url}/api/tasks/{task_id}/data",
            files={"client_files[0]": ("images.zip", zip_stream, "application/zip")},
            data={
                "image_quality": 100,
                "use_zip_chunks": "true",
                "sorting_method": "lexicographical",
            },
        )
        r.raise_for_status()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._session.get(f"{self.url}/api/tasks/{task_id}/status").json()
            state = s.get("state", "")
            if state == "Finished":
                return
            if state == "Failed":
                raise RuntimeError(f"task {task_id} data processing failed: {s}")
            time.sleep(poll_interval)
        raise TimeoutError(f"task {task_id} data upload did not finish in {timeout}s")

    # ---------- Annotations ----------

    def import_annotations(
        self,
        task_id: int,
        zip_stream: BinaryIO,
        fmt: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> None:
        r = self._session.post(
            f"{self.url}/api/tasks/{task_id}/annotations",
            params={"format": fmt},
            files={"annotation_file": ("annotations.zip", zip_stream, "application/zip")},
        )
        if r.status_code not in (200, 201, 202):
            raise RuntimeError(f"annotation import HTTP {r.status_code}: {r.text[:500]}")
        rq_id = (r.json() or {}).get("rq_id")
        if not rq_id:
            return  # synchronous response
        self._wait_request(rq_id, poll_interval=poll_interval, timeout=timeout)

    def _wait_request(self, rq_id: str, poll_interval: float, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self._session.get(f"{self.url}/api/requests/{rq_id}")
            r.raise_for_status()
            s = r.json()
            status = s.get("status")
            if status == "finished":
                return
            if status == "failed":
                raise RuntimeError(s.get("message") or f"request {rq_id} failed")
            time.sleep(poll_interval)
        raise TimeoutError(f"request {rq_id} did not finish in {timeout}s")

    def count_task_annotations(self, task_id: int) -> int:
        r = self._session.get(f"{self.url}/api/tasks/{task_id}/annotations")
        r.raise_for_status()
        d = r.json()
        return len(d.get("shapes", [])) + len(d.get("tracks", [])) + len(d.get("tags", []))

    # ---------- Dataset export ----------

    def export_dataset(
        self,
        project_id: int,
        fmt: str = "COCO 1.0",
        save_images: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> bytes:
        """Trigger export and return the downloaded ZIP bytes."""
        r = self._session.post(
            f"{self.url}/api/projects/{project_id}/dataset/export",
            params={"format": fmt, "save_images": str(save_images).lower()},
        )
        r.raise_for_status()
        rq_id = r.json().get("rq_id")
        if not rq_id:
            raise RuntimeError("CVAT did not return rq_id for export")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sr = self._session.get(f"{self.url}/api/requests/{rq_id}")
            sr.raise_for_status()
            s = sr.json()
            status = s.get("status")
            if status == "finished":
                url = s.get("result_url")
                if not url:
                    raise RuntimeError(f"export {rq_id} finished but no result_url")
                dr = self._session.get(url)
                dr.raise_for_status()
                return dr.content
            if status == "failed":
                raise RuntimeError(s.get("message") or f"export {rq_id} failed")
            time.sleep(poll_interval)
        raise TimeoutError(f"export {rq_id} did not finish in {timeout}s")
