"""Tests for core.lib.cvat — HTTP client for CVAT REST API.

Uses the `responses` library to mock HTTP. No real CVAT server is contacted.
"""
import io
import json
import zipfile
from pathlib import Path

import pytest
import responses

from core.lib.cvat import CvatClient


@pytest.fixture()
def client():
    return CvatClient("http://localhost:8080", auth=("admin", "admin"))


@responses.activate
def test_get_project_by_name_returns_match(client):
    responses.get(
        "http://localhost:8080/api/projects",
        match=[responses.matchers.query_param_matcher({"search": "Foo"})],
        json={
            "count": 1,
            "results": [{"id": 7, "name": "Foo", "tasks": {"count": 0}}],
        },
    )
    p = client.get_project_by_name("Foo")
    assert p == {"id": 7, "name": "Foo", "tasks": {"count": 0}}


@responses.activate
def test_get_project_by_name_returns_none_when_no_exact_match(client):
    responses.get(
        "http://localhost:8080/api/projects",
        match=[responses.matchers.query_param_matcher({"search": "Foo"})],
        json={"count": 1, "results": [{"id": 7, "name": "Foobar"}]},
    )
    assert client.get_project_by_name("Foo") is None


@responses.activate
def test_create_project_with_labels(client):
    responses.post(
        "http://localhost:8080/api/projects",
        json={"id": 42, "name": "Test"},
        status=201,
    )
    pid = client.create_project("Test", ["A", "B"])
    assert pid == 42
    body = json.loads(responses.calls[0].request.body)
    assert body["name"] == "Test"
    assert body["labels"] == [{"name": "A"}, {"name": "B"}]


@responses.activate
def test_create_task(client):
    responses.post(
        "http://localhost:8080/api/tasks",
        json={"id": 99},
        status=201,
    )
    tid = client.create_task("EAF-089-2025", project_id=42)
    assert tid == 99


@responses.activate
def test_upload_data_zip_polls_until_finished(client):
    responses.post(
        "http://localhost:8080/api/tasks/99/data",
        status=202,
    )
    responses.get(
        "http://localhost:8080/api/tasks/99/status",
        json={"state": "Started"},
    )
    responses.get(
        "http://localhost:8080/api/tasks/99/status",
        json={"state": "Finished"},
    )
    buf = io.BytesIO(b"zipcontent")
    client.upload_data(99, buf, poll_interval=0)
    # 3 calls: POST data + 2 GET status
    assert len(responses.calls) == 3


@responses.activate
def test_import_annotations_uses_request_id_polling(client):
    """CVAT current API: POST returns 202 with rq_id; GET /api/requests/{rq_id}."""
    responses.post(
        "http://localhost:8080/api/tasks/99/annotations",
        json={"rq_id": "rq_xyz"},
        status=202,
    )
    responses.get(
        "http://localhost:8080/api/requests/rq_xyz",
        json={"status": "started"},
    )
    responses.get(
        "http://localhost:8080/api/requests/rq_xyz",
        json={"status": "finished"},
    )
    buf = io.BytesIO(b"cocozip")
    client.import_annotations(99, buf, fmt="COCO 1.0", poll_interval=0)
    assert len(responses.calls) == 3


@responses.activate
def test_import_annotations_raises_on_failure(client):
    responses.post(
        "http://localhost:8080/api/tasks/99/annotations",
        json={"rq_id": "rq_bad"},
        status=202,
    )
    responses.get(
        "http://localhost:8080/api/requests/rq_bad",
        json={"status": "failed", "message": "bad coco"},
    )
    with pytest.raises(RuntimeError, match="bad coco"):
        client.import_annotations(99, io.BytesIO(b"x"), fmt="COCO 1.0", poll_interval=0)


@responses.activate
def test_get_task_annotations_count(client):
    responses.get(
        "http://localhost:8080/api/tasks/99/annotations",
        json={"shapes": [{}, {}, {}], "tracks": [], "tags": []},
    )
    n = client.count_task_annotations(99)
    assert n == 3
