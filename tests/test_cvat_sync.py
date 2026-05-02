"""Tests for core.cvat_sync — push (creates project + tasks + uploads) and pull (export)."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from core.cvat_sync import _filter_coco_for_task, push


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "test"
    (proj / "data" / "pdfs").mkdir(parents=True)
    (proj / "data" / "images" / "doc-A").mkdir(parents=True)
    (proj / "data" / "images" / "doc-B").mkdir(parents=True)
    # Create 2 fake PNGs per PDF
    for i in (1, 2):
        (proj / "data" / "images" / "doc-A" / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG")
        (proj / "data" / "images" / "doc-B" / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG")
    # Empty PDF placeholders so push iterates them
    (proj / "data" / "pdfs" / "doc-A.pdf").write_bytes(b"%PDF")
    (proj / "data" / "pdfs" / "doc-B.pdf").write_bytes(b"%PDF")
    (proj / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"slug": "test"},
                "cvat": {"project_name": "Test Project", "url": "http://localhost:8080"},
                "labels": ["A", "B", "C"],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return proj


def test_push_creates_project_and_tasks_and_uploads(fake_project, monkeypatch):
    client = MagicMock()
    client.get_project_by_name.return_value = None
    client.create_project.return_value = 42
    client.get_task_by_name.return_value = None
    client.create_task.side_effect = [101, 102]

    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)

    push("test", coco_path=None)

    client.create_project.assert_called_once_with("Test Project", ["A", "B", "C"])
    assert client.create_task.call_count == 2
    assert client.upload_data.call_count == 2
    client.import_annotations.assert_not_called()


def test_push_skips_existing_project_and_tasks_with_images(fake_project, monkeypatch):
    client = MagicMock()
    client.get_project_by_name.return_value = {"id": 42, "name": "Test Project"}
    client.get_task_by_name.side_effect = [
        {"id": 101, "size": 2},  # already has 2 images
        {"id": 102, "size": 0},  # empty, needs upload
    ]
    client.count_task_annotations.return_value = 0

    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)

    push("test", coco_path=None)

    client.create_project.assert_not_called()
    client.create_task.assert_not_called()
    assert client.upload_data.call_count == 1  # only doc-B uploaded


def test_filter_coco_for_task_renumbers_ids_and_strips_suffix():
    coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "A"}],
        "images": [
            {"id": 1, "file_name": "pagina-001.png", "width": 10, "height": 20},
            {"id": 2, "file_name": "pagina-002.png", "width": 10, "height": 20},
            {"id": 3, "file_name": "pagina-001_1.png", "width": 10, "height": 20},
            {"id": 4, "file_name": "pagina-002_1.png", "width": 10, "height": 20},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"id": 2, "image_id": 3, "category_id": 1, "bbox": [0, 0, 5, 5]},
        ],
    }
    out = _filter_coco_for_task(
        coco, image_filenames={"pagina-001.png", "pagina-002.png"}, strip_suffix=False
    )
    assert len(out["images"]) == 2
    assert {i["file_name"] for i in out["images"]} == {"pagina-001.png", "pagina-002.png"}
    assert out["images"][0]["id"] == 1
    assert out["images"][1]["id"] == 2

    out2 = _filter_coco_for_task(
        coco, image_filenames={"pagina-001.png", "pagina-002.png"}, strip_suffix=True
    )
    # strip_suffix removes _1 from source filenames before matching
    assert len(out2["images"]) == 2
    names = {i["file_name"] for i in out2["images"]}
    assert names == {"pagina-001.png", "pagina-002.png"}
