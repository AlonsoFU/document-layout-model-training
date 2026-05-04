"""Tests for core.predict — pre-annotation pipeline.

Mocks the model and processor; uses small fake images.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from core.predict import predict


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "test"
    (proj / "data" / "pdfs").mkdir(parents=True)
    (proj / "data" / "images" / "doc-A").mkdir(parents=True)
    (proj / "cvat").mkdir(parents=True)
    # Create 2 fake PNGs (real RGB images so PIL.open works)
    from PIL import Image
    for i in (1, 2):
        Image.new("RGB", (100, 200), color=(255, 255, 255)).save(
            proj / "data" / "images" / "doc-A" / f"pagina-{i:03d}.png"
        )
    (proj / "data" / "pdfs" / "doc-A.pdf").write_bytes(b"%PDF")
    (proj / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"slug": "test"},
                "labels": ["Caption", "Picture", "Table", "Text"],
                "postprocess": {
                    "thresholds": {"default": 0.5},
                    "nms_iou": 0.5,
                    "full_page_picture_filter": 0.9,
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return proj


def _fake_load_heron(model_name, device):
    """Returns a (model, processor) pair where post_process_object_detection
    returns one Caption box at score 0.9 per image."""
    import torch

    model = MagicMock()
    processor = MagicMock()

    def _processor_call(images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 100, 100)}

    processor.side_effect = _processor_call

    def _post_process(outputs, target_sizes, threshold):
        return [
            {
                "boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),  # 0 = Caption (DocLayNet order)
            }
        ]

    processor.post_process_object_detection = _post_process

    model_called = MagicMock()
    model.return_value = model_called  # forward() returns object, used in `outputs`

    # Make the processor callable
    processor.__call__ = _processor_call
    return model, processor


def test_predict_writes_coco_with_correct_label_id(fake_project, monkeypatch):
    monkeypatch.setattr("core.predict.load_heron", _fake_load_heron)
    # No torch.cuda needed — _fake_load_heron avoids that path
    predict("test", mode="pre-annotate")

    # Find the produced COCO
    out_files = sorted((fake_project / "cvat" / "pre_annotations").glob("*.json"))
    assert len(out_files) == 1
    coco = json.loads(out_files[0].read_text())

    # Should have 2 images, 2 annotations (one Caption per image)
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 2
    # category_id is the 1-based position of "Caption" in the project's labels list
    # labels = ["Caption", "Picture", "Table", "Text"] -> Caption is id 1
    assert all(ann["category_id"] == 1 for ann in coco["annotations"])
    assert all(ann["score"] == pytest.approx(0.9) for ann in coco["annotations"])


def test_predict_output_pdf_calls_fitz(monkeypatch, tmp_path):
    """Smoke: --output=anotado.pdf path goes through the PDF code branch."""
    pytest.skip("Wired in implementation; see Task 3 step 5 smoke test for actual PDF generation.")


def test_predict_drops_below_threshold(fake_project, monkeypatch):
    """Lower the score and confirm filtering."""
    def _low_score_load(model_name, device):
        m, p = _fake_load_heron(model_name, device)
        import torch

        def _post(outputs, target_sizes, threshold):
            return [{
                "boxes": torch.tensor([[10.0, 10.0, 90.0, 90.0]]),
                "scores": torch.tensor([0.3]),  # below default 0.5
                "labels": torch.tensor([0]),
            }]
        p.post_process_object_detection = _post
        return m, p

    monkeypatch.setattr("core.predict.load_heron", _low_score_load)
    predict("test", mode="pre-annotate")

    out_files = sorted((fake_project / "cvat" / "pre_annotations").glob("*.json"))
    coco = json.loads(out_files[-1].read_text())
    assert coco["annotations"] == []
