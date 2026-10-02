"""GTSRB source, archive joins, and public dataset behavior."""

import csv
import io
import zipfile

import pytest
from PIL import Image as PILImage

from stable_datasets.images import GTSRB
from stable_datasets.schema import ClassLabel, Image, Version
from stable_datasets.splits import Split


_TRAIN_PREFIX = "GTSRB/Final_Training/Images/"
_TEST_PREFIX = "GTSRB/Final_Test/Images/"
_FIELDS = ("Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")


def _builder():
    builder = object.__new__(GTSRB)
    builder.__init__()
    return builder


def _ppm(size):
    buffer = io.BytesIO()
    PILImage.new("RGB", size, (20, 80, 140)).save(buffer, format="PPM")
    return buffer.getvalue()


def _csv(rows, *, labeled):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow((*_FIELDS, *(("ClassId",) if labeled else ())))
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _row(filename, width, height, label=None):
    values = (filename, width, height, 1, 1, width - 2, height - 2)
    return (*values, label) if label is not None else values


def _archives(tmp_path, *, train_rows=None, train_images=None, gt_rows=None):
    train_path = tmp_path / "train.zip"
    images_path = tmp_path / "test_images.zip"
    gt_path = tmp_path / "test_gt.zip"

    if train_rows is None:
        train_rows = {
            2: [_row("00000_00001.ppm", 8, 9, 2), _row("00000_00000.ppm", 7, 8, 2)],
            0: [_row("00000_00000.ppm", 6, 7, 0)],
        }
    with zipfile.ZipFile(train_path, "w") as archive:
        for class_id, rows in train_rows.items():
            prefix = f"{_TRAIN_PREFIX}{class_id:05d}/"
            image_rows = train_images[class_id] if train_images is not None else rows
            written = set()
            for row in reversed(image_rows):
                if row[0] in written:
                    continue
                archive.writestr(prefix + row[0], _ppm((int(row[1]), int(row[2]))))
                written.add(row[0])
            archive.writestr(prefix + f"GT-{class_id:05d}.csv", _csv(rows, labeled=True))

    test_rows = [_row("00002.ppm", 9, 10), _row("00000.ppm", 5, 6), _row("00001.ppm", 7, 8)]
    with zipfile.ZipFile(images_path, "w") as archive:
        for row in test_rows:
            archive.writestr(_TEST_PREFIX + row[0], _ppm((row[1], row[2])))
        archive.writestr(_TEST_PREFIX + "GT-final_test.test.csv", _csv(test_rows, labeled=False))

    if gt_rows is None:
        gt_rows = [_row("00001.ppm", 7, 8, 0), _row("00002.ppm", 9, 10, 7), _row("00000.ppm", 5, 6, 42)]
    with zipfile.ZipFile(gt_path, "w") as archive:
        archive.writestr("GT-final_test.csv", _csv(gt_rows, labeled=True))
    return train_path, images_path, gt_path


def test_metadata_and_source():
    builder = _builder()
    assert GTSRB.VERSION == Version("1.0.0")
    assert set(GTSRB.SOURCE.assets) == {"train", "test_images", "test_ground_truth"}
    assert all("sid.erda.dk/public/archives/" in asset.url for asset in GTSRB.SOURCE.assets.values())
    assert set(builder.info.features) == {"image", "label"}
    assert isinstance(builder.info.features["image"], Image)
    labels = builder.info.features["label"]
    assert isinstance(labels, ClassLabel)
    assert labels.names == [str(i) for i in range(43)]
    assert builder.info.supervised_keys == ("image", "label")
    assert builder.info.license == ""
    assert builder._candidate_splits() == [Split.TRAIN, Split.TEST]


def test_synthetic_dataset_joins_and_reuses_cache(tmp_path, monkeypatch):
    paths = _archives(tmp_path)
    monkeypatch.setattr("stable_datasets.images.gtsrb.bulk_download", lambda *args, **kwargs: paths)
    cache_dir = tmp_path / "processed"
    datasets = GTSRB(split=None, processed_cache_dir=cache_dir, download_dir=tmp_path / "downloads")
    assert set(datasets) == {"train", "test"}
    assert len(datasets["train"]) == 3
    assert len(datasets["test"]) == 3
    assert [datasets["train"][i]["label"] for i in range(3)] == [0, 2, 2]
    assert [datasets["train"][i]["image"].size for i in range(3)] == [(6, 7), (7, 8), (8, 9)]
    assert [datasets["test"][i]["label"] for i in range(3)] == [42, 0, 7]
    assert [datasets["test"][i]["image"].size for i in range(3)] == [(5, 6), (7, 8), (9, 10)]
    for dataset in datasets.values():
        for sample in dataset:
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], PILImage.Image)
            assert sample["image"].mode == "RGB"

    def unexpected_download(*args, **kwargs):
        raise AssertionError("cache reuse should not download")

    monkeypatch.setattr("stable_datasets.images.gtsrb.bulk_download", unexpected_download)
    cached = GTSRB(split=None, processed_cache_dir=cache_dir)
    assert {name: len(dataset) for name, dataset in cached.items()} == {"train": 3, "test": 3}


@pytest.mark.parametrize("bad", ["missing_image", "invalid_label", "duplicate", "wrong_class"])
def test_training_annotation_errors(tmp_path, bad):
    rows = [_row("00000_00000.ppm", 6, 7, 0)]
    if bad == "missing_image":
        rows.append(_row("missing.ppm", 6, 7, 0))
    elif bad == "invalid_label":
        rows = [_row("00000_00000.ppm", 6, 7, 43)]
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "wrong_class":
        rows = [_row("00000_00000.ppm", 6, 7, 1)]
    train_images = {0: rows[:1]} if bad == "missing_image" else None
    train_path, _, _ = _archives(tmp_path, train_rows={0: rows}, train_images=train_images)
    with pytest.raises(
        ValueError,
        match={
            "missing_image": "missing image",
            "invalid_label": "out of range",
            "duplicate": "Duplicate GTSRB annotation",
            "wrong_class": "disagrees with class directory",
        }[bad],
    ):
        list(_builder()._generate_examples(train_path, Split.TRAIN))


@pytest.mark.parametrize("bad", ["missing", "duplicate", "extra", "invalid_label", "metadata_mismatch"])
def test_test_ground_truth_errors(tmp_path, bad):
    rows = [_row("00000.ppm", 5, 6, 42), _row("00001.ppm", 7, 8, 0), _row("00002.ppm", 9, 10, 7)]
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "extra":
        rows.append(_row("00003.ppm", 6, 7, 1))
    elif bad == "invalid_label":
        rows[0] = _row("00000.ppm", 5, 6, 43)
    elif bad == "metadata_mismatch":
        rows[0] = _row("00000.ppm", 6, 6, 42)
    _, images_path, gt_path = _archives(tmp_path, gt_rows=rows)
    with pytest.raises(
        ValueError,
        match={
            "missing": "missing GT rows",
            "duplicate": "Duplicate GTSRB annotation",
            "extra": "extra GT rows",
            "invalid_label": "out of range",
            "metadata_mismatch": "annotations disagree",
        }[bad],
    ):
        list(_builder()._generate_examples(images_path, Split.TEST, gt_path))


@pytest.mark.large
def test_official_final_archives():
    datasets = GTSRB(split=None)
    assert set(datasets) == {"train", "test"}
    assert len(datasets["train"]) == 39209
    assert len(datasets["test"]) == 12630
    for dataset in datasets.values():
        labels = dataset.table.column("label").to_pylist()
        assert set(labels) == set(range(43))
        assert min(labels) == 0 and max(labels) == 42
        sizes = set()
        for index in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[index]
            assert set(sample) == {"image", "label"}
            assert isinstance(sample["image"], PILImage.Image)
            assert sample["image"].mode == "RGB"
            assert all(side > 0 for side in sample["image"].size)
            sizes.add(sample["image"].size)
        assert len(sizes) > 1
