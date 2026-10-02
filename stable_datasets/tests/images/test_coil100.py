from collections import Counter, defaultdict
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import COIL100
from stable_datasets.schema import ClassLabel, Value, Version
from stable_datasets.schema import Image as ImageFeature


def test_coil100_metadata():
    info = COIL100._info(object.__new__(COIL100))

    assert COIL100.VERSION == Version("1.0.0")
    assert list(COIL100.SOURCE.assets) == ["train"]
    assert list(info.features) == ["image", "label", "angle"]
    assert isinstance(info.features["image"], ImageFeature)
    assert isinstance(info.features["label"], ClassLabel)
    assert info.features["label"].names == [f"obj{object_id}" for object_id in range(1, 101)]
    assert isinstance(info.features["angle"], Value)
    assert info.features["angle"].dtype == "int32"
    assert info.supervised_keys == ("image", "label")
    assert info.license == ""


@pytest.mark.parametrize("name", ["obj0__0.png", "obj101__0.png", "obj1__7.png", "obj1__0_extra.png"])
def test_coil100_rejects_invalid_views(tmp_path, name):
    archive_path = tmp_path / "invalid.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"coil-100/{name}", b"not an image")

    builder = object.__new__(COIL100)
    with pytest.raises(ValueError, match="COIL-100"):
        list(builder._generate_examples(archive_path, "train"))


@pytest.mark.large
def test_coil100_official_dataset():
    ds = COIL100(split="train")
    assert len(ds) == 7200
    assert set(COIL100(split=None)) == {"train"}

    for index, label, angle in [(0, 0, 0), (71, 0, 355), (72, 1, 0), (7199, 99, 355)]:
        sample = ds[index]
        assert set(sample) == {"image", "label", "angle"}
        assert sample["label"] == label
        assert sample["angle"] == angle
        assert isinstance(sample["label"], int)
        assert isinstance(sample["angle"], int)
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].mode == "RGB"
        assert sample["image"].size == (128, 128)
        assert np.asarray(sample["image"]).dtype == np.uint8

    labels = ds.table.column("label").to_pylist()
    angles = ds.table.column("angle").to_pylist()
    assert set(labels) == set(range(100))
    assert set(angles) == set(range(0, 360, 5))
    assert set(Counter(labels).values()) == {72}
    views = defaultdict(set)
    for label, angle in zip(labels, angles):
        assert angle not in views[label]
        views[label].add(angle)
    assert all(view_angles == set(range(0, 360, 5)) for view_angles in views.values())
