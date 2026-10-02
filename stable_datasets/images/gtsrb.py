"""Official German Traffic Sign Recognition Benchmark (GTSRB)."""

import csv
import io
import re
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Image, Version
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


_ARCHIVE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
_TRAIN_PREFIX = "GTSRB/Final_Training/Images/"
_TEST_PREFIX = "GTSRB/Final_Test/Images/"
_TEST_ANNOTATIONS = _TEST_PREFIX + "GT-final_test.test.csv"
_TEST_GROUND_TRUTH = "GT-final_test.csv"
_METADATA_FIELDS = ("Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
_TRAIN_IMAGE = re.compile(r"GTSRB/Final_Training/Images/(\d{5})/([^/]+\.ppm)")
_TRAIN_CSV = re.compile(r"GTSRB/Final_Training/Images/(\d{5})/GT-(\d{5})\.csv")


def _annotations(archive, member, *, labeled):
    """Read one official semicolon-delimited CSV, checking its schema and rows."""
    expected = ("Filename", *_METADATA_FIELDS, *(("ClassId",) if labeled else ()))
    with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
        reader = csv.DictReader(text, delimiter=";")
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected GTSRB CSV columns in {member}: {reader.fieldnames!r}")

        rows = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed GTSRB annotation in {member}, line {line_number}")
            filename = row["Filename"]
            if not filename or "/" in filename or "\\" in filename or not filename.endswith(".ppm"):
                raise ValueError(f"Invalid GTSRB filename in {member}, line {line_number}: {filename!r}")
            if filename in rows:
                raise ValueError(f"Duplicate GTSRB annotation for {filename!r} in {member}")
            try:
                values = tuple(int(row[field]) for field in _METADATA_FIELDS)
                label = int(row["ClassId"]) if labeled else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid GTSRB annotation in {member}, line {line_number}") from exc
            width, height, x1, y1, x2, y2 = values
            if not (width > 0 and height > 0 and 0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
                raise ValueError(f"Invalid GTSRB dimensions or ROI for {filename!r} in {member}")
            if labeled and not 0 <= label < 43:
                raise ValueError(f"GTSRB ClassId out of range for {filename!r} in {member}: {label}")
            rows[filename] = (values, label)
    return rows


def _image_members(archive, pattern):
    """Index PPM members without relying on ZIP member order."""
    members = {}
    for member in archive.infolist():
        if not member.filename.lower().endswith(".ppm"):
            continue
        match = pattern.fullmatch(member.filename)
        if match is None:
            raise ValueError(f"Unexpected GTSRB image path: {member.filename}")
        if member.filename in members:
            raise ValueError(f"Duplicate GTSRB image member: {member.filename}")
        members[member.filename] = member
    return members


def _load_image(archive, member, dimensions):
    with archive.open(member) as raw, PILImage.open(raw) as source:
        image = source.convert("RGB")
        image.load()
    if image.size != dimensions:
        raise ValueError(f"GTSRB image dimensions disagree with annotations for {member.filename}")
    return image


class GTSRB(BaseDatasetBuilder):
    """GTSRB classification with the official final train and labeled test splits."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/houbensebastian/GermanTrafficSignBenchmarks",
        assets={
            "train": DownloadInfo(url=f"{_ARCHIVE_URL}/GTSRB_Final_Training_Images.zip"),
            "test_images": DownloadInfo(url=f"{_ARCHIVE_URL}/GTSRB_Final_Test_Images.zip"),
            "test_ground_truth": DownloadInfo(url=f"{_ARCHIVE_URL}/GTSRB_Final_Test_GT.zip"),
        },
        citation=r"""@inproceedings{Stallkamp-IJCNN-2011,
    author = {Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
    title = {The {G}erman {T}raffic {S}ign {R}ecognition {B}enchmark: A multi-class classification competition},
    booktitle = {IEEE International Joint Conference on Neural Networks},
    pages = {1453--1460},
    year = {2011}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description="Official GTSRB final training and test images with 43 traffic sign classes.",
            features=Features({"image": Image(), "label": ClassLabel(num_classes=43)}),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE.homepage,
            citation=self.SOURCE.citation,
            license=self.SOURCE.license,
        )

    def _candidate_splits(self):
        return [Split.TRAIN, Split.TEST]

    def _split_generators(self):
        assets = self._source().assets
        keys = ("train", "test_images", "test_ground_truth")
        paths = bulk_download([assets[key] for key in keys], dest_folder=self._raw_download_dir)
        by_key = dict(zip(keys, paths))
        return [
            SplitGenerator(name=Split.TRAIN, gen_kwargs={"data_path": by_key["train"], "split": Split.TRAIN}),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={
                    "data_path": by_key["test_images"],
                    "ground_truth_path": by_key["test_ground_truth"],
                    "split": Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_path, split, ground_truth_path=None):
        if split == Split.TRAIN:
            yield from self._training_examples(data_path)
        elif split == Split.TEST:
            if ground_truth_path is None:
                raise ValueError("GTSRB test split requires the separate ground-truth archive")
            yield from self._test_examples(data_path, ground_truth_path)
        else:
            raise ValueError(f"Unknown GTSRB split: {split!r}")

    def _training_examples(self, data_path):
        with zipfile.ZipFile(data_path) as archive:
            images = _image_members(archive, _TRAIN_IMAGE)
            annotated = set()
            csv_members = []
            for member in archive.infolist():
                if member.filename.endswith(".csv"):
                    match = _TRAIN_CSV.fullmatch(member.filename)
                    if match is None or match.group(1) != match.group(2):
                        raise ValueError(f"Unexpected GTSRB training CSV path: {member.filename}")
                    csv_members.append(member)
            if len({member.filename for member in csv_members}) != len(csv_members):
                raise ValueError("Duplicate GTSRB training CSV member")

            examples = []
            for member in sorted(csv_members, key=lambda item: item.filename):
                class_id = int(_TRAIN_CSV.fullmatch(member.filename).group(1))
                if not 0 <= class_id < 43:
                    raise ValueError(f"GTSRB class directory out of range: {member.filename}")
                rows = _annotations(archive, member, labeled=True)
                for filename, (dimensions, label) in rows.items():
                    if label != class_id:
                        raise ValueError(f"GTSRB ClassId disagrees with class directory for {filename!r}")
                    image_name = f"{_TRAIN_PREFIX}{class_id:05d}/{filename}"
                    if image_name not in images:
                        raise ValueError(f"GTSRB annotation references missing image: {image_name}")
                    annotated.add(image_name)
                    examples.append((class_id, filename, image_name, dimensions, label))

            if set(images) != annotated:
                raise ValueError(f"GTSRB training images without annotations: {len(set(images) - annotated)}")
            for _, _, image_name, dimensions, label in sorted(examples):
                yield image_name, {"image": _load_image(archive, images[image_name], dimensions[:2]), "label": label}

    def _test_examples(self, data_path, ground_truth_path):
        with zipfile.ZipFile(data_path) as images_archive, zipfile.ZipFile(ground_truth_path) as gt_archive:
            images = _image_members(images_archive, re.compile(r"GTSRB/Final_Test/Images/([^/]+\.ppm)"))
            image_rows = _annotations(images_archive, _TEST_ANNOTATIONS, labeled=False)
            gt_rows = _annotations(gt_archive, _TEST_GROUND_TRUTH, labeled=True)
            image_names = {name.removeprefix(_TEST_PREFIX) for name in images}
            if set(image_rows) != image_names:
                raise ValueError(
                    f"GTSRB test image annotations do not match images: "
                    f"{len(image_names - set(image_rows))} missing rows, {len(set(image_rows) - image_names)} extra rows"
                )
            if set(gt_rows) != image_names:
                raise ValueError(
                    f"GTSRB test ground truth does not match images: "
                    f"{len(image_names - set(gt_rows))} missing GT rows, {len(set(gt_rows) - image_names)} extra GT rows"
                )
            for filename in sorted(image_names):
                dimensions, label = gt_rows[filename]
                if image_rows[filename][0] != dimensions:
                    raise ValueError(f"GTSRB test annotations disagree with ground truth for {filename!r}")
                image_name = _TEST_PREFIX + filename
                yield (
                    image_name,
                    {
                        "image": _load_image(images_archive, images[image_name], dimensions[:2]),
                        "label": label,
                    },
                )
