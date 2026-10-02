"""Columbia Object Image Library (COIL-100)."""

import re
import zipfile
from pathlib import PurePosixPath

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Value, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


_IMAGE_NAME = re.compile(r"obj([1-9][0-9]{0,2})__(0|[1-9][0-9]{0,2})\.png")
_EXPECTED_VIEWS = {(object_id, angle) for object_id in range(1, 101) for angle in range(0, 360, 5)}


class COIL100(BaseDatasetBuilder):
    """100 physical objects, each photographed at 72 viewing angles."""

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php",
        assets={
            "train": DownloadInfo(
                url="https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
            )
        },
        citation="""@techreport{nene1996coil100,
  author = {Nene, Sameer A. and Nayar, Shree K. and Murase, Hiroshi},
  title = {Columbia Object Image Library (COIL-100)},
  institution = {Columbia University},
  number = {CUCS-006-96},
  month = feb,
  year = {1996}
}""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "COIL-100 contains 7,200 RGB images of 100 physical objects at 128x128 resolution: "
                "72 views per object in 5-degree increments. The single train split contains the "
                "complete official dataset; Columbia provides no predefined classification split."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=[f"obj{object_id}" for object_id in range(1, 101)]),
                    "angle": Value("int32"),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        with zipfile.ZipFile(data_path) as archive:
            views = {}
            for member in archive.infolist():
                if member.is_dir() or PurePosixPath(member.filename).suffix.lower() != ".png":
                    continue
                match = _IMAGE_NAME.fullmatch(PurePosixPath(member.filename).name)
                if match is None:
                    raise ValueError(f"Unexpected COIL-100 image filename: {member.filename}")
                view = (int(match[1]), int(match[2]))
                if view not in _EXPECTED_VIEWS or view in views:
                    raise ValueError(f"Invalid or duplicate COIL-100 view: {member.filename}")
                views[view] = member

            if views.keys() != _EXPECTED_VIEWS:
                missing = _EXPECTED_VIEWS - views.keys()
                raise ValueError(f"COIL-100 archive is missing {len(missing)} expected views")

            for index, ((object_id, angle), member) in enumerate(sorted(views.items())):
                with archive.open(member) as image_file, PILImage.open(image_file) as source_image:
                    image = source_image.convert("RGB")
                if image.size != (128, 128):
                    raise ValueError(f"Unexpected COIL-100 image size in {member.filename}: {image.size}")
                yield index, {"image": image, "label": object_id - 1, "angle": angle}
