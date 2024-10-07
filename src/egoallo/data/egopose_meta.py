from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


@dataclass(frozen=True)
class TakeMeta:
    uids_all: tuple[str, ...]

    name_from_uid: dict[str, str]
    uid_from_name: dict[str, str]
    meta_from_uid: dict[str, dict[str, Any]]

    # train only has trajectories with egopose annotation
    uids_from_split: dict[
        Literal["train", "val", "test", "no_annotations"], tuple[str, ...]
    ]
    type_from_uid: dict[str, Literal["physical", "procedural"]]

    @classmethod
    def load(cls, dataset_path: Path) -> TakeMeta:
        takes_json_path = dataset_path / "takes.json"
        meta_from_uid: dict[str, dict[str, Any]] = {
            take_meta["take_uid"]: take_meta
            for take_meta in json.loads(takes_json_path.read_text())
        }
        name_from_uid = {uid: meta["take_name"] for uid, meta in meta_from_uid.items()}
        uids_from_split = {
            split: cls._get_split_take_uids(dataset_path, split, name_from_uid)
            for split in ("train", "val", "test")
        }
        uids_from_split["no_annotations"] = cls._filter_uids(
            list(
                # uids_from_split["test"]
                set(name_from_uid.keys())
                - set(
                    #     uids_from_split["train"]
                    uids_from_split["test"]
                    #     + uids_from_split["val"]
                )
            ),
            name_from_uid,
        )
        return TakeMeta(
            uids_all=tuple(name_from_uid.keys()),
            name_from_uid=name_from_uid,
            uid_from_name={v: k for k, v in name_from_uid.items()},
            uids_from_split=uids_from_split,  # type: ignore
            meta_from_uid=meta_from_uid,
            type_from_uid={
                take_uid: cls._get_take_type(name_from_uid[take_uid])
                for take_uid in name_from_uid.keys()
            },
        )

    @staticmethod
    def _filter_uids(
        uids: Sequence[str], name_from_uid: dict[str, str]
    ) -> tuple[str, ...]:
        # Filter out things like climbing and bouldering...
        test_keywords = {
            "basketball",
            "bike",
            "cooking",
            "covid",
            "cpr",
            "dance",
            "guitar",
            "music",
            "pcr",
            "piano",
            "soccer",
            "violin",
        }
        return tuple(
            uid
            for uid in uids
            if any(
                map(
                    lambda keyword: keyword in name_from_uid[uid].lower(), test_keywords
                )
            )
        )

    @staticmethod
    def _get_split_take_uids(
        dataset_path: Path,
        split: Literal["train", "val", "test"],
        name_from_uid: dict[str, str],
    ) -> tuple[str, ...]:
        ego_pose_split_dir = dataset_path / "annotations" / "ego_pose" / split
        camera_pose_dir = ego_pose_split_dir / "camera_pose"
        camera_pose_take_uids = map(lambda x: x.stem, camera_pose_dir.iterdir())

        return TakeMeta._filter_uids(list(camera_pose_take_uids), name_from_uid)

    @staticmethod
    def _get_take_type(take_name: str) -> Literal["physical", "procedural"]:
        found_keyword: str | None = None
        take_type: Literal["physical", "procedural", None] = None
        for keyword in (
            "dance",
            "soccer",
            "basketball",
            "piano",
            "guitar",
            "violin",
            "bouldering",
            "music",
            "rockclimbing",
        ):
            if keyword in take_name.lower():
                assert take_type is None
                take_type = "physical"
                found_keyword = keyword
        for keyword in (
            "cpr",
            "pcr",
            "cooking",
            "covid",
            "bike",
            "sushi",
            "salad",
            "samosa",
            "omelet",
        ):
            if keyword in take_name.lower():
                assert take_type is None
                take_type = "procedural"
                found_keyword = keyword
        assert take_type is not None, f"Could not detect take type for {take_name}"
        assert found_keyword is not None
        return take_type
