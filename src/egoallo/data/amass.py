from pathlib import Path
from typing import Any, Literal, assert_never, cast

import h5py
import numpy as np
import torch
import torch.utils
import torch.utils.data

from .dataclass import EgoTrainingData

AMASS_SPLITS = {
    "train": [
        "ACCAD",
        "BMLhandball",
        "BMLmovi",
        "BioMotionLab_NTroje",
        "CMU",
        "DFaust_67",
        "DanceDB",
        "EKUT",
        "Eyes_Japan_Dataset",
        "KIT",
        "MPI_Limits",
        "TCD_handMocap",
        "TotalCapture",
    ],
    "val": [
        "HumanEva",
        "MPI_HDM05",
        "MPI_mosh",
        "SFU",
    ],
    "test": [
        "Transitions_mocap",
        "SSM_synced",
    ],
    "test_humor": [
        # HuMoR splits are different...
        "Transitions_mocap",
        "HumanEva",
    ],
    # This is just used for debugging / overfitting...
    "just_humaneva": [
        "HumanEva",
    ],
}


class EgoAmassHdf5Dataset(torch.utils.data.Dataset[EgoTrainingData]):
    """Dataset which loads from our preprocessed hdf5 file.

    Args:
        hdf5_path: Path to the HDF5 file containing the dataset.
        file_list_path: Path to the file containing the list of NPZ files in the dataset.
        splits: List of splits to include in the dataset.
        subseq_len: Length of subsequences to sample from the dataset.
        cache_files: Whether to cache the entire dataset in memory.
        deterministic_slices: Set to True to always use the same slices. This
            is useful for reproducible eval.
    """

    def __init__(
        self,
        hdf5_path: Path,
        file_list_path: Path,
        splits: tuple[
            Literal["train", "val", "test", "test_humor", "just_humaneva"], ...
        ],
        subseq_len: int,
        cache_files: bool,
        slice_strategy: Literal[
            "deterministic", "random_uniform_len", "random_variable_len"
        ],
        min_subseq_len: int | None = None,
        random_variable_len_proportion: float = 0.3,
        random_variable_len_min: int = 16,
    ) -> None:
        datasets = []
        for split in set(splits):
            datasets.extend(AMASS_SPLITS[split])

        self._slice_strategy: Literal[
            "deterministic", "random_uniform_len", "random_variable_len"
        ] = slice_strategy
        self._random_variable_len_proportion = random_variable_len_proportion
        self._random_variable_len_min = random_variable_len_min
        self._hdf5_path = hdf5_path

        with h5py.File(self._hdf5_path, "r") as hdf5_file:
            self._groups = [
                p
                for p in file_list_path.read_text().splitlines()
                if p.partition("/")[0] in datasets
                and cast(
                    h5py.Dataset,
                    cast(h5py.Group, hdf5_file[p])["T_world_root"],
                ).shape[0]
                >= (subseq_len if min_subseq_len is None else min_subseq_len)
                # These datasets had weird joint positions in the original
                # version of the processed data. They should be fine now.
                # and not p.endswith("KIT/317/run05_poses.npz")
                # and not p.endswith("KIT/424/run02_poses.npz")
            ]
            self._subseq_len = subseq_len
            assert len(self._groups) > 0
            assert len(cast(h5py.Group, hdf5_file[self._groups[0]]).keys()) > 0

            # Number of subsequences we would have to sample in order to see each timestep once.
            # This is an underestimate, since sampled subsequences can overlap.
            self._approximated_length = (
                sum(
                    cast(
                        h5py.Dataset, cast(h5py.Group, hdf5_file[g])["T_world_root"]
                    ).shape[0]
                    for g in self._groups
                )
                // subseq_len
            )

        self._cache: dict[str, dict[str, Any]] | None = {} if cache_files else None

    def __getitem__(self, index: int) -> EgoTrainingData:
        group_index = index % len(self._groups)
        slice_index = index // len(self._groups)
        del index

        # Get group corresponding to a single NPZ file.
        group = self._groups[group_index]

        # We open the file only if we're not loading from the cache.
        hdf5_file = None

        if self._cache is not None:
            if group not in self._cache:
                hdf5_file = h5py.File(self._hdf5_path, "r")
                assert hdf5_file is not None
                self._cache[group] = {
                    k: np.array(v)
                    for k, v in cast(h5py.Group, hdf5_file[group]).items()
                }
            npz_group = self._cache[group]
        else:
            hdf5_file = h5py.File(self._hdf5_path, "r")
            npz_group = hdf5_file[group]
            assert isinstance(npz_group, h5py.Group)

        total_t = cast(h5py.Dataset, npz_group["T_world_root"]).shape[0]
        assert total_t >= self._subseq_len

        # Determine slice indexing.
        mask = torch.ones(self._subseq_len, dtype=torch.bool)
        if self._slice_strategy == "deterministic":
            # A deterministic, non-overlapping slice.
            valid_start_indices = total_t - self._subseq_len
            start_t = (
                (slice_index * self._subseq_len) % valid_start_indices
                if valid_start_indices > 0
                else 0
            )
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_uniform_len":
            # Sample a random slice. Ideally we could make this more reproducible...
            start_t = np.random.randint(0, total_t - self._subseq_len + 1)
            end_t = start_t + self._subseq_len
        elif self._slice_strategy == "random_variable_len":
            # Sample a random slice. Ideally we could make this more reproducible...
            random_subseq_len = min(
                # With 30% likelihood, sample a shorter subsequence.
                (
                    np.random.randint(self._random_variable_len_min, self._subseq_len)
                    if np.random.random() < self._random_variable_len_proportion
                    # Otherwise, use the full subsequence.
                    else self._subseq_len
                ),
                total_t,
            )
            start_t = np.random.randint(0, total_t - random_subseq_len + 1)
            end_t = start_t + random_subseq_len
            mask[random_subseq_len:] = False
        else:
            assert_never(self._slice_strategy)

        # Read slices of the dataset.
        kwargs: dict[str, Any] = {}
        for k in npz_group.keys():
            # Possibly saved in the hdf5 file, but we don't need/want to read them.
            if k == "joints_wrt_world":
                continue

            v = npz_group[k]
            assert isinstance(k, str)
            assert isinstance(v, (h5py.Dataset, np.ndarray))
            if k == "betas":
                assert v.shape == (1, 16)
                array = v[:]
            else:
                assert v.shape[0] == total_t
                array = v[start_t:end_t]

            # Pad to subsequence length.
            if array.shape[0] != self._subseq_len:
                array = np.concatenate(
                    [
                        array,
                        # It's important to not pad with np.zeros() here, because that
                        # results in invalid rotations that produce NaNs.
                        np.repeat(
                            array[-1:,], self._subseq_len - array.shape[0], axis=0
                        ),
                    ],
                    axis=0,
                )
            kwargs[k] = torch.from_numpy(array)
        kwargs["mask"] = mask

        # Older versions of the processed dataset don't have hands.
        if "hand_quats" not in kwargs:
            kwargs["hand_quats"] = None

        # Close the file if we opened it.
        if hdf5_file is not None:
            hdf5_file.close()

        return EgoTrainingData(**kwargs)

    def __len__(self) -> int:
        return self._approximated_length
