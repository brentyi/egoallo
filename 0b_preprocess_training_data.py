"""Translate data from HuMoR-style npz format to an hdf5-based one.

Due to AMASS licensing, we unfortunately can't re-distribute our preprocessed dataset. If you have questions
or run into issues, please reach out.

To generate the npz format yourself, you can use our updated version of the `process_amass_data.py` script:
- https://gist.github.com/brentyi/a52bfaee37734d30d60e3ab1a092e645

Instructions for running this script can be found in the HuMoR repository:
- https://github.com/davrempe/humor/tree/main/data
"""

import queue
import threading
import time
from pathlib import Path

import h5py
import torch
import torch.cuda
import tyro

from egoallo import fncsmpl
from egoallo.data.amass import EgoTrainingData


def main(
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    data_npz_dir: Path = Path("./data/processed_30fps_no_skating/"),
    output_file: Path = Path("./data/egoalgo_no_skating_dataset.hdf5"),
    output_list_file: Path = Path("./data/egoalgo_no_skating_dataset_files.txt"),
    include_hands: bool = True,
) -> None:
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path)

    assert torch.cuda.is_available()

    task_queue = queue.Queue[Path]()
    for path in list(data_npz_dir.glob("**/*.npz")):
        task_queue.put_nowait(path)

    total_count = task_queue.qsize()
    start_time = time.time()

    output_hdf5 = h5py.File(output_file, "w")
    file_list: list[str] = []

    def worker(device_idx: int) -> None:
        device_body_model = body_model.to("cuda:" + str(device_idx))

        while True:
            try:
                npz_path = task_queue.get_nowait()
            except queue.Empty:
                break

            print(f"Processing {npz_path} on device {device_idx}...")
            train_data = EgoTrainingData.load_from_npz(
                device_body_model, npz_path, include_hands=include_hands
            )

            assert "neutral" in str(npz_path)
            group_name = str(npz_path).rpartition("neutral/")[2]
            print(f"Writing to group {group_name} on {device_idx}...")
            group = output_hdf5.create_group(group_name)
            file_list.append(group_name)

            for k, v in vars(train_data).items():
                # No need to write the mask, which will always be ones when we
                # load from the npz file!
                if k == "mask":
                    continue

                # Chunk into 32 timesteps at a time.
                assert v.dtype == torch.float32
                if v.shape[0] == train_data.T_world_cpf.shape[0]:
                    chunks = (min(32, v.shape[0]),) + v.shape[1:]
                else:
                    assert v.shape[0] == 1
                    chunks = v.shape
                group.create_dataset(k, data=v.numpy(force=True), chunks=chunks)

            print(
                f"Finished ~{total_count - task_queue.qsize()}/{total_count},",
                f"{(total_count - task_queue.qsize())/total_count * 100:.2f}% in",
                f"{time.time() - start_time} seconds",
            )

    workers = [
        threading.Thread(target=worker, args=(i,))
        for i in range(torch.cuda.device_count())
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    output_list_file.write_text("\n".join(file_list))


if __name__ == "__main__":
    tyro.cli(main)
