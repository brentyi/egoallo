# egoallo

**[Project page](https://egoallo.github.io/) &bull;
[arXiv](https://arxiv.org/abs/2410.03665)**

Code release for our preprint:

<table><tr><td>
    Brent Yi<sup>1</sup>, Vickie Ye<sup>1</sup>, Maya Zheng<sup>1</sup>, Yunqi Li<sup>2</sup>, Lea M&uuml;ller<sup>1</sup>, Georgios Pavlakos<sup>3</sup>, Yi Ma<sup>1</sup>, Jitendra Malik<sup>1</sup>, and Angjoo Kanazawa<sup>1</sup>.
    <strong>Estimating Body and Hand Motion in an Ego-sensed World.</strong>
    arXiV, 2024.
</td></tr>
</table>
<sup>1</sup><em>UC Berkeley</em>, <sup>2</sup><em>ShanghaiTech</em>, <sup>3</sup><em>UT Austin</em>

---

## Updates

- **Oct 7, 2024:** Initial release. (training code, core implementation details)
- **Oct 14, 2024:** Added model checkpoint, dataset preprocessing, inference, and visualization scripts.

## Overview

**TLDR;** We use egocentric SLAM poses and images to estimate 3D human body pose, height, and hands.

https://github.com/user-attachments/assets/7d28e07f-ab83-4749-ac6b-abe692d9ba20

This repository is structured as follows:

```
.
├── download_checkpoint_and_data.sh
│                            - Download model checkpoint and sample data.
├── 0_preprocess_training_data.py
│                            - Preprocessing script for training datasets.
├── 1_train_motion_prior.py
│                            - Training script for motion diffusion model.
├── 2_run_hamer_on_vrs.py
│                            - Run HaMeR on inference data (expects Aria VRS).
├── 3_aria_inference.py
│                            - Run full pipeline on inference data.
├── 4_visualize_outputs.py
│                            - Visualize outputs from inference.
├── 5_eval_body_metrics.py
│                            - Compute and print body estimation accuracy metrics.
│
├── src/egoallo/
│   ├── data/                - Dataset utilities.
│   ├── transforms/          - SO(3) / SE(3) transformation helpers.
│   └── *.py                 - All core implementation.
│
└── pyproject.toml          - Python dependencies/package metadata.
```

## Getting started

EgoAllo requires Python 3.12 or newer.

1. **Clone the repository.**
   ```bash
   git clone https://github.com/brentyi/egoallo.git
   ```
2. **Install general dependencies.**
   ```bash
   cd egoallo
   pip install -e .
   ```
3. **Download+unzip model checkpoint and sample data.**

   ```bash
   bash download_checkpoint_and_data.sh
   ```

   You can also download the zip files manually: here are links to the [checkpoint](https://drive.google.com/file/d/14bDkWixFgo3U6dgyrCRmLoXSsXkrDA2w/view?usp=drive_link) and [example trajectories](https://drive.google.com/file/d/14zQ95NYxL4XIT7KIlFgAYTPCRITWxQqu/view?usp=drive_link).

4. **Download the SMPL-H model file.**

   You can find the "Extended SMPL+H model" from the [MANO project webpage](https://mano.is.tue.mpg.de/).
   Our scripts assumes an npz file located at `./data/smplh/neutral/model.npz`, but this can be overridden at the command-line (`--smplh-npz-path {your path}`).

5. **Visualize model outputs.**

   The example trajectories directory includes example outputs from our model. You can visualize them with:

   ```bash
   python 4_visualize_outputs.py --search-root-dir ./egoallo_example_trajectories
   ```

## Running inference

1. **Installing inference dependencies.**

   Our guidance optimization uses a Levenberg-Marquardt optimizer that's implemented in JAX. If you want to run this on an NVIDIA GPU, you'll need to install JAX with CUDA support:

   ```bash
   # Also see: https://jax.readthedocs.io/en/latest/installation.html
   pip install -U "jax[cuda12]"
   ```

   You'll also need [jaxls](https://github.com/brentyi/jaxls):

   ```bash
   pip install git+https://github.com/brentyi/jaxls.git
   ```

2. **Running inference on example data.**

   Here's an example command for running EgoAllo on the "coffeemachine" sequence:

   ```bash
   python 3_aria_inference.py --traj-root ./egoallo_example_trajectories/coffeemachine
   ```

   You can run `python 3_aria_inference.py --help` to see the full list of options.

3. **Running inference on your own data.**

   To run inference on your own data, you can copy the structure of the example trajectories. The key files are:

   - A VRS file from Project Aria, which contains calibrations and images.
   - SLAM outputs from Project Aria's MPS: `closed_loop_trajectory.csv` and `semidense_points.csv.gz`.
   - (optional) HaMeR outputs, which we save to a `hamer_outputs.pkl`.
   - (optional) Project Aria wrist and palm tracking outputs.

4. **Running HaMeR on your own data.**

   To generate the `hamer_outputs.pkl` file, you'll need to install [hamer_helper](https://github.com/brentyi/hamer_helper).

   Then, as an example for running on our coffeemachine sequence:

   ```bash
   python 2_run_hamer_on_vrs.py --traj-root ./egoallo_example_trajectories/coffeemachine
   ```

## Status

This repository currently contains:

- `egoallo` package, which contains reference training and sampling implementation details.
- Training script.
- Model checkpoints.
- Dataset preprocessing script.
- Inference script.
- Visualization script.
- Setup instructions.

While we've put effort into cleaning up our code for release, this is research
code and there's room for improvement. If you have questions or comments,
please reach out!
