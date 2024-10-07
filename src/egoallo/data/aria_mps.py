from pathlib import Path
from typing import Literal

import numpy as np
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence


def load_point_cloud_and_find_ground(
    points_path: Path,
    return_points: Literal["all", "filtered", "less_filtered"] = "less_filtered",
) -> tuple[np.ndarray, float]:
    """Load an Aria MPS point cloud and find the ground plane."""

    filtered_points_npz_cache_path = points_path.parent / "_cached_filtered_points.npz"
    less_filtered_points_npz_cache_path = (
        points_path.parent / "_cached_less_filtered_points.npz"
    )

    # Read world points as an Nx3 array.
    if (
        filtered_points_npz_cache_path.exists()
        and less_filtered_points_npz_cache_path.exists()
    ):
        if return_points == "all":
            points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore
        else:
            points_data = None

        print("Loading pre-filtered points")
        filtered_points_data = np.load(filtered_points_npz_cache_path)["points"]
        less_filtered_points_data = np.load(less_filtered_points_npz_cache_path)[
            "points"
        ]
    else:
        points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore

        print("Loading + filtering points")
        assert points_path.exists()
        filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.0001,
            threshold_dep=0.005,
        )
        less_filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.001,
            threshold_dep=0.05,
        )
        filtered_points_data = np.array(
            [x.position_world for x in filtered_points_data]
        )  # type: ignore
        less_filtered_points_data = np.array(
            [x.position_world for x in less_filtered_points_data]
        )

        filtered_points_npz_cache_path.parent.mkdir(exist_ok=True, parents=True)

        np.savez_compressed(filtered_points_npz_cache_path, points=filtered_points_data)
        print("Cached filtered points to", filtered_points_npz_cache_path)
        np.savez_compressed(
            less_filtered_points_npz_cache_path, points=less_filtered_points_data
        )
        print("Cached less filtered points to", less_filtered_points_npz_cache_path)

    assert filtered_points_data.shape == (filtered_points_data.shape[0], 3)

    # RANSAC floor plane.
    # We consider points in the lowest 10% of the point cloud.
    filtered_zs = filtered_points_data[:, 2]
    zs = filtered_zs

    # Median-based outlier dropping, this doesn't work very well.
    # d = np.abs(zs - np.median(zs))
    # mdev = np.median(d)
    # zs = zs[d / mdev < 2.0]

    done = False
    best_z = 0.0
    while not done:
        # Slightly silly outlier dropping that just... works better.
        zs = np.sort(zs)[len(zs) // 10_000 : -len(zs) // 10_000]

        # Get bottom 10% or 15%.
        alpha = 0.1 if filtered_points_data.shape[0] < 10_000 else 0.15
        min_z = np.min(zs)
        max_z = np.max(zs)

        zs = zs[zs <= min_z + (max_z - min_z) * alpha]

        best_inliers = 0
        best_z = 0.0
        for i in range(10_000):
            z = np.random.choice(zs)
            inliers_bool = np.abs(zs - z) < 0.01
            inliers = np.sum(inliers_bool)
            if inliers > best_inliers:
                best_z = z
                best_inliers = inliers

        looser_inliers = np.sum(np.abs(filtered_zs - best_z) <= 0.075)
        if looser_inliers <= 3:
            # If we found a really small group... seems like noise. Let's remove the inlier points and re-compute.
            filtered_zs = filtered_zs[np.abs(filtered_zs - best_z) >= 0.01]
            zs = filtered_zs
        else:
            done = True

    # Re-fit plane to inliers.
    floor_z = float(np.median(zs[np.abs(zs - best_z) < 0.01]))
    if return_points == "filtered":
        return filtered_points_data, floor_z
    elif return_points == "less_filtered":
        return less_filtered_points_data, floor_z
    else:
        assert points_data is not None
        return np.array([x.position_world for x in points_data]), floor_z
