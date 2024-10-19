"""Utilities for writing training scripts."""

import dataclasses
import pdb
import signal
import subprocess
import sys
import time
import traceback as tb
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Protocol,
    Sized,
    get_type_hints,
    overload,
)

import torch


def flattened_hparam_dict_from_dataclass(
    dataclass: Any, prefix: str | None = None
) -> Dict[str, Any]:
    """Convert a config object in the form of a nested dataclass into a
    flattened dictionary, for use with Tensorboard hparams."""
    assert dataclasses.is_dataclass(dataclass)
    cls = type(dataclass)
    hints = get_type_hints(cls)

    output = {}
    for field in dataclasses.fields(dataclass):
        field_type = hints[field.name]
        value = getattr(dataclass, field.name)
        if dataclasses.is_dataclass(field_type):
            inner = flattened_hparam_dict_from_dataclass(value, prefix=None)
            inner = {".".join([field.name, k]): v for k, v in inner.items()}
            output.update(inner)
        # Cast to type supported by tensorboard hparams.
        elif isinstance(value, (int, float, str, bool, torch.Tensor)):
            output[field.name] = value
        else:
            output[field.name] = str(value)

    if prefix is None:
        return output
    else:
        return {f"{prefix}.{k}": v for k, v in output.items()}


def pdb_safety_net():
    """Attaches a "safety net" for unexpected errors in a Python script.

    When called, PDB will be automatically opened when either (a) the user hits Ctrl+C
    or (b) we encounter an uncaught exception. Helpful for bypassing minor errors,
    diagnosing problems, and rescuing unsaved models.
    """

    # Open PDB on Ctrl+C
    def handler(sig, frame):
        pdb.set_trace()

    signal.signal(signal.SIGINT, handler)

    # Open PDB when we encounter an uncaught exception
    def excepthook(type_, value, traceback):  # pragma: no cover (impossible to test)
        tb.print_exception(type_, value, traceback, limit=100)
        pdb.post_mortem(traceback)

    sys.excepthook = excepthook


class SizedIterable[ContainedType](Iterable[ContainedType], Sized, Protocol):
    """Protocol for objects that define both `__iter__()` and `__len__()` methods.

    This is particularly useful for managing minibatches, which can be iterated over but
    only in order due to multiprocessing/prefetching optimizations, and for which length
    evaluation is useful for tools like `tqdm`."""


@dataclasses.dataclass
class LoopMetrics:
    counter: int
    iterations_per_sec: float
    time_elapsed: float


@overload
def range_with_metrics(stop: int, /) -> SizedIterable[LoopMetrics]: ...


@overload
def range_with_metrics(start: int, stop: int, /) -> SizedIterable[LoopMetrics]: ...


@overload
def range_with_metrics(
    start: int, stop: int, step: int, /
) -> SizedIterable[LoopMetrics]: ...


def range_with_metrics(*args: int) -> SizedIterable[LoopMetrics]:
    """Light wrapper for `fifteen.utils.loop_metric_generator()`, for use in place of
    `range()`. Yields a LoopMetrics object instead of an integer."""
    return _RangeWithMetrics(args=args)


@dataclasses.dataclass
class _RangeWithMetrics:
    args: tuple[int, ...]

    def __iter__(self):
        loop_metrics = loop_metric_generator()
        for counter in range(*self.args):
            yield dataclasses.replace(next(loop_metrics), counter=counter)

    def __len__(self) -> int:
        return len(range(*self.args))


def loop_metric_generator(counter_init: int = 0) -> Generator[LoopMetrics, None, None]:
    """Generator for computing loop metrics.

    Note that the first `iteration_per_sec` metric will be 0.0.

    Example usage:
    ```
    # Note that this is an infinite loop.
    for metric in loop_metric_generator():
        time.sleep(1.0)
        print(metric)
    ```

    or:
    ```
    loop_metrics = loop_metric_generator()
    while True:
        time.sleep(1.0)
        print(next(loop_metrics).iterations_per_sec)
    ```
    """

    counter = counter_init
    del counter_init
    time_start = time.time()
    time_prev = time_start
    while True:
        time_now = time.time()
        yield LoopMetrics(
            counter=counter,
            iterations_per_sec=1.0 / (time_now - time_prev) if counter > 0 else 0.0,
            time_elapsed=time_now - time_start,
        )
        time_prev = time_now
        counter += 1


def get_git_commit_hash(cwd: Path | None = None) -> str:
    """Returns the current Git commit hash."""
    if cwd is None:
        cwd = Path.cwd()
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd.as_posix())
        .decode("ascii")
        .strip()
    )


def get_git_diff(cwd: Path | None = None) -> str:
    """Returns the output of `git diff HEAD`."""
    if cwd is None:
        cwd = Path.cwd()
    return (
        subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd.as_posix())
        .decode("ascii")
        .strip()
    )
