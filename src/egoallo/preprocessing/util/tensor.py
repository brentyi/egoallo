from loguru import logger as guru
from typing import TypeVar, Dict, List
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from PIL import Image


def batch_sum(x, nldims=1):
    """
    Sum across all but batch dimension(s)
    :param x (B, *)
    :param nldims (optional int=1) number of leading dims to keep
    """
    if x.ndim > nldims:
        return x.sum(dim=tuple(range(nldims, x.ndim)))
    return x


def batch_mean(x, nldims=1):
    """
    Mean across all but batch dimension(s)
    :param x (B, *)
    :param nldims (optional int=1) number of leading dims to keep
    """
    if x.ndim > nldims:
        return x.mean(dim=tuple(range(nldims, x.ndim)))
    return x


def pad_dim(x, max_len, dim=0, start=0, **kwargs):
    """
    pads x to max_len in specified dim
    :param x (tensor)
    :param max_len (int)
    :param start (int default 0)
    :param dim (optional int default 0)
    """
    N = x.shape[dim]
    if max_len == N:
        return x

    if max_len < N:
        return torch.narrow(x, dim, start, max_len)

    if dim < 0:
        dim = x.ndim + dim
    pad = [0, 0] * x.ndim
    pad[2 * dim + 1] = start
    pad[2 * dim] = max_len - (N + start)
    return F.pad(x, pad[::-1], **kwargs)


def pad_back(x, max_len, dim=0, **kwargs):
    return pad_dim(x, max_len, dim, 0, **kwargs)


def pad_front(x, max_len, dim=0, **kwargs):
    N = x.shape[dim]
    return pad_dim(x, max_len, dim=dim, start=max_len - N, **kwargs)


def read_image(path, scale=1):
    im = Image.open(path)
    if scale == 1:
        return np.array(im)
    W, H = im.size
    w, h = int(scale * W), int(scale * H)
    return np.array(im.resize((w, h), Image.ANTIALIAS))


T = TypeVar("T")


def move_to(obj: T, device) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}  # type: ignore
    if isinstance(obj, (list, tuple)):
        return [move_to(x, device) for x in obj]  # type: ignore
    return obj  # otherwise do nothing


def detach_all(obj: T) -> T:
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: detach_all(v) for k, v in obj.items()}  # type: ignore
    if isinstance(obj, (list, tuple)):
        return [detach_all(x) for x in obj]  # type: ignore
    return obj  # otherwise do nothing


def to_torch(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, dict):
        return {k: to_torch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_torch(x) for x in obj]
    return obj


def to_np(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    if isinstance(obj, dict):
        return {k: to_np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_np(x) for x in obj]
    return obj


def load_npz_as_dict(path, **kwargs):
    npz = np.load(path, **kwargs)
    return {key: npz[key] for key in npz.files}


def get_device(i=0):
    device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def invert_nested_dict(d):
    """
    invert nesting of dict of dicts
    """
    outer_keys = d.keys()  # outer nested keys
    inner_keys = next(iter(d.values())).keys()  # inner nested keys
    return {
        inner: {outer: d[outer][inner] for outer in outer_keys} for inner in inner_keys
    }


def batchify_dicts(dict_list: List[Dict]) -> Dict:
    """
    given a list of dicts with shared keys,
    return a dict of those keys stacked into lists
    """
    keys = dict_list[0].keys()
    if not all(d.keys() == keys for d in dict_list):
        guru.warning("found dicts with not same keys! using first element's keys")
    return {k: [d[k] for d in dict_list] for k in keys}


def batchify_recursive(dict_list: List[Dict], levels: int = -1):
    x = dict_list[0]
    keys = x.keys()
    out = {}
    for k in keys:
        if isinstance(x[k], dict) and levels != 0:
            # aggregate the values with this key into
            # a list of dicts and batchify recursively
            vals = batchify_recursive(
                [d[k] for d in dict_list], levels=levels - 1
            )
        elif isinstance(x[k], (list, tuple)) and levels != 0:
            # aggregate the values with this key into
            # a flattened list and batchify recursively
            vals = [x for d in dict_list for x in d[k]]
            # perhaps another list of dicts
            if isinstance(vals[0], dict) and levels != 0:
                vals = batchify_recursive( vals, levels=levels - 1)
        else:
            # aggregate the values with this key into a list as is
            vals = [d[k] for d in dict_list]
        out[k] = vals
    return out


def unbatch_dict(batched_dict, batch_size):
    """
    :param d (dict) of batched tensors
    return len B list of dicts of unbatched tensors
    """
    out_list = [{} for _ in range(batch_size)]
    for k, v in batched_dict.items():
        for b in range(batch_size):
            out_list[b][k] = get_batch_element(v, b, batch_size)
    return out_list


def get_batch_element(batch, idx, batch_size):
    if isinstance(batch, torch.Tensor):
        return batch[idx] if idx < batch.shape[0] else batch
    if isinstance(batch, dict):
        return {k: get_batch_element(v, idx, batch_size) for k, v in batch.items()}
    if isinstance(batch, list):
        if len(batch) == batch_size:
            return batch[idx]
        return [get_batch_element(v, idx, batch_size) for v in batch]
    if isinstance(batch, tuple):
        if len(batch) == batch_size:
            return batch[idx]
        return tuple(get_batch_element(v, idx, batch_size) for v in batch)
    return batch


def narrow_dict(input_dict, tdim, start, length):
    """
    slice dict of tensors
    :param d (dict)
    :param idcs (tensor or list)
    """
    input_batch = {}
    for k, v in input_dict.items():
        input_batch[k] = narrow_obj(v, tdim, start, length)
    return input_batch


def narrow_list(input_list, tdim, start, length):
    return [narrow_obj(x, tdim, start, length) for x in input_list]


def narrow_obj(v, tdim, start, length):
    if isinstance(v, dict):
        return narrow_dict(v, tdim, start, length)
    if isinstance(v, (tuple, list)):
        return narrow_list(v, tdim, start, length)
    if not isinstance(v, Tensor):
        return v
    if v.ndim <= tdim or v.shape[tdim] < start + length:
        return v
    return v.narrow(tdim, start, length)


def scatter_intervals(tensor, start, end, T: int = -1):
    """
    Scatter the tensor contents into intervals from start to end
    output tensor indexed from 0 to end.max()
    :param tensor (B, S, *)
    :param start (B) start indices
    :param end (B) end indices
    :param T (int, optional) max length
    returns (B, T, *) scattered tensor
    """
    assert isinstance(tensor, torch.Tensor) and tensor.ndim >= 2
    if T < 0:
        T = end.max()
    assert torch.all(end <= T)

    B, S, *dims = tensor.shape
    start, end = start.long(), end.long()
    # get idcs that go past the last time step so we don't have repeat indices in scatter
    idcs = time_segment_idcs(start, end, min_len=T, clip=False)  # (B, T)
    # mask out the extra padding
    mask = idcs >= end[:, None]
    tensor[mask] = 0

    idcs = idcs.reshape(B, S, *(1,) * len(dims)).repeat(1, 1, *dims)
    output = torch.zeros(
        B, idcs.max() + 1, *dims, device=tensor.device, dtype=tensor.dtype
    )
    output.scatter_(1, idcs, tensor)
    # slice out the extra segments
    return output[:, :T]


def get_scatter_mask(start, end, T):
    """
    get the mask of selected intervals
    """
    B = start.shape[0]
    start, end = start.long(), end.long()
    assert torch.all(end <= T)
    idcs = time_segment_idcs(start, end, clip=True)
    mask = torch.zeros(B, T, device=start.device, dtype=torch.bool)
    mask.scatter_(1, idcs, 1)
    return mask


def select_intervals(series, start, end, pad_len: int = -1):
    """
    Select slices of a tensor from start to end
    will pad uneven sequences to all the max segment length
    :param series (B, T, *)
    :param start (B)
    :param end (B)
    returns (B, S, *) selected segments, S = max(end - start)
    """
    B, T, *dims = series.shape
    assert torch.all(end <= T)
    sel = time_segment_idcs(start, end, min_len=pad_len, clip=True)
    S = sel.shape[1]
    sel = sel.reshape(B, S, *(1,) * len(dims)).repeat(1, 1, *dims)
    return torch.gather(series, 1, sel)


def get_select_mask(start, end):
    """
    get the mask of unpadded elementes for the selected time segments
    e.g. sel[mask] are the unpadded elements
    :param start (B)
    :param end (B)
    """
    idcs = time_segment_idcs(start, end, clip=False)
    return idcs < end[:, None]  # (B, S)


def time_segment_idcs(start, end, min_len: int = -1, clip: bool = True):
    """
    :param start (B)
    :param end (B)
    returns (B, S) long tensor of indices, where S = max(end - start)
    """
    start, end = start.long(), end.long()
    S = max(int((end - start).max()), min_len)
    seg = torch.arange(S, dtype=torch.int64, device=start.device)
    idcs = start[:, None] + seg[None, :]  # (B, S)
    if clip:
        # clip at the lengths of each track
        imax = torch.maximum(end - 1, start)[:, None]
        idcs = idcs.clamp(max=imax)
    return idcs
