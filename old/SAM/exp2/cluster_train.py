import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_SUPPORTED_SUFFIXES = (".pt", ".npy", ".json", ".txt")
_DEFAULT_DISTANCE_METRIC = "l2"
_SUPPORTED_DISTANCE_METRICS = ("hybrid", "l2")

_DIM_TO_AXIS = {"h": 0, "w": 1, "f": 2}
_SCHEMES: Tuple[Tuple[str, str, str], ...] = (
    ("w", "h", "f"),
    ("w", "f", "h"),
    ("h", "w", "f"),
    ("h", "f", "w"),
    ("f", "w", "h"),
    ("f", "h", "w"),
)
_EPS = 1e-8


@dataclass
class SamplingModel:
    scheme: str
    dims_order: Tuple[str, str, str]
    shape_meta: Dict[str, int]
    norm_mode: str  # "div" or "sub"
    distance_metric: str
    vec_sample_indices: torch.Tensor
    vec_assign_map: torch.Tensor  # (S_amp, S_rest)
    vec_medoid_coords: torch.Tensor  # (K_v, 2)
    vec_max_pos: torch.Tensor  # (K_v,)
    outlier_map: torch.Tensor  # (S_amp, S_rest) bool
    train_hybrid_error: float
    train_vec_mse: float
    train_vec_cos: float


def _ensure_tensor(x: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if device is not None:
        x = x.to(device)
    return x.float()


def _reshape_attn_to_3d(attn_2d: torch.Tensor, l_h: int, l_w: int) -> Tuple[torch.Tensor, int]:
    l = attn_2d.shape[-1]
    l_f = l // (l_h * l_w)
    if l_f * l_h * l_w != l:
        raise ValueError("输入维度与提供的l_h/l_w不匹配")
    view_6d = attn_2d.reshape(l_f, l_h, l_w, l_f, l_h, l_w)
    attn_3d = view_6d.permute(1, 4, 2, 5, 0, 3).reshape(l_h * l_h, l_w * l_w, l_f * l_f)
    return attn_3d, l_f


def _reshape_3d_to_attn(tensor_3d: torch.Tensor, l_h: int, l_w: int, l_f: int) -> torch.Tensor:
    tensor_6d = tensor_3d.reshape(l_h, l_h, l_w, l_w, l_f, l_f)
    attn = tensor_6d.permute(4, 0, 2, 5, 1, 3).reshape(l_h * l_w * l_f, l_h * l_w * l_f)
    return attn


def _permute_base(tensor: torch.Tensor, dims_order: Tuple[str, str, str]) -> torch.Tensor:
    perm = tuple(_DIM_TO_AXIS[d] for d in dims_order)
    return tensor.permute(perm)


def _inverse_permute(tensor: torch.Tensor, dims_order: Tuple[str, str, str]) -> torch.Tensor:
    inv = [0, 0, 0]
    for idx, dim in enumerate(dims_order):
        inv[_DIM_TO_AXIS[dim]] = idx
    return tensor.permute(tuple(inv))


def _max_normalize_rows(data: torch.Tensor, mode: str = "div") -> Tuple[torch.Tensor, torch.Tensor]:
    max_vals = torch.amax(data, dim=-1, keepdim=True)

    if mode == "div":
        denom = max_vals.clone()
        denom[denom.abs() < _EPS] = _EPS
        scaled = data / denom
    elif mode == "sub":
        scaled = data - max_vals
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    return scaled, max_vals.squeeze(-1)


def _resolve_distance_metric(metric: Optional[str]) -> str:
    metric_normalized = (metric or _DEFAULT_DISTANCE_METRIC).lower()
    if metric_normalized not in _SUPPORTED_DISTANCE_METRICS:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return metric_normalized


def _compute_assignment_scores(
    data: torch.Tensor,
    centers: torch.Tensor,
    data_l2: torch.Tensor,
    centers_l2: torch.Tensor,
    metric: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if metric not in _SUPPORTED_DISTANCE_METRICS:
        raise ValueError(f"Unsupported distance metric: {metric}")
    dists = torch.cdist(data, centers, p=2)
    if metric == "hybrid":
        sqrt_mse = dists / np.sqrt(data.shape[-1])
        cos = torch.bmm(data_l2, centers_l2.transpose(1, 2))
        scores = sqrt_mse / torch.clamp(cos, min=1e-6)
        return scores, sqrt_mse, cos
    cos = torch.bmm(data_l2, centers_l2.transpose(1, 2))
    return dists, dists, cos


def _objective_from_mse(mse: torch.Tensor, cos: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "hybrid":
        return torch.sqrt(mse) / torch.clamp(cos, min=1e-6)
    return torch.sqrt(mse)


def _objective_value(mse: float, cos: float, metric: str) -> float:
    if metric == "hybrid":
        return float(np.sqrt(mse) / max(cos, 1e-6))
    return float(np.sqrt(mse))


def _kmeans_plus_plus_init(data: torch.Tensor, k: int) -> torch.Tensor:
    B, n, dim = data.shape
    centers = torch.empty(B, k, dim, device=data.device, dtype=data.dtype)

    for b in range(B):
        data_b = data[b]
        first_idx = torch.randint(0, n, (1,), device=data.device)
        centers[b, 0] = data_b[first_idx]
        if k == 1:
            continue

        dist_sq = torch.sum((data_b - centers[b, 0]) ** 2, dim=1)
        dist_sq = torch.clamp(dist_sq, min=_EPS)

        for cid in range(1, k):
            probs = dist_sq / dist_sq.sum()
            next_idx = torch.multinomial(probs, 1)
            centers[b, cid] = data_b[next_idx]
            new_dist_sq = torch.sum((data_b - centers[b, cid]) ** 2, dim=1)
            dist_sq = torch.minimum(dist_sq, torch.clamp(new_dist_sq, min=_EPS))

    return centers


def _kmeans_hybrid_medoid(
    data: torch.Tensor,
    k: int,
    max_iter: int,
    tol: float,
    init_centers: Optional[torch.Tensor] = None,
    metric: str = _DEFAULT_DISTANCE_METRIC,
    prune_ratio: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metric = _resolve_distance_metric(metric)
    is_batched = data.ndim == 3
    if not is_batched:
        data = data.unsqueeze(0)
        if init_centers is not None:
            init_centers = init_centers.unsqueeze(0)

    B, n, dim = data.shape
    k = min(k, max(1, n))
    data_l2 = torch.nn.functional.normalize(data, p=2, dim=-1)

    if init_centers is not None and init_centers.numel() > 0:
        init_centers = init_centers.to(data.device)
        if init_centers.ndim == 2:
            init_centers = init_centers.unsqueeze(0)
        existing_k = init_centers.shape[1]
        if existing_k >= k:
            centers = init_centers[:, :k].clone()
        else:
            centers = torch.empty(B, k, dim, device=data.device, dtype=data.dtype)
            centers[:, :existing_k] = init_centers
            if k - existing_k > 0:
                extra = _kmeans_plus_plus_init(data, k - existing_k)
                centers[:, existing_k:] = extra
    else:
        centers = _kmeans_plus_plus_init(data, k)

    assign = torch.zeros(B, n, dtype=torch.long, device=data.device)
    outlier_mask = torch.zeros(B, n, dtype=torch.bool, device=data.device)

    for _ in range(max_iter):
        centers_l2 = torch.nn.functional.normalize(centers, p=2, dim=-1)
        score_matrix, _, _ = _compute_assignment_scores(
            data, centers, data_l2, centers_l2, metric
        )
        assign = torch.argmin(score_matrix, dim=-1)

        one_hot = torch.nn.functional.one_hot(assign, centers.shape[1]).float()
        sum_centers = torch.bmm(one_hot.transpose(1, 2), data)
        counts = one_hot.sum(dim=1).unsqueeze(-1)
        mask = counts > 0.5
        new_centers = torch.where(mask, sum_centers / counts.clamp(min=1.0), centers)

        shift = torch.norm(new_centers - centers, dim=-1).mean(dim=1)
        centers = new_centers
        if torch.max(shift) <= tol:
            break

    centers_l2 = torch.nn.functional.normalize(centers, p=2, dim=-1)
    score_matrix, sqrt_mse, cos_matrix = _compute_assignment_scores(
        data, centers, data_l2, centers_l2, metric
    )
    assign = torch.argmin(score_matrix, dim=-1)

    batch_indices = torch.arange(B, device=data.device).unsqueeze(1).expand(-1, n)
    assigned_centers = centers[batch_indices, assign]
    point_mse = torch.mean((data - assigned_centers) ** 2, dim=-1)

    cluster_mse = torch.zeros(B, centers.shape[1], device=data.device, dtype=data.dtype)
    cluster_mse.scatter_add_(1, assign, point_mse)
    counts = torch.zeros(B, centers.shape[1], device=data.device, dtype=data.dtype)
    counts.scatter_add_(1, assign, torch.ones_like(point_mse))
    cluster_mse = cluster_mse / counts.clamp(min=1.0)
    cluster_mse[counts == 0] = float("inf")

    if prune_ratio > 0 and centers.shape[1] > 1:
        prune_count = int(math.ceil(centers.shape[1] * prune_ratio))
        prune_count = min(prune_count, centers.shape[1] - 1)
        if prune_count > 0:
            cluster_error_mean = cluster_mse.mean(dim=0)
            _, indices_desc = torch.sort(cluster_error_mean, descending=True)
            drop_idx = indices_desc[:prune_count]
            keep_idx = torch.sort(indices_desc[prune_count:])[0]

            for cid in drop_idx:
                outlier_mask |= assign == int(cid.item())

            centers = centers[:, keep_idx]
            centers_l2 = torch.nn.functional.normalize(centers, p=2, dim=-1)
            score_matrix, sqrt_mse, cos_matrix = _compute_assignment_scores(
                data, centers, data_l2, centers_l2, metric
            )
            assign = torch.argmin(score_matrix, dim=-1)
            batch_indices = torch.arange(B, device=data.device).unsqueeze(1).expand(-1, n)
            assigned_centers = centers[batch_indices, assign]
            point_mse = torch.mean((data - assigned_centers) ** 2, dim=-1)

    mse = torch.mean((data - assigned_centers) ** 2, dim=(1, 2))
    assigned_centers_l2 = torch.nn.functional.normalize(assigned_centers, p=2, dim=-1)
    cosine_sim = torch.sum(data_l2 * assigned_centers_l2, dim=-1).mean(dim=1)

    medoid_indices = torch.zeros(B, centers.shape[1], dtype=torch.long, device=data.device)
    for cid in range(centers.shape[1]):
        mask = (assign == cid) & (~outlier_mask)
        errors = score_matrix[:, :, cid].clone()
        errors[~mask] = float("inf")
        min_val, min_idx = torch.min(errors, dim=1)
        is_empty = min_val == float("inf")
        min_idx[is_empty] = 0
        medoid_indices[:, cid] = min_idx

    if not is_batched:
        return (
            centers.squeeze(0),
            assign.squeeze(0),
            medoid_indices.squeeze(0),
            outlier_mask.squeeze(0),
            mse.item(),
            cosine_sim.item(),
        )

    return centers, assign, medoid_indices, outlier_mask, mse, cosine_sim


def _prepare_shape_meta(l_h: int, l_w: int, l_f: int, dims_order: Tuple[str, str, str]) -> Dict[str, int]:
    return {
        "l_h": l_h,
        "l_w": l_w,
        "l_f": l_f,
        "S_h": l_h * l_h,
        "S_w": l_w * l_w,
        "S_f": l_f * l_f,
        "dims_order": dims_order,
    }


class ThreeDClusterEvaluator:
    def __init__(
        self,
        K_v: int = 16,
        max_iter: int = 25,
        tol: float = 1e-4,
        distance_metric: str = _DEFAULT_DISTANCE_METRIC,
        outlier_ratio: float = 0.05,
    ):
        self.K_v = K_v
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = _resolve_distance_metric(distance_metric)
        self.outlier_ratio = outlier_ratio

    def train(
        self,
        heads_files_map: List[Dict[int, List[str]]],
        l_h: int,
        l_w: int,
        device: str,
        apply_softmax: bool,
    ) -> Tuple[List[SamplingModel], List[List[Dict]]]:
        norm_mode = "div" if apply_softmax else "sub"
        num_heads = len(heads_files_map)

        first_head_map = heads_files_map[0]
        sorted_iters = sorted(first_head_map.keys())

        first_file = next(iter(first_head_map.values()))[0]
        first_score = _load_score(first_file, "cpu")
        first_attn, l_f = _reshape_attn_to_3d(first_score, l_h, l_w)

        gpu_device = torch.device(device)

        best_models: List[Optional[SamplingModel]] = [None] * num_heads
        best_hybrid_errors = [float("inf")] * num_heads
        all_summaries = [[] for _ in range(num_heads)]

        for dims in _SCHEMES:
            vec_centers: Optional[torch.Tensor] = None
            last_vec_mse = torch.zeros(num_heads, device=gpu_device)
            last_vec_cos = torch.zeros(num_heads, device=gpu_device)
            last_vec_medoids: Optional[torch.Tensor] = None
            last_vec_assign: Optional[torch.Tensor] = None
            last_vec_max_pos: Optional[torch.Tensor] = None
            last_outlier_mask: Optional[torch.Tensor] = None

            permuted_template = _permute_base(first_attn, dims)
            S_vec = permuted_template.shape[0]
            S_amp = permuted_template.shape[1]
            S_rest = permuted_template.shape[2]

            current_k = self.K_v

            for iter_idx in sorted_iters:
                batch_vectors: List[torch.Tensor] = []

                for h_idx in range(num_heads):
                    files = heads_files_map[h_idx].get(iter_idx, [])
                    head_vectors: List[torch.Tensor] = []

                    for fpath in files:
                        score = _load_score(fpath, "cpu")
                        if apply_softmax:
                            score = torch.softmax(score, dim=-1)
                        attn3d, _ = _reshape_attn_to_3d(score, l_h, l_w)
                        permuted = _permute_base(attn3d, dims)

                        vectors = permuted.reshape(S_vec, -1).transpose(0, 1)
                        vec_norm, _ = _max_normalize_rows(
                            vectors, mode=("div" if apply_softmax else "sub")
                        )
                        head_vectors.append(vec_norm)

                    if head_vectors:
                        batch_vectors.append(torch.cat(head_vectors, dim=0))

                curr_vectors = torch.stack(batch_vectors, dim=0).to(gpu_device)

                vec_centers, vec_assign, vec_medoids, vec_outliers, vec_mse, vec_cos = _kmeans_hybrid_medoid(
                    curr_vectors,
                    current_k,
                    self.max_iter,
                    self.tol,
                    init_centers=vec_centers,
                    metric=self.distance_metric,
                    prune_ratio=self.outlier_ratio,
                )
                current_k = vec_centers.shape[1]

                if iter_idx == sorted_iters[-1]:
                    last_vec_mse = vec_mse
                    last_vec_cos = vec_cos
                    last_vec_medoids = vec_medoids
                    last_outlier_mask = vec_outliers

                    B, K_v_curr = vec_medoids.shape
                    batch_idx_v = torch.arange(B, device=gpu_device).unsqueeze(1).expand(-1, K_v_curr)
                    selected_vecs = curr_vectors[batch_idx_v, vec_medoids]
                    last_vec_max_pos = torch.argmax(selected_vecs, dim=2).to(torch.int16)

                    n_files = len(heads_files_map[0][iter_idx])
                    vec_assign_reshaped = vec_assign.reshape(B, n_files, S_amp, S_rest)
                    last_vec_assign, _ = torch.mode(vec_assign_reshaped, dim=1)

                del curr_vectors
                del batch_vectors

            if last_outlier_mask is None:
                last_outlier_mask = torch.zeros_like(vec_assign, dtype=torch.bool)

            vec_objective = _objective_from_mse(last_vec_mse, last_vec_cos, self.distance_metric)
            outlier_mask_reshaped = last_outlier_mask.reshape(num_heads, n_files, S_amp, S_rest)
            aggregated_outliers = outlier_mask_reshaped.any(dim=1)

            for h_idx in range(num_heads):
                err = vec_objective[h_idx].item()
                summary = {
                    "scheme": "{}-{}-{}".format(*dims),
                    "vec_mse": last_vec_mse[h_idx].item(),
                    "vec_cos": last_vec_cos[h_idx].item(),
                    "hybrid_err": err,
                }
                all_summaries[h_idx].append(summary)

                if err < best_hybrid_errors[h_idx]:
                    best_hybrid_errors[h_idx] = err

                    shape_meta = _prepare_shape_meta(l_h, l_w, l_f, dims)
                    shape_meta.update(
                        {
                            "S_first": S_vec,
                            "S_second": S_amp,
                            "S_third": S_rest,
                        }
                    )

                    h_vec_medoids = last_vec_medoids[h_idx]
                    vec_spatial_indices = h_vec_medoids % (S_amp * S_rest)
                    v_coords_1 = (vec_spatial_indices // S_rest).to(torch.int16)
                    v_coords_2 = (vec_spatial_indices % S_rest).to(torch.int16)
                    vec_medoid_coords = torch.stack([v_coords_1, v_coords_2], dim=1)

                    vec_map = last_vec_assign[h_idx].to(torch.int16)
                    outlier_map = aggregated_outliers[h_idx].to(torch.bool)
                    vec_map[outlier_map] = -1

                    best_models[h_idx] = SamplingModel(
                        scheme="{}-{}-{}".format(*dims),
                        dims_order=dims,
                        shape_meta=shape_meta,
                        norm_mode=norm_mode,
                        distance_metric=self.distance_metric,
                        vec_sample_indices=vec_spatial_indices.cpu(),
                        vec_assign_map=vec_map.cpu(),
                        vec_medoid_coords=vec_medoid_coords.cpu(),
                        vec_max_pos=last_vec_max_pos[h_idx].cpu(),
                        outlier_map=outlier_map.cpu(),
                        train_hybrid_error=err,
                        train_vec_mse=last_vec_mse[h_idx].item(),
                        train_vec_cos=last_vec_cos[h_idx].item(),
                    )

        return best_models, all_summaries

    def validate(self, score: torch.Tensor, model: SamplingModel) -> Dict[str, float]:
        score = _ensure_tensor(score)
        l_h = model.shape_meta["l_h"]
        l_w = model.shape_meta["l_w"]
        l_f = model.shape_meta["l_f"]

        attn3d, _ = _reshape_attn_to_3d(score, l_h, l_w)
        permuted = _permute_base(attn3d, model.dims_order)

        S_vec = model.shape_meta["S_first"]
        S_amp = model.shape_meta["S_second"]
        S_rest = model.shape_meta["S_third"]

        vectors = permuted.reshape(S_vec, -1).transpose(0, 1)
        vec_norm, vec_max = _max_normalize_rows(vectors, mode=model.norm_mode)

        vec_indices = model.vec_sample_indices.to(vec_norm.device)
        current_vec_centers = vec_norm[vec_indices]

        metric = _resolve_distance_metric(model.distance_metric)
        vec_l2 = torch.nn.functional.normalize(vec_norm, p=2, dim=1)
        centers_v_l2 = torch.nn.functional.normalize(current_vec_centers, p=2, dim=1)
        score_v, _, _ = _compute_assignment_scores(
            vec_norm.unsqueeze(0),
            current_vec_centers.unsqueeze(0),
            vec_l2.unsqueeze(0),
            centers_v_l2.unsqueeze(0),
            metric,
        )
        vec_assign = torch.argmin(score_v.squeeze(0), dim=1)

        rec_vec_norm = current_vec_centers[vec_assign]
        vec_max_vals = vec_max.unsqueeze(1)

        if model.norm_mode == "div":
            rec_vectors = (rec_vec_norm * vec_max_vals.clamp_min(_EPS)).transpose(0, 1)
        else:
            rec_vectors = (rec_vec_norm + vec_max_vals).transpose(0, 1)

        rec_permuted = rec_vectors.reshape(S_vec, S_amp, S_rest)

        outlier_map = model.outlier_map.to(rec_permuted.device)
        if outlier_map.any():
            rec_flat = rec_permuted.reshape(S_vec, -1)
            perm_flat = permuted.reshape(S_vec, -1)
            outlier_flat = outlier_map.view(-1)
            rec_flat[:, outlier_flat] = perm_flat[:, outlier_flat]
            rec_permuted = rec_flat.reshape(S_vec, S_amp, S_rest)

        score_rec = _reshape_3d_to_attn(_inverse_permute(rec_permuted, model.dims_order), l_h, l_w, l_f)

        mse = torch.mean((score_rec - score) ** 2).item()
        rec_flat = score_rec.reshape(-1)
        ref_flat = score.reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(rec_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()

        objective = _objective_value(mse, cos_sim, metric)

        return {
            "rec_mse": mse,
            "rec_cos": cos_sim,
            "rec_objective": objective,
            "rec_hybrid": objective,
        }


def _load_score(path: str, device: str = "cpu") -> torch.Tensor:
    lower = path.lower()
    if lower.endswith((".pt", ".pth")):
        data = torch.load(path, map_location=device)
        if isinstance(data, dict):
            for key in ("score", "scores", "data"):
                if key in data:
                    data = data[key]
                    break
    elif lower.endswith(".npy"):
        data = np.load(path)
    elif lower.endswith(".npz"):
        with np.load(path) as npz:
            key = "score" if "score" in npz.files else npz.files[0]
            data = npz[key]
    elif lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            for key in ("score", "scores", "data"):
                if key in loaded:
                    data = loaded[key]
                    break
            else:
                data = next((v for v in loaded.values() if isinstance(v, (list, tuple))), loaded)
        else:
            data = loaded
    elif lower.endswith(".txt"):
        data = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported score file: {path}")
    tensor = torch.as_tensor(data, device=device, dtype=torch.float32)
    if tensor.ndim < 2:
        raise ValueError(f"Score tensor must be at least 2-D, got shape {tuple(tensor.shape)}")
    return tensor


def _collect_score_files(score_dir: str) -> List[str]:
    files = [
        os.path.join(score_dir, f)
        for f in os.listdir(score_dir)
        if f.lower().endswith(_SUPPORTED_SUFFIXES) and "score" in f.lower()
    ]
    files.sort()
    return files


def _extract_meta(filename: str, pattern: Optional[re.Pattern]) -> Dict[str, int]:
    if not pattern:
        return {}
    match = pattern.search(filename)
    if not match:
        return {}
    labels = ("iter", "layer", "head")
    meta: Dict[str, int] = {}
    for idx, value in enumerate(match.groups()):
        key = labels[idx] if idx < len(labels) else f"group_{idx}"
        try:
            meta[key] = int(value)
        except ValueError:
            meta[key] = value
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Cluster Evaluation (Vector Only)")
    parser.add_argument("--score_dir", type=str, required=True, help="包含多份得分文件的目录，自动批处理")
    parser.add_argument(
        "--file_regex",
        type=str,
        default=r"score_It(\d+)_L(\d+)_H(\d+)",
        help="用于解析文件名中迭代/层/头信息的正则",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./attn_analysis/3d_cluster/3d_cluster.log",
        help="目录模式日志输出",
    )
    parser.add_argument("--B_h", type=int, required=True, help="l_h")
    parser.add_argument("--B_w", type=int, required=True, help="l_w")
    parser.add_argument("--K_v", type=int, default=1024, help="向量聚类中心数")
    parser.add_argument("--max_iter", type=int, default=50, help="k-means最大迭代次数")
    parser.add_argument("--tol", type=float, default=1e-4, help="k-means收敛阈值")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="./attn_analysis/3d_cluster/3d_cluster_result.json",
    )
    parser.add_argument(
        "--blueprint_dir",
        type=str,
        default="./attn_analysis/3d_cluster/blueprints",
        help="保存聚类蓝图的目录",
    )
    parser.add_argument(
        "--cluster_iter",
        type=int,
        default=5,
        help="用于聚类训练的最大迭代次数（不包含）。It < cluster_iter 的文件用于训练，It >= cluster_iter 的文件用于验证。",
    )
    parser.add_argument(
        "--apply_softmax",
        action="store_true",
        help="如果为真，应用Softmax并使用除法归一化；如果为假，直接使用减法归一化。",
    )
    parser.add_argument("--heads_batch_size", type=int, default=1, help="并行训练的Head数量")
    parser.add_argument("--skip_eval", action="store_true", help="跳过验证阶段")
    parser.add_argument(
        "--distance_metric",
        choices=_SUPPORTED_DISTANCE_METRICS,
        default=_DEFAULT_DISTANCE_METRIC,
        help="选择聚类距离度量（hybrid或l2）",
    )
    parser.add_argument(
        "--outlier_ratio",
        type=float,
        default=0.05,
        help="聚类完成后基于平均MSE剔除的中心比例（默认5%）。",
    )
    args = parser.parse_args()
    torch.cuda.manual_seed_all(42)

    evaluator = ThreeDClusterEvaluator(
        K_v=args.K_v,
        max_iter=args.max_iter,
        tol=args.tol,
        distance_metric=args.distance_metric,
        outlier_ratio=args.outlier_ratio,
    )

    pattern = re.compile(args.file_regex) if args.file_regex else None
    try:
        files = _collect_score_files(args.score_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{args.score_dir}' not found.")
        return

    if not files:
        print(f"No supported score files found under {args.score_dir}")
        return

    layer_groups: Dict[int, Dict[int, List[str]]] = {}

    for fpath in files:
        meta = _extract_meta(os.path.basename(fpath), pattern)
        if "layer" in meta and "head" in meta:
            layer = meta["layer"]
            head = meta["head"]
            layer_groups.setdefault(layer, {}).setdefault(head, []).append(fpath)

    sorted_layers = sorted(layer_groups.keys())
    print(f"Found {len(files)} files across {len(sorted_layers)} Layers.")

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.blueprint_dir, exist_ok=True)

    with open(args.log_file, "w", encoding="utf-8") as log:
        log.write("File | Iter | Layer | Head | RecMSE | RecCos | RecHybrid | TrainHybrid | BestScheme\n")
        log.write("-" * 100 + "\n")

    results_json: List[Dict[str, Any]] = []

    for layer in sorted_layers:
        layer_blueprints: Dict[str, Dict[str, Any]] = {}
        heads = sorted(layer_groups[layer].keys())

        print(f"=== Processing Layer {layer} ({len(heads)} heads) ===")

        for i in range(0, len(heads), args.heads_batch_size):
            batch_heads = heads[i : i + args.heads_batch_size]
            print(f"  > Processing Batch Heads: {batch_heads}")

            batch_train_files_map = []
            batch_test_files_map = []
            valid_batch_heads: List[int] = []

            for head in batch_heads:
                group_files = layer_groups[layer][head]
                train_files_map: Dict[int, List[str]] = {}
                test_files: List[str] = []

                for fpath in group_files:
                    meta = _extract_meta(os.path.basename(fpath), pattern)
                    iter_idx = meta.get("iter", -1)
                    if iter_idx == -1:
                        continue

                    if iter_idx < args.cluster_iter:
                        train_files_map.setdefault(iter_idx, []).append(fpath)
                    else:
                        test_files.append(fpath)

                if not train_files_map:
                    print(f"Skipping L{layer} H{head}: No training files")
                    continue

                batch_train_files_map.append(train_files_map)
                batch_test_files_map.append(test_files)
                valid_batch_heads.append(head)

            if not valid_batch_heads:
                continue

            try:
                best_models, _ = evaluator.train(
                    batch_train_files_map,
                    args.B_h,
                    args.B_w,
                    args.device,
                    args.apply_softmax,
                )

                for idx, head in enumerate(valid_batch_heads):
                    best_model = best_models[idx]
                    blueprint_key = f"H{head}"
                    layer_blueprints[blueprint_key] = {
                        "scheme": best_model.scheme,
                        "dims_order": best_model.dims_order,
                        "shape_meta": best_model.shape_meta,
                        "norm_mode": best_model.norm_mode,
                        "distance_metric": best_model.distance_metric,
                        "vec_map": best_model.vec_assign_map.cpu(),
                        "vec_medoids": best_model.vec_medoid_coords.cpu(),
                        "vec_max_pos": (
                            best_model.vec_max_pos.cpu()
                            if best_model.vec_max_pos.numel() > 0
                            else None
                        ),
                        "outlier_map": best_model.outlier_map.cpu(),
                    }
            except Exception as e:
                print(f"Error training batch {valid_batch_heads}: {e}")
                import traceback

                traceback.print_exc()
                continue

            if not args.skip_eval:
                for idx, head in enumerate(valid_batch_heads):
                    best_model = best_models[idx]
                    test_files = batch_test_files_map[idx]

                    for path in test_files:
                        try:
                            score = _load_score(path, args.device)
                            if args.apply_softmax:
                                score = torch.softmax(score, dim=-1)

                            metrics = evaluator.validate(score, best_model)
                            meta = _extract_meta(os.path.basename(path), pattern)

                            entry = {
                                "file": os.path.basename(path),
                                "meta": meta,
                                "metrics": metrics,
                                "train_scheme": best_model.scheme,
                                "train_hybrid_error": best_model.train_hybrid_error,
                            }
                            results_json.append(entry)

                            log_line = (
                                f"{entry['file']} | {meta.get('iter', '-')}"
                                f" | {meta.get('layer', '-')} | {meta.get('head', '-')}"
                                f" | {metrics['rec_mse']:.6f} | {metrics['rec_cos']:.4f} | {metrics['rec_hybrid']:.4f}"
                                f" | {best_model.train_hybrid_error:.4f} | {best_model.scheme}\n"
                            )
                            print(log_line.strip())
                            with open(args.log_file, "a", encoding="utf-8") as log:
                                log.write(log_line)

                        except Exception as e:
                            print(f"Error processing {os.path.basename(path)}: {e}")

            del best_models
            torch.cuda.empty_cache()

        layer_file = os.path.join(args.blueprint_dir, f"blueprint_L{layer}.pt")
        torch.save(layer_blueprints, layer_file)
        print(f"Saved blueprints for Layer {layer} to {layer_file}")

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump({"results": results_json}, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved to {args.result_json} and {args.log_file}")


if __name__ == "__main__":
    main()