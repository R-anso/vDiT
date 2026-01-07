import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

_SUPPORTED_SUFFIXES = (".npy", ".json", ".txt")

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
class SchemeClusterResult:
    scheme: str
    dims_order: Tuple[str, str, str]
    shape_meta: Dict[str, int]
    vec_centers: torch.Tensor
    vec_assignments_matrix: torch.Tensor
    vec_center_max_idx: torch.Tensor
    mag_centers: torch.Tensor
    mag_assignments_vec: torch.Tensor
    mag_assignments_matrix: torch.Tensor
    mag_center_max_idx: torch.Tensor
    vec_mse: float
    vec_cos: float
    mag_mse: float
    mag_cos: float
    total_hybrid_error: float


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


def _max_normalize_rows(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_vals = torch.amax(data, dim=1, keepdim=True)
    denom = max_vals.clone()
    denom[denom.abs() < _EPS] = _EPS
    scaled = data / denom
    return scaled, max_vals.squeeze(1)


def _kmeans(data: torch.Tensor, k: int, max_iter: int, tol: float) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    n, dim = data.shape
    k = min(k, max(1, n))
    
    data_l2 = torch.nn.functional.normalize(data, p=2, dim=1)

    if k == n:
        centers = data.clone()
        assign = torch.arange(n, device=data.device)
        return centers, assign, 0.0, 1.0

    idx = torch.randperm(n, device=data.device)[:k]
    centers = data[idx].clone()

    prev_hybrid = float('inf')

    for _ in range(max_iter):
        # --- 1. Assignment based on Hybrid Error ---
        # Calculate Pairwise MSE: (N, K)
        # cdist calculates Euclidean distance (sqrt of sum of squared diffs)
        dists = torch.cdist(data, centers, p=2)
        # MSE = dist^2 / dim. So sqrt(MSE) = dist / sqrt(dim)
        sqrt_mse_matrix = dists / np.sqrt(dim)
        
        # Calculate Pairwise Cosine: (N, K)
        centers_l2 = torch.nn.functional.normalize(centers, p=2, dim=1)
        cos_matrix = torch.mm(data_l2, centers_l2.t())
        
        # Calculate Hybrid Error Matrix: (N, K)
        # Hybrid = sqrt(MSE) / max(Cos, eps)
        hybrid_matrix = sqrt_mse_matrix / torch.clamp(cos_matrix, min=1e-6)
        
        # Assign to the center with minimal Hybrid Error
        assign = torch.argmin(hybrid_matrix, dim=1)

        # --- 2. Update Centers (Standard Mean) ---
        new_centers = centers.clone()
        for cid in range(k):
            mask = assign == cid
            if mask.any():
                new_centers[cid] = data[mask].mean(dim=0)
        centers = new_centers
        
        # --- 3. Convergence Check based on Global Hybrid Error ---
        current_mse = torch.mean((data - centers[assign]) ** 2).item()
        
        centers_l2_new = torch.nn.functional.normalize(centers, p=2, dim=1)
        assigned_centers_l2 = centers_l2_new[assign]
        current_cos = torch.sum(data_l2 * assigned_centers_l2, dim=1).mean().item()
        
        current_hybrid = current_mse / max(current_cos, 1e-6)**2
        
        if abs(current_hybrid - prev_hybrid) < tol:
            break
        prev_hybrid = current_hybrid
    
    mse = torch.mean((data - centers[assign]) ** 2).item()
    
    centers_l2 = torch.nn.functional.normalize(centers, p=2, dim=1)
    assigned_centers_l2 = centers_l2[assign]
    cosine_sim = torch.sum(data_l2 * assigned_centers_l2, dim=1).mean().item()

    return centers, assign, mse, cosine_sim


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
    def __init__(self, K_v: int = 16, K_m: int = 16, max_iter: int = 25, tol: float = 1e-4):
        self.K_v = K_v
        self.K_m = K_m
        self.max_iter = max_iter
        self.tol = tol

    def evaluate(
        self,
        score: torch.Tensor,
        l_h: int,
        l_w: int,
    ) -> Tuple[List[SchemeClusterResult], List[List[Dict[str, float]]]]:
        score = _ensure_tensor(score)
        leading = int(score.numel() // (score.shape[-1] * score.shape[-2]))
        flat = score.reshape(leading, score.shape[-2], score.shape[-1])
        best_results: List[SchemeClusterResult] = []
        error_summaries: List[List[Dict[str, float]]] = []

        for idx in range(leading):
            attn = flat[idx]
            attn3d, l_f = _reshape_attn_to_3d(attn, l_h, l_w)
            shape_meta = _prepare_shape_meta(l_h, l_w, l_f, ("h", "w", "f"))
            scheme_results: List[SchemeClusterResult] = []

            for dims in _SCHEMES:
                permuted = _permute_base(attn3d, dims)
                S_vec = permuted.shape[0]
                S_amp = permuted.shape[1]
                S_rest = permuted.shape[2]

                vectors = permuted.reshape(S_vec, -1).transpose(0, 1)
                vec_norm, vec_max = _max_normalize_rows(vectors)
                vec_centers, vec_assign, vec_mse, vec_cos = _kmeans(vec_norm, self.K_v, self.max_iter, self.tol)
                vec_assign_mat = vec_assign.reshape(S_amp, S_rest)
                vec_center_max_idx = torch.argmax(vec_centers, dim=1)

                max_mat = vec_max.reshape(S_amp, S_rest)
                amp_vectors = max_mat.transpose(0, 1)
                amp_norm, _ = _max_normalize_rows(amp_vectors)
                mag_centers, mag_assign, mag_mse, mag_cos = _kmeans(amp_norm, self.K_m, self.max_iter, self.tol)
                mag_assign_vec = mag_assign.clone()
                mag_assign_mat = mag_assign_vec.unsqueeze(0).expand(S_amp, -1).contiguous()
                mag_center_max_idx = torch.argmax(mag_centers, dim=1)

                vec_hybrid = np.sqrt(vec_mse) / max(vec_cos, 1e-6)
                mag_hybrid = np.sqrt(mag_mse) / max(mag_cos, 1e-6)
                total_hybrid = vec_hybrid + mag_hybrid

                scheme_results.append(
                    SchemeClusterResult(
                        scheme="{}-{}-{}".format(*dims),
                        dims_order=dims,
                        shape_meta={
                            **shape_meta,
                            "S_first": S_vec,
                            "S_second": S_amp,
                            "S_third": S_rest,
                        },
                        vec_centers=vec_centers,
                        vec_assignments_matrix=vec_assign_mat,
                        vec_center_max_idx=vec_center_max_idx,
                        mag_centers=mag_centers,
                        mag_assignments_vec=mag_assign_vec,
                        mag_assignments_matrix=mag_assign_mat,
                        mag_center_max_idx=mag_center_max_idx,
                        vec_mse=vec_mse,
                        vec_cos=vec_cos,
                        mag_mse=mag_mse,
                        mag_cos=mag_cos,
                        total_hybrid_error=total_hybrid
                    )
                )

            scheme_results.sort(key=lambda r: r.total_hybrid_error)
            best_results.append(scheme_results[0])
            error_summaries.append(
                [
                    {
                        "scheme": r.scheme, 
                        "vec_mse": r.vec_mse, "vec_cos": r.vec_cos,
                        "mag_mse": r.mag_mse, "mag_cos": r.mag_cos,
                        "hybrid_err": r.total_hybrid_error
                    }
                    for r in scheme_results
                ]
            )

        return best_results, error_summaries

    @staticmethod
    def recover(
        score_ref: torch.Tensor,
        result: SchemeClusterResult,
    ) -> Tuple[torch.Tensor, float, float, float]:
        score_ref = _ensure_tensor(score_ref, device=result.vec_centers.device)
        l_h = result.shape_meta["l_h"]
        l_w = result.shape_meta["l_w"]
        l_f = result.shape_meta["l_f"]

        ref3d, _ = _reshape_attn_to_3d(score_ref, l_h, l_w)
        ref_perm = _permute_base(ref3d, result.dims_order)

        S_vec = result.shape_meta["S_first"]
        S_amp = result.shape_meta["S_second"]
        S_rest = result.shape_meta["S_third"]

        vec_assign = result.vec_assignments_matrix.reshape(-1).long()
        vec_norm = result.vec_centers[vec_assign]
        vec_norm = vec_norm.reshape(S_amp, S_rest, S_vec).permute(2, 0, 1)

        mag_assign = result.mag_assignments_vec.long()
        amp_norm = result.mag_centers[mag_assign].transpose(0, 1).reshape(S_amp, S_rest)

        scale_vals = []
        for ridx in range(S_rest):
            mag_cluster = mag_assign[ridx]
            amp_peak_idx = result.mag_center_max_idx[mag_cluster].item()
            vec_cluster = result.vec_assignments_matrix[amp_peak_idx, ridx].long()
            vec_peak_idx = result.vec_center_max_idx[vec_cluster].item()
            scale_val = ref_perm[vec_peak_idx, amp_peak_idx, ridx]
            scale_vals.append(scale_val)
        scale_vec = torch.stack(scale_vals)
        max_mat = amp_norm * scale_vec.unsqueeze(0)

        max_flat = max_mat.reshape(1, -1).transpose(0, 1)
        scaled_vectors = (
            result.vec_centers[vec_assign] * max_flat.clamp_min(_EPS)
        ).reshape(S_amp, S_rest, S_vec).permute(2, 0, 1)

        base_tensor = _inverse_permute(scaled_vectors, result.dims_order)
        score_rec = _reshape_3d_to_attn(base_tensor, l_h, l_w, l_f)
        
        mse = torch.mean((score_rec - score_ref) ** 2).item()
        
        rec_flat = score_rec.reshape(-1)
        ref_flat = score_ref.reshape(-1)
        cos_sim = torch.nn.functional.cosine_similarity(rec_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
        
        hybrid_err = np.sqrt(mse) / max(cos_sim, 1e-6)
        
        return score_rec, mse, cos_sim, hybrid_err

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

def _summarize_outputs(best_results: List[SchemeClusterResult], summaries: List[List[Dict[str, float]]]) -> Dict[str, List[Dict[str, float]]]:
    return {
        "best": [
            {
                "index": idx,
                "scheme": r.scheme,
                "vec_mse": float(r.vec_mse),
                "vec_cos": float(r.vec_cos),
                "mag_mse": float(r.mag_mse),
                "mag_cos": float(r.mag_cos),
                "hybrid_error": float(r.total_hybrid_error),
            }
            for idx, r in enumerate(best_results)
        ],
        "summaries": summaries,
    }

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
            meta[key] = value  # fallback for非数字片段
    return meta

def main() -> None:
    parser = argparse.ArgumentParser(description="3D Cluster Evaluation")
    parser.add_argument("--score_dir", type=str, required=True, help="包含多份得分文件的目录，自动批处理")
    parser.add_argument("--file_regex", type=str, default=r"score_It(\d+)_L(\d+)_H(\d+)", help="用于解析文件名中迭代/层/头信息的正则")
    parser.add_argument("--log_file", type=str, default="./attn_analysis/3d_cluster/3d_cluster.log", help="目录模式日志输出")
    parser.add_argument("--B_h", type=int, required=True, help="l_h")
    parser.add_argument("--B_w", type=int, required=True, help="l_w")
    parser.add_argument("--K_v", type=int, default=256, help="向量聚类中心数")
    parser.add_argument("--K_m", type=int, default=16, help="幅度聚类中心数")
    parser.add_argument("--max_iter", type=int, default=25, help="k-means最大迭代次数")
    parser.add_argument("--tol", type=float, default=1e-4, help="k-means收敛阈值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result_json", type=str, default="./attn_analysis/3d_cluster/3d_cluster_result.json")
    parser.add_argument("--future_iter", type=int, default=0, help="参考张量的相对迭代偏移量。0表示自重建，1表示与下一次迭代对比")
    parser.add_argument("--apply_softmax", action="store_true", help="在评估前对输入数据应用 Softmax")
    args = parser.parse_args()
    torch.cuda.manual_seed_all(42)

    evaluator = ThreeDClusterEvaluator(K_v=args.K_v, K_m=args.K_m, max_iter=args.max_iter, tol=args.tol)

    pattern = re.compile(args.file_regex) if args.file_regex else None
    try:
        files = _collect_score_files(args.score_dir)
    except FileNotFoundError:
        print(f"Error: Directory '{args.score_dir}' not found.")
        return

    if not files:
        print(f"No supported score files found under {args.score_dir}")
        return

    file_index: Dict[Tuple[int, int, int], str] = {}
    file_meta_map: Dict[str, Dict[str, int]] = {}
    
    for fpath in files:
        meta = _extract_meta(os.path.basename(fpath), pattern)
        file_meta_map[fpath] = meta
        if "iter" in meta and "layer" in meta and "head" in meta:
            key = (meta["iter"], meta["layer"], meta["head"])
            file_index[key] = fpath

    header = "File | Iter | Layer | Head | Entries | BestScheme | HybridErr | VecMSE | VecCos | MagMSE | MagCos | RecMSE | RecCos | RecHybrid\n"
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    with open(args.log_file, "w", encoding="utf-8") as log:
        log.write(header)
        log.write("-" * len(header) + "\n")
    
    print(f"Found {len(files)} files. Processing with future_iter={args.future_iter}...")

    results = []
    for path in files:
        try:
            score = _load_score(path, args.device)
            if args.apply_softmax:
                score = torch.softmax(score, dim=-1)

            best_results, summaries = evaluator.evaluate(score, l_h=args.B_h, l_w=args.B_w)
            payload = _summarize_outputs(best_results, summaries)
            meta = file_meta_map[path]
            
            rec_mse_val = -1.0
            rec_cos_val = -1.0
            rec_hybrid_val = -1.0
            
            if "iter" in meta and "layer" in meta and "head" in meta:
                target_iter = meta["iter"] + args.future_iter
                target_key = (target_iter, meta["layer"], meta["head"])
                
                ref_path = file_index.get(target_key)
                if ref_path:
                    if args.future_iter == 0:
                        ref_score = score
                    else:
                        ref_score = _load_score(ref_path, args.device)
                        if args.apply_softmax:
                            ref_score = torch.softmax(ref_score, dim=-1)
                    
                    total_mse = 0.0
                    total_cos = 0.0
                    total_hybrid = 0.0
                    count = 0
                    
                    leading = int(ref_score.numel() // (ref_score.shape[-1] * ref_score.shape[-2]))
                    ref_flat = ref_score.reshape(leading, ref_score.shape[-2], ref_score.shape[-1])
                    
                    for idx, res in enumerate(best_results):
                        if idx < leading:
                            _, mse, cos, hybrid = ThreeDClusterEvaluator.recover(ref_flat[idx], res)
                            total_mse += mse
                            total_cos += cos
                            total_hybrid += hybrid
                            count += 1
                    
                    if count > 0:
                        rec_mse_val = total_mse / count
                        rec_cos_val = total_cos / count
                        rec_hybrid_val = total_hybrid / count
                        payload["recover_mse"] = rec_mse_val
                        payload["recover_cos"] = rec_cos_val
                        payload["recover_hybrid"] = rec_hybrid_val
                        payload["ref_file"] = os.path.basename(ref_path)

            entry = {"file": os.path.basename(path), "meta": meta, **payload}
            results.append(entry)
            
            best_overall = min(payload["best"], key=lambda b: b["hybrid_error"])
            
            mse_str = f"{rec_mse_val:.6f}" if rec_mse_val >= 0 else "-"
            cos_str = f"{rec_cos_val:.4f}" if rec_cos_val >= 0 else "-"
            hybrid_str = f"{rec_hybrid_val:.4f}" if rec_hybrid_val >= 0 else "-"
            
            log_line = (
                f"{entry['file']} | {meta.get('iter', '-')}"
                f" | {meta.get('layer', '-')} | {meta.get('head', '-')}"
                f" | {len(payload['best'])} | {best_overall['scheme']} | {best_overall['hybrid_error']:.4f}"
                f" | {best_overall['vec_mse']:.6f} | {best_overall['vec_cos']:.4f}"
                f" | {best_overall['mag_mse']:.6f} | {best_overall['mag_cos']:.4f}"
                f" | {mse_str} | {cos_str} | {hybrid_str}\n"
            )
            print(log_line.strip())
            with open(args.log_file, "a", encoding="utf-8") as log:
                log.write(log_line)
        except Exception as e:
            print(f"Error processing {os.path.basename(path)}: {e}")
            import traceback
            traceback.print_exc()
            with open(args.log_file, "a", encoding="utf-8") as log:
                log.write(f"{os.path.basename(path)} | ERROR: {e}\n")

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump({"files": results}, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved to {args.result_json} and {args.log_file}")


if __name__ == "__main__":
    main()