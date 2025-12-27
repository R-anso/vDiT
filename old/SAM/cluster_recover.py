import os
import re
from typing import Dict, Tuple

import torch


class ClusterRecover:
    def __init__(
        self,
        blueprint_dir: str,
        recover_enabled: bool = True,
        preload_all: bool = False,
        parallelism: int = 1024,
        mag_en: bool = False,
    ):
        """
        仅支持向量聚类蓝图的恢复器。
        """
        self.blueprint_dir = blueprint_dir
        self.recover_enabled = recover_enabled
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.parallelism = parallelism

        if mag_en:
            print("[ClusterRecover] Warning: mag_en=True is no longer supported; ignoring.")
        if self.recover_enabled and preload_all:
            self._preload_blueprints()

    def _preload_blueprints(self) -> None:
        if not os.path.exists(self.blueprint_dir):
            print(f"[ClusterRecover] Warning: Directory {self.blueprint_dir} not found.")
            return

        print(f"[ClusterRecover] Preloading blueprints from {self.blueprint_dir} ...")
        pattern = re.compile(r"blueprint_L(\d+)\.pt")
        count = 0
        for name in os.listdir(self.blueprint_dir):
            match = pattern.match(name)
            if not match:
                continue
            layer_idx = int(match.group(1))
            path = os.path.join(self.blueprint_dir, name)
            try:
                self.cache[layer_idx] = torch.load(path, map_location="cpu")
                count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[ClusterRecover] Failed to load {name}: {exc}")
        print(f"[ClusterRecover] Successfully preloaded {count} layer blueprints.")

    def load_blueprint(
        self,
        layer_idx: int,
        head_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, Dict[str, int]]:
        if not self.recover_enabled:
            raise RuntimeError("Recover functionality is disabled.")

        if layer_idx in self.cache:
            layer_data = self.cache[layer_idx]
        else:
            filename = f"blueprint_L{layer_idx}.pt"
            filepath = os.path.join(self.blueprint_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Blueprint file not found: {filepath}")
            layer_data = torch.load(filepath, map_location="cpu")
            self.cache[layer_idx] = layer_data

        head_key = f"H{head_idx}"
        if head_key not in layer_data:
            raise KeyError(f"Head {head_key} not found in layer {layer_idx} blueprint")

        head_data = layer_data[head_key]

        vec_medoid_indices = head_data["vec_medoids"]
        vec_assign_map = head_data["vec_map"].to(torch.int16)
        if "vec_max_pos" not in head_data:
            raise KeyError(f"'vec_max_pos' missing in blueprint for {head_key}.")
        vec_medoid_max_pos = head_data["vec_max_pos"].to(torch.int16)

        outlier_map = head_data.get("outlier_map")
        if outlier_map is None:
            outlier_map = torch.zeros_like(vec_assign_map, dtype=torch.bool)
        else:
            outlier_map = outlier_map.to(torch.bool)

        norm_mode = head_data.get("norm_mode", "div")
        scheme = head_data.get("scheme", "w-h-f")
        shape_meta = head_data.get("shape_meta", {})

        return (
            vec_medoid_indices,
            vec_assign_map,
            vec_medoid_max_pos,
            outlier_map,
            norm_mode,
            scheme,
            shape_meta,
        )

    def prepare_data(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cluster_info: Tuple[torch.Tensor, ...],
        l_h: int,
        l_w: int,
        l_f: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            vec_medoid_indices,
            vec_assign_map,
            vec_medoid_max_pos,
            outlier_map,
            _norm_mode,
            scheme,
            shape_meta,
        ) = cluster_info

        if shape_meta:
            if (
                l_h != shape_meta.get("l_h")
                or l_w != shape_meta.get("l_w")
                or l_f != shape_meta.get("l_f")
            ):
                raise ValueError("Input spatial sizes mismatch blueprint metadata.")

        device = q.device
        dtype = q.dtype

        vec_medoid_indices = vec_medoid_indices.to(device=device).long()
        vec_medoid_max_pos = vec_medoid_max_pos.to(device=device).long()
        vec_assign_map = vec_assign_map.to(device=device).long()
        outlier_map = outlier_map.to(device=device, dtype=torch.bool)

        safe_assign_map = vec_assign_map.clone()
        safe_assign_map[outlier_map] = 0

        dim_map = {"f": 0, "h": 1, "w": 2}
        perm = [dim_map[d] for d in scheme.split("-")]

        q_3d = q.view(l_f, l_h, l_w, -1)
        k_3d = k.view(l_f, l_h, l_w, -1)
        q_perm = q_3d.permute(*perm, 3)
        k_perm = k_3d.permute(*perm, 3)

        L1, L2, L3 = q_perm.shape[:3]
        S2, S3 = L2 * L2, L3 * L3
        scale = 1.0 / (q.shape[-1] ** 0.5)

        def decode(idx: torch.Tensor, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return idx // length, idx % length

        K_v = vec_medoid_indices.shape[0]
        chunk_size = 512
        partial_s_list, row_max_list, sum_exp_list = [], [], []

        for start in range(0, K_v, chunk_size):
            end = min(start + chunk_size, K_v)
            indices_chunk = vec_medoid_indices[start:end]

            idx_flat_2 = indices_chunk[:, 0]
            idx_flat_3 = indices_chunk[:, 1]

            q_idx_2, k_idx_2 = decode(idx_flat_2, L2)
            q_idx_3, k_idx_3 = decode(idx_flat_3, L3)

            q_sample = q_perm[:, q_idx_2, q_idx_3, :].permute(1, 0, 2)
            k_sample = k_perm[:, k_idx_2, k_idx_3, :].permute(1, 0, 2)

            scores = torch.bmm(q_sample, k_sample.transpose(1, 2)) * scale
            row_max = scores.max(dim=-1, keepdim=True).values
            block_max = row_max.max(dim=1, keepdim=True).values

            row_max_rel = row_max - block_max
            exp_scores = torch.exp(scores - row_max)
            sum_exp_chunk = exp_scores.sum(dim=-1, keepdim=True)

            partial_s_list.append(exp_scores)
            row_max_list.append(row_max_rel)
            sum_exp_list.append(sum_exp_chunk)

        partial_s = torch.cat(partial_s_list, dim=0)
        row_max_val = torch.cat(row_max_list, dim=0)
        sum_exp = torch.cat(sum_exp_list, dim=0)

        idx_flat_2_range = torch.arange(S2, device=device)
        idx_flat_3_range = torch.arange(S3, device=device)

        q_idx_2_range, k_idx_2_range = decode(idx_flat_2_range, L2)
        q_idx_3_range, k_idx_3_range = decode(idx_flat_3_range, L3)

        q_idx_2_grid = q_idx_2_range.unsqueeze(1).expand(-1, S3)
        k_idx_2_grid = k_idx_2_range.unsqueeze(1).expand(-1, S3)
        q_idx_3_grid = q_idx_3_range.unsqueeze(0).expand(S2, -1)
        k_idx_3_grid = k_idx_3_range.unsqueeze(0).expand(S2, -1)

        vec_cluster_ids = safe_assign_map
        max_pos_1 = vec_medoid_max_pos[vec_cluster_ids]
        q_idx_1_grid, k_idx_1_grid = decode(max_pos_1, L1)

        q_sample = q_perm[q_idx_1_grid, q_idx_2_grid, q_idx_3_grid, :]
        k_sample = k_perm[k_idx_1_grid, k_idx_2_grid, k_idx_3_grid, :]
        block_scale = (q_sample * k_sample).sum(dim=-1) * scale
        block_scale = block_scale.to(dtype=dtype)

        outlier_positions = torch.nonzero(outlier_map, as_tuple=False)
        if outlier_positions.numel() == 0:
            outlier_exp = torch.empty(0, L1, L1, device=device, dtype=dtype)
            outlier_row = torch.empty(0, L1, 1, device=device, dtype=dtype)
            outlier_sum = torch.empty(0, L1, 1, device=device, dtype=dtype)
        else:
            num_outliers = outlier_positions.shape[0]
            outlier_exp = torch.empty(num_outliers, L1, L1, device=device, dtype=dtype)
            outlier_row = torch.empty(num_outliers, L1, 1, device=device, dtype=dtype)
            outlier_sum = torch.empty(num_outliers, L1, 1, device=device, dtype=dtype)

            for idx, (flat_2, flat_3) in enumerate(outlier_positions):
                flat_2_int = int(flat_2.item())
                flat_3_int = int(flat_3.item())

                q_idx_2, k_idx_2 = decode(flat_2, L2)
                q_idx_3, k_idx_3 = decode(flat_3, L3)

                q_block = q_perm[:, q_idx_2, q_idx_3, :]
                k_block = k_perm[:, k_idx_2, k_idx_3, :]

                scores = torch.matmul(q_block, k_block.transpose(0, 1)) * scale
                row_max = scores.max(dim=-1, keepdim=True).values
                block_max = row_max.max(dim=0, keepdim=True).values

                row_rel = row_max - block_max
                exp_scores = torch.exp(scores - row_max)
                sum_vals = exp_scores.sum(dim=-1, keepdim=True)

                block_scale[flat_2_int, flat_3_int] = block_max.squeeze()
                outlier_exp[idx] = exp_scores
                outlier_row[idx] = row_rel
                outlier_sum[idx] = sum_vals

        return (
            partial_s,
            row_max_val,
            sum_exp,
            block_scale,
            outlier_exp,
            outlier_row,
            outlier_sum,
            outlier_positions.to(torch.long),
        )

    def recover_matrix(
        self,
        partial_s: torch.Tensor,
        row_max_val: torch.Tensor,
        sum_exp: torch.Tensor,
        block_scale: torch.Tensor,
        outlier_exp: torch.Tensor,
        outlier_row: torch.Tensor,
        outlier_sum: torch.Tensor,
        outlier_positions: torch.Tensor,
        v: torch.Tensor,
        cluster_info: Tuple[torch.Tensor, ...],
        l_h: int,
        l_w: int,
        l_f: int,
    ) -> torch.Tensor:
        (
            _vec_medoid_indices,
            vec_assign_map,
            vec_medoid_max_pos,
            _outlier_map,
            _norm_mode,
            scheme,
            _shape_meta,
        ) = cluster_info

        device = v.device
        dtype = v.dtype

        partial_s = partial_s.to(device=device, dtype=dtype)
        row_max_val = row_max_val.to(device=device, dtype=dtype)
        sum_exp = sum_exp.to(device=device, dtype=dtype)
        block_scale = block_scale.to(device=device, dtype=dtype)

        outlier_exp = outlier_exp.to(device=device, dtype=dtype)
        outlier_row = outlier_row.to(device=device, dtype=dtype)
        outlier_sum = outlier_sum.to(device=device, dtype=dtype)
        outlier_positions = outlier_positions.to(device=device, dtype=torch.long)

        vec_assign_map = vec_assign_map.to(device=device).long()
        vec_medoid_max_pos = vec_medoid_max_pos.to(device=device).long()

        safe_assign_map = vec_assign_map.clone()
        safe_assign_map[vec_assign_map < 0] = 0

        dim_map = {"f": 0, "h": 1, "w": 2}
        perm = [dim_map[d] for d in scheme.split("-")]

        v_3d = v.view(l_f, l_h, l_w, -1)
        v_perm = v_3d.permute(*perm, 3)
        L1, L2, L3 = v_perm.shape[:3]
        D_v = v.shape[1]
        N_grid = L2 * L3

        dim_order = [l_f, l_h, l_w]
        L1_size = dim_order[perm[0]]
        L2_size = dim_order[perm[1]]
        L3_size = dim_order[perm[2]]
        if L1 != L1_size or L2 != L2_size or L3 != L3_size:
            raise ValueError("Blueprint permutation incompatible with runtime geometry.")

        all_v_blocks = v_perm.permute(2, 1, 0, 3).reshape(L3 * L2, L1, D_v)

        grid = torch.arange(N_grid, device=device)
        q_h = grid % L2
        q_f = grid // L2
        k_h = q_h
        k_f = q_f

        idx_flat_h = (q_h.unsqueeze(1) * L2 + k_h.unsqueeze(0)).reshape(-1)
        idx_flat_f = (q_f.unsqueeze(1) * L3 + k_f.unsqueeze(0)).reshape(-1)

        cluster_ids_raw = vec_assign_map[idx_flat_h, idx_flat_f].reshape(N_grid, N_grid)
        cluster_ids_safe = safe_assign_map[idx_flat_h, idx_flat_f].reshape(N_grid, N_grid)

        if outlier_positions.numel() > 0:
            outlier_index_map = torch.full(
                vec_assign_map.shape, -1, dtype=torch.long, device=device
            )
            outlier_index_map[outlier_positions[:, 0], outlier_positions[:, 1]] = torch.arange(
                outlier_positions.shape[0], device=device
            )
            outlier_index_grid = outlier_index_map[idx_flat_h, idx_flat_f].reshape(N_grid, N_grid)
        else:
            outlier_index_grid = None

        block_scale_tensor = block_scale.view(L2, L2, L3, L3)
        block_mags = block_scale_tensor[
            q_h.unsqueeze(1),
            k_h.unsqueeze(0),
            q_f.unsqueeze(1),
            k_f.unsqueeze(0),
        ]

        output_blocks = torch.zeros(N_grid, L1, D_v, device=device, dtype=dtype)

        for i_start in range(0, N_grid, self.parallelism):
            i_end = min(i_start + self.parallelism, N_grid)
            batch_size = i_end - i_start

            batch_cluster_ids = cluster_ids_safe[i_start:i_end]
            batch_ec_all = partial_s[batch_cluster_ids]
            batch_row_max_all = row_max_val[batch_cluster_ids]
            batch_sum_exp_all = sum_exp[batch_cluster_ids]
            batch_block_mags = block_mags[i_start:i_end].unsqueeze(-1).unsqueeze(-1)

            if outlier_index_grid is not None:
                batch_outlier_idx = outlier_index_grid[i_start:i_end]
                override_positions = torch.nonzero(batch_outlier_idx >= 0, as_tuple=False)
                for b_idx, j_idx in override_positions:
                    out_idx = int(batch_outlier_idx[b_idx, j_idx].item())
                    batch_ec_all[b_idx, j_idx] = outlier_exp[out_idx]
                    batch_row_max_all[b_idx, j_idx] = outlier_row[out_idx]
                    batch_sum_exp_all[b_idx, j_idx] = outlier_sum[out_idx]

            batch_real_row_max = batch_row_max_all + batch_block_mags

            curr_max = torch.full((batch_size, L1, 1), float("-inf"), device=device, dtype=dtype)
            curr_denom = torch.zeros(batch_size, L1, 1, device=device, dtype=dtype)
            curr_acc = torch.zeros(batch_size, L1, D_v, device=device, dtype=dtype)

            for j in range(N_grid):
                real_row_max = batch_real_row_max[:, j]
                ec = batch_ec_all[:, j]
                sum_curr = batch_sum_exp_all[:, j]
                v_block = all_v_blocks[j]

                new_max = torch.maximum(curr_max, real_row_max)
                scale_prev = torch.exp(curr_max - new_max)
                scale_curr = torch.exp(real_row_max - new_max)

                curr_denom = curr_denom * scale_prev + sum_curr * scale_curr
                term = torch.matmul(ec, v_block) * scale_curr
                curr_acc = curr_acc * scale_prev + term
                curr_max = new_max

            output_blocks[i_start:i_end] = curr_acc / (curr_denom + 1e-8)

        output_perm = torch.zeros(L1, L2, L3, D_v, device=device, dtype=dtype)
        for idx in range(N_grid):
            h_idx = q_h[idx].item()
            f_idx = q_f[idx].item()
            output_perm[:, h_idx, f_idx, :] = output_blocks[idx]

        inv_perm = [0, 0, 0]
        for axis, p in enumerate(perm):
            inv_perm[p] = axis

        output_3d = output_perm.permute(*inv_perm, 3)
        return output_3d.reshape(v.shape[0], D_v)
    

if __name__ == "__main__":
    torch.manual_seed(42)

    l_h = l_w = l_f = 2
    d_model = 4
    L = l_h * l_w * l_f
    scale = 1.0

    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    v = torch.cat([torch.eye(d_model), torch.eye(d_model)], dim=0)

    vec_medoid_indices = torch.tensor([[0, 0], [3, 3]], dtype=torch.long)
    vec_medoid_max_pos = torch.tensor([0, 3], dtype=torch.long)
    vec_assign_map = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.int16,
    )

    outlier_map = torch.zeros_like(vec_assign_map, dtype=torch.bool)
    outlier_map[1, 0] = True  # force h=1, f=0 block to be treated as an outlier
    vec_assign_map[outlier_map] = -1

    cluster_info = (
        vec_medoid_indices,
        vec_assign_map,
        vec_medoid_max_pos,
        outlier_map,
        "sub",
        "w-h-f",
        {"l_h": l_h, "l_w": l_w, "l_f": l_f},
    )

    recover = ClusterRecover(".", recover_enabled=True, preload_all=False, parallelism=2)
    (
        partial_s,
        row_max_val,
        sum_exp,
        block_scale,
        outlier_exp,
        outlier_row,
        outlier_sum,
        outlier_positions,
    ) = recover.prepare_data(q, k, cluster_info, l_h, l_w, l_f)

    print("Outlier positions (flat_h, flat_f):", outlier_positions.tolist())
    print("Stored cluster ids at outlier:", vec_assign_map[outlier_map].tolist())

    recovered = recover.recover_matrix(
        partial_s,
        row_max_val,
        sum_exp,
        block_scale,
        outlier_exp,
        outlier_row,
        outlier_sum,
        outlier_positions,
        v,
        cluster_info,
        l_h,
        l_w,
        l_f,
    )

    full_scores = torch.matmul(q, k.t()) * scale
    reference = torch.softmax(full_scores, dim=-1).matmul(v)
    max_diff = torch.max(torch.abs(recovered - reference)).item()

    print("Recovery matches reference:", torch.allclose(recovered, reference, atol=1e-5))
    print("Max abs diff:", max_diff)