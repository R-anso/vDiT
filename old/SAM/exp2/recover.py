import torch
import os
from typing import Tuple, Dict, Optional
import traceback
import re

class ClusterRecover:
    def __init__(self, blueprint_dir: str, recover_enabled: bool = True, preload_all: bool = False, parallelism: int = 1024, mag_en: bool = False):
        """
        初始化 ClusterRecover 类。
        
        Args:
            blueprint_dir (str): blueprint 文件所在的目录。
            recover_enabled (bool): 是否启用恢复功能。
            preload_all (bool): 是否在初始化时将所有蓝图加载到 CPU 内存中。
            parallelism (int): 恢复矩阵计算时的并行度设置。
        """
        self.blueprint_dir = blueprint_dir
        self.recover_enabled = recover_enabled
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {} # LayerIdx -> LayerData
        self.parallelism = parallelism
        self.mag_en = mag_en       

        if self.recover_enabled and preload_all:
            self._preload_blueprints()

    def _preload_blueprints(self):
        """预加载目录下所有符合规则的蓝图文件到内存"""
        if not os.path.exists(self.blueprint_dir):
            print(f"[ClusterRecover] Warning: Directory {self.blueprint_dir} not found.")
            return

        print(f"[ClusterRecover] Preloading blueprints from {self.blueprint_dir} ...")
        files = os.listdir(self.blueprint_dir)
        pattern = re.compile(r"blueprint_L(\d+)\.pt")
        
        count = 0
        for f in files:
            match = pattern.match(f)
            if match:
                layer_idx = int(match.group(1))
                filepath = os.path.join(self.blueprint_dir, f)
                try:
                    # map_location='cpu' 确保占用的是内存而不是显存
                    self.cache[layer_idx] = torch.load(filepath, map_location='cpu')
                    count += 1
                except Exception as e:
                    print(f"[ClusterRecover] Failed to load {f}: {e}")
        
        print(f"[ClusterRecover] Successfully preloaded {count} layer blueprints.")

    def load_blueprint(self, layer_idx: int, head_idx: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, 
        torch.Tensor, torch.Tensor, torch.Tensor, str, str, Dict[str, int]
    ]:
        """
        读取指定层和头的 blueprint 文件，并返回聚类信息。
        """
        if not self.recover_enabled:
            raise RuntimeError("Recover functionality is disabled.")

        # 1. 尝试从缓存获取
        if layer_idx in self.cache:
            layer_data = self.cache[layer_idx]
        else:
            # 2. 缓存未命中，则从磁盘加载 (Lazy Load)
            filename = f"blueprint_L{layer_idx}.pt"
            filepath = os.path.join(self.blueprint_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Blueprint file not found: {filepath}")
                
            # 加载整个层的 blueprint 到 CPU
            layer_data = torch.load(filepath, map_location='cpu')
            # 存入缓存，避免下次 IO
            self.cache[layer_idx] = layer_data
        
        head_key = f"H{head_idx}"
        if head_key not in layer_data:
            raise KeyError(f"Head {head_key} not found in layer {layer_idx} blueprint")
            
        head_data = layer_data[head_key]
        
        # 提取数据 (此时数据还在 CPU 上)
        vec_medoid_indices = head_data['vec_medoids']
        vec_assign_map = head_data['vec_map'].to(torch.int16)
        
        if 'vec_max_pos' not in head_data:
             raise KeyError(f"'vec_max_pos' missing in blueprint for {head_key}.")
        vec_medoid_max_pos = head_data['vec_max_pos'].to(torch.int16)
        
        mag_medoid_indices = head_data.get('mag_medoids')
        mag_assign_map = head_data.get('mag_map')
        mag_medoid_max_pos = head_data.get('mag_max_pos')

        if self.mag_en and (
            mag_medoid_indices is None
            or mag_assign_map is None
            or mag_medoid_max_pos is None
        ):
            raise KeyError(
                f"Missing magnitude clusters for {head_key} in layer {layer_idx} blueprint."
            )

        if mag_medoid_indices is not None:
            mag_medoid_indices = mag_medoid_indices.to(torch.long)
        if mag_assign_map is not None:
            mag_assign_map = mag_assign_map.to(torch.uint8)
        if mag_medoid_max_pos is not None:
            mag_medoid_max_pos = mag_medoid_max_pos.to(torch.int16)
        
        norm_mode = head_data.get('norm_mode', 'div') # 默认为 div
        scheme = head_data.get('scheme', 'w-h-f') # [Fix] 读取 scheme
        shape_meta = head_data.get('shape_meta', {}) # [Fix] 读取 shape_meta 用于校验
        
        return (
            vec_medoid_indices, 
            vec_assign_map, 
            vec_medoid_max_pos, 
            mag_medoid_indices, 
            mag_assign_map, 
            mag_medoid_max_pos,
            norm_mode,
            scheme,
            shape_meta
        )

    def prepare_data(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cluster_info: Tuple[torch.Tensor, ...],
        l_h: int,
        l_w: int,
        l_f: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            vec_medoid_indices, vec_assign_map, vec_medoid_max_pos,
            mag_medoid_indices, mag_assign_map, mag_medoid_max_pos,
            norm_mode, scheme, shape_meta
        ) = cluster_info

        if shape_meta:
            if l_h != shape_meta.get('l_h') or l_w != shape_meta.get('l_w') or l_f != shape_meta.get('l_f'):
                pass

        device = q.device
        dtype = q.dtype
        mag_available = (
            mag_medoid_indices is not None
            and mag_assign_map is not None
            and mag_medoid_max_pos is not None
        )

        vec_medoid_indices = vec_medoid_indices.to(device).long()
        vec_medoid_max_pos = vec_medoid_max_pos.to(device).long()
        vec_assign_map = vec_assign_map.to(device).long()

        if mag_available:
            mag_medoid_indices = mag_medoid_indices.to(device).long()
            mag_assign_map = mag_assign_map.to(device).long()
            mag_medoid_max_pos = mag_medoid_max_pos.to(device).long()
        else:
            mag_medoid_indices = None
            mag_assign_map = None
            mag_medoid_max_pos = None

        dims = scheme.split('-')
        dim_map = {'f': 0, 'h': 1, 'w': 2}
        perm = [dim_map[d] for d in dims]

        q_3d = q.view(l_f, l_h, l_w, -1)
        k_3d = k.view(l_f, l_h, l_w, -1)
        q_perm = q_3d.permute(*perm, 3)
        k_perm = k_3d.permute(*perm, 3)

        L1, L2, L3 = q_perm.shape[:3]
        S2, S3 = L2 * L2, L3 * L3
        scale = 1.0 / (q.shape[-1] ** 0.5)

        def decode(idx, L):
            return idx // L, idx % L

        K_v = vec_medoid_indices.shape[0]
        chunk_size = 512
        partial_s_list, row_max_val_list, sum_exp_list = [], [], []

        for i in range(0, K_v, chunk_size):
            end = min(i + chunk_size, K_v)
            indices_chunk = vec_medoid_indices[i:end]

            idx_flat_2 = indices_chunk[:, 0]
            idx_flat_3 = indices_chunk[:, 1]

            q_idx_2, k_idx_2 = decode(idx_flat_2, L2)
            q_idx_3, k_idx_3 = decode(idx_flat_3, L3)

            q_sampled = q_perm[:, q_idx_2, q_idx_3, :].permute(1, 0, 2)
            k_sampled = k_perm[:, k_idx_2, k_idx_3, :].permute(1, 0, 2)

            scores = torch.bmm(q_sampled, k_sampled.transpose(1, 2)) * scale
            row_max = scores.max(dim=-1, keepdim=True)[0]
            block_max = row_max.max(dim=1, keepdim=True)[0]

            row_max_rel = row_max - block_max
            exp_scores = torch.exp(scores - row_max)
            sum_exp_chunk = exp_scores.sum(dim=-1, keepdim=True)

            partial_s_list.append(exp_scores)
            row_max_val_list.append(row_max_rel)
            sum_exp_list.append(sum_exp_chunk)

        partial_s = torch.cat(partial_s_list, dim=0)
        row_max_val = torch.cat(row_max_val_list, dim=0)
        sum_exp = torch.cat(sum_exp_list, dim=0)

        if mag_available:
            K_a = mag_medoid_indices.shape[0]
            mag_centers_list = []
            for i in range(0, K_a, chunk_size):
                end = min(i + chunk_size, K_a)
                idx_flat_3_mag = mag_medoid_indices[i:end]
                q_idx_3_mag, k_idx_3_mag = decode(idx_flat_3_mag, L3)

                idx_flat_2_range = torch.arange(S2, device=device)
                q_idx_2_range, k_idx_2_range = decode(idx_flat_2_range, L2)

                current_chunk_size = end - i
                q_idx_3_grid = q_idx_3_mag.unsqueeze(1).expand(-1, S2)
                k_idx_3_grid = k_idx_3_mag.unsqueeze(1).expand(-1, S2)
                idx_flat_3_grid = idx_flat_3_mag.unsqueeze(1).expand(-1, S2)

                q_idx_2_grid = q_idx_2_range.unsqueeze(0).expand(current_chunk_size, -1)
                k_idx_2_grid = k_idx_2_range.unsqueeze(0).expand(current_chunk_size, -1)
                idx_flat_2_grid = idx_flat_2_range.unsqueeze(0).expand(current_chunk_size, -1)

                vec_cluster_ids = vec_assign_map[idx_flat_2_grid, idx_flat_3_grid]
                max_pos_1 = vec_medoid_max_pos[vec_cluster_ids]
                q_idx_1_grid, k_idx_1_grid = decode(max_pos_1, L1)

                q_target = q_perm[q_idx_1_grid, q_idx_2_grid, q_idx_3_grid, :]
                k_target = k_perm[k_idx_1_grid, k_idx_2_grid, k_idx_3_grid, :]
                mag_centers_list.append((q_target * k_target).sum(dim=-1) * scale)

            mag_centers = torch.cat(mag_centers_list, dim=0)
        else:
            mag_centers = torch.empty(0, device=device, dtype=dtype)

        idx_flat_2_range = torch.arange(S2, device=device)
        idx_flat_3_range = torch.arange(S3, device=device)
        q_idx_2_range, k_idx_2_range = decode(idx_flat_2_range, L2)
        q_idx_3_range, k_idx_3_range = decode(idx_flat_3_range, L3)

        q_idx_2_grid = q_idx_2_range.unsqueeze(1).expand(-1, S3)
        k_idx_2_grid = k_idx_2_range.unsqueeze(1).expand(-1, S3)
        q_idx_3_grid = q_idx_3_range.unsqueeze(0).expand(S2, -1)
        k_idx_3_grid = k_idx_3_range.unsqueeze(0).expand(S2, -1)

        vec_cluster_ids_blocks = vec_assign_map
        max_pos_1_blocks = vec_medoid_max_pos[vec_cluster_ids_blocks]
        q_idx_1_blocks, k_idx_1_blocks = decode(max_pos_1_blocks, L1)

        q_sample = q_perm[q_idx_1_blocks, q_idx_2_grid, q_idx_3_grid, :]
        k_sample = k_perm[k_idx_1_blocks, k_idx_2_grid, k_idx_3_grid, :]
        block_scale = (q_sample * k_sample).sum(dim=-1) * scale

        if mag_available:
            mag_cluster_ids = mag_assign_map[idx_flat_3_range]
            max_pos_2 = mag_medoid_max_pos[mag_cluster_ids]
            q_idx_2_scale, k_idx_2_scale = decode(max_pos_2, L2)
            vec_cluster_ids_at_peak = vec_assign_map[max_pos_2, idx_flat_3_range]
            max_pos_1_scale = vec_medoid_max_pos[vec_cluster_ids_at_peak]
            q_idx_1_scale, k_idx_1_scale = decode(max_pos_1_scale, L1)

            q_scale_vec = q_perm[q_idx_1_scale, q_idx_2_scale, q_idx_3_range, :]
            k_scale_vec = k_perm[k_idx_1_scale, k_idx_2_scale, k_idx_3_range, :]
            scale_vec = (q_scale_vec * k_scale_vec).sum(dim=-1) * scale
        else:
            scale_vec = torch.empty(0, device=device, dtype=dtype)

        return partial_s, row_max_val, sum_exp, mag_centers, scale_vec, block_scale
    
    def recover_matrix(
        self,
        partial_s: torch.Tensor,
        row_max_val: torch.Tensor,
        sum_exp: torch.Tensor,
        mag_centers: torch.Tensor,
        scale_vec: torch.Tensor,
        block_scale: Optional[torch.Tensor],
        v: torch.Tensor,
        cluster_info: Tuple[torch.Tensor, ...],
        l_h: int,
        l_w: int,
        l_f: int,
    ) -> torch.Tensor:
        (
            vec_medoid_indices,
            vec_assign_map,
            vec_medoid_max_pos,
            mag_medoid_indices,
            mag_assign_map,
            mag_medoid_max_pos,
            norm_mode,
            scheme,
            shape_meta,
        ) = cluster_info
        device = partial_s.device
        dtype = partial_s.dtype
        L, D_v = v.shape

        dims = scheme.split("-")
        dim_map = {"f": 0, "h": 1, "w": 2}
        perm = [dim_map[d] for d in dims]

        orig_dims = [l_f, l_h, l_w]
        L1 = orig_dims[perm[0]]
        L2 = orig_dims[perm[1]]
        L3 = orig_dims[perm[2]]
        N_grid = L2 * L3

        vec_assign_map = vec_assign_map.to(device=device).long()
        if mag_assign_map is not None:
            mag_assign_map = mag_assign_map.to(device=device).long()
        if mag_medoid_max_pos is not None:
            mag_medoid_max_pos = mag_medoid_max_pos.to(device=device).long()
        if mag_centers is not None and mag_centers.device != device:
            mag_centers = mag_centers.to(device=device, dtype=dtype)
        if scale_vec is not None and scale_vec.device != device:
            scale_vec = scale_vec.to(device=device)

        use_mag = (
            self.mag_en
            and mag_centers is not None
            and mag_centers.numel() > 0
            and mag_assign_map is not None
            and mag_medoid_max_pos is not None
            and scale_vec is not None
            and scale_vec.numel() > 0
        )

        v_3d = v.view(l_f, l_h, l_w, -1)
        v_perm = v_3d.permute(*perm, 3)
        all_v_blocks = v_perm.permute(2, 1, 0, 3).reshape(L3 * L2, L1, D_v)

        grid = torch.arange(N_grid, device=device)
        q_h = grid % L2
        q_f = grid // L2
        k_h = q_h
        k_f = q_f

        idx_flat_h = (q_h.unsqueeze(1) * L2 + k_h.unsqueeze(0)).reshape(-1)
        idx_flat_f = (q_f.unsqueeze(1) * L3 + k_f.unsqueeze(0)).reshape(-1)

        cluster_ids = vec_assign_map[idx_flat_h, idx_flat_f].reshape(N_grid, N_grid)

        if use_mag:
            full_mag_vectors = mag_centers[mag_assign_map]
            anchor_indices = mag_medoid_max_pos[mag_assign_map]
            anchor_vals = full_mag_vectors.gather(1, anchor_indices.unsqueeze(1)).squeeze(1)

            if norm_mode == "div":
                ratio = scale_vec / (anchor_vals + 1e-8)
                real_mag_vectors = full_mag_vectors * ratio.unsqueeze(1)
            else:
                diff = scale_vec - anchor_vals
                real_mag_vectors = full_mag_vectors + diff.unsqueeze(1)

            block_mags = real_mag_vectors[idx_flat_f, idx_flat_h].reshape(N_grid, N_grid)
        elif self.mag_en:
            raise ValueError("mag_en=True 需要有效的幅度聚类信息。")
        else:
            if block_scale is None:
                raise ValueError("mag_en=False 需要提供 block_scale（来自 prepare_data）。")
            block_scale = block_scale.to(device=device, dtype=dtype)
            block_scale_tensor = block_scale.view(L2, L2, L3, L3)
            block_mags = block_scale_tensor[
                q_h.unsqueeze(1), k_h.unsqueeze(0), q_f.unsqueeze(1), k_f.unsqueeze(0)
            ]

        output_blocks = torch.zeros(N_grid, L1, D_v, device=device, dtype=dtype)

        for i_start in range(0, N_grid, self.parallelism):
            i_end = min(i_start + self.parallelism, N_grid)
            batch_size = i_end - i_start
            
            # 初始化当前 batch 的累加器
            curr_max = torch.full((batch_size, L1, 1), float("-inf"), device=device, dtype=dtype)
            curr_denom = torch.zeros(batch_size, L1, 1, device=device, dtype=dtype)
            curr_acc = torch.zeros(batch_size, L1, D_v, device=device, dtype=dtype)
            
            # 提取当前 batch 的 cluster_ids 和 block_mags
            # cluster_ids: [N_grid, N_grid]
            batch_cluster_ids = cluster_ids[i_start:i_end]  # [B, N_grid]
            batch_block_mags = block_mags[i_start:i_end].unsqueeze(-1).unsqueeze(-1)  # [B, N_grid, 1, 1]

            # 预取聚类数据
            batch_ec_all = partial_s[batch_cluster_ids]         # [B, N_grid, L1, L1]
            batch_row_max_all = row_max_val[batch_cluster_ids]  # [B, N_grid, L1, 1]
            batch_sum_exp_all = sum_exp[batch_cluster_ids]      # [B, N_grid, L1, 1]
            batch_real_row_max_all = batch_row_max_all + batch_block_mags  # [B, N_grid, L1, 1]

            for j in range(N_grid):
                real_row_max = batch_real_row_max_all[:, j]
                ec = batch_ec_all[:, j]
                sum_curr = batch_sum_exp_all[:, j]
                v_block = all_v_blocks[j]

                # Online Softmax 更新
                new_max = torch.maximum(curr_max, real_row_max)
                scale_prev = torch.exp(curr_max - new_max)
                scale_curr = torch.exp(real_row_max - new_max)

                curr_denom = curr_denom * scale_prev + sum_curr * scale_curr
                
                # ec @ v_block: [B, L1, L1] @ [L1, D_v] -> [B, L1, D_v]
                # torch.matmul 支持广播
                term = torch.matmul(ec, v_block) * scale_curr
                
                curr_acc = curr_acc * scale_prev + term
                curr_max = new_max

            output_blocks[i_start:i_end] = curr_acc / (curr_denom + 1e-8)

        output_perm = torch.zeros(L1, L2, L3, D_v, device=device, dtype=dtype)
        for idx in range(N_grid):
            h = q_h[idx].item()
            f = q_f[idx].item()
            output_perm[:, h, f, :] = output_blocks[idx]

        inv_perm = [0, 0, 0]
        for axis, p in enumerate(perm):
            inv_perm[p] = axis

        output_3d = output_perm.permute(*inv_perm, 3)
        return output_3d.reshape(L, D_v)
    
if __name__ == "__main__":
    # 测试代码
    torch.manual_seed(42)
    
    print("=== 测试 prepare_data 函数 ===")
    device = "cpu" # 或者 "cuda" 如果可用

    # 1. 定义维度
    l_w, l_h, l_f = 2, 2, 2
    D = 4
    L = l_w * l_h * l_f # 8
    
    # 2. 构造简单的 Q 和 K，使得A矩阵的[0:1][0:1] = [1, 0; 0, 0], [0:1][6:7]=[0,0;0,1]即可。
    q = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],], device=device, dtype=torch.float32)
    k = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],], device=device, dtype=torch.float32)
    
    # 3. 构造 Cluster Info
    # 假设 scheme = "w-h-f" -> permuted dims: f(2), h(2), w(2)
    # L1=2 (w), L2=2 (h), L3=2 (f)
    # S1=4, S2=4, S3=4
    
    K_v = 2
    K_a = 1
    
    # vec_medoid_indices: [K_v, 2] -> 指向 (S2, S3) 网格中的位置
    vec_medoid_indices = torch.tensor([
        [0, 0], # Cluster 0 采样自 (h=0, f=0)
        [1, 1]  # Cluster 1 采样自 (h=1, f=1)
    ], device=device).long()
    
    # vec_assign_map: [S2, S3] -> [4, 4]
    vec_assign_map  = torch.tensor([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ], dtype=torch.int16).to(device)
    
    # vec_medoid_max_pos: [K_v] -> 指向 S1 (w*w=4) 中的位置
    vec_medoid_max_pos = torch.tensor([0, 3], device=device).long()
    
    # mag_medoid_indices: [K_a] -> 指向 S3 (f*f=4) 中的位置
    mag_medoid_indices = torch.tensor([0], device=device).long()
    
    # mag_assign_map: [S3] -> [4]
    mag_assign_map = torch.zeros(4, device=device).long()
    
    # mag_medoid_max_pos: [K_a] -> 指向 S2 (h*h=4) 中的位置
    mag_medoid_max_pos = torch.tensor([2], device=device).long()
    
    norm_mode = 'sub' # 使用减法模式
    scheme = "w-h-f"
    shape_meta = {'l_h': l_h, 'l_w': l_w, 'l_f': l_f}
    
    cluster_info = (
        vec_medoid_indices, vec_assign_map, vec_medoid_max_pos,
        mag_medoid_indices, mag_assign_map, mag_medoid_max_pos,
        norm_mode, scheme, shape_meta
    )
    
    # 4. 实例化并运行
    recover = ClusterRecover(".", recover_enabled=True, preload_all=False, parallelism=1, mag_en=True)
    
    try:
        results = recover.prepare_data(
            q, k, cluster_info, l_h, l_w, l_f
        )
        
        partial_s, row_max_val, sum_exp, mag_centers, scale_vec = results
        
        print("\n--- 数值检查 ---")
        print(f"partial score:\n{partial_s}")
        print(f"row max val:\n{row_max_val}")
        print(f"sum exp:\n{sum_exp}")
        print(f"mag centers:\n{mag_centers}")
        print(f"scale vec:\n{scale_vec}")
    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()


    print("=== 开始手动构造数据测试 recover_matrix ===")
    
    # 1. 定义参数 (参照 recover.md 的例子)
    l_w, l_h, l_f = 2, 2, 2
    L = l_w * l_h * l_f # 8
    D = 4
    K_v = 2
    K_a = 1
    scheme = "w-h-f"
    shape_meta = {'l_h': l_h, 'l_w': l_w, 'l_f': l_f}
    
    # 2. 手动构造 recover_matrix 所需的输入数据
    
    # (1) partial_s [K_v, l_w, l_w] = [2, 2, 2]
    # v0 对应 recover.md 中的 v0 矩阵
    e = torch.exp(torch.tensor(1.0))
    v0_s = torch.tensor([
        [1, 1/e],
        [1, 1]
    ])
    # v1 对应 recover.md 中的 v1 矩阵
    v1_s = torch.tensor([
        [1, 1],
        [1/e, 1]
    ])
    partial_s = torch.stack([v0_s, v1_s])
    
    # (2) row_max_val [K_v, l_w, 1]
    # v0: row0 max=1, row1 max=0. block max=1. -> [0, -1]
    v0_row_max = torch.tensor([[0.0], [-1.0]])
    # v1: row0 max=0, row1 max=1. block max=1. -> [-1, 0]
    v1_row_max = torch.tensor([[-1.0], [0.0]])
    row_max_val = torch.stack([v0_row_max, v1_row_max])
    
    # (3) sum_exp [K_v, l_w, 1]
    # v0: row0 sum=e^0+e^-1 = 1+1/e. row1 sum=e^0+e^0=2.
    v0_sum = torch.tensor([[1 + 1/e], [2.0]])
    # v1: row0 sum=2. row1 sum=1+1/e.
    v1_sum = torch.tensor([[2.0], [1 + 1/e]])
    sum_exp = torch.stack([v0_sum, v1_sum])
    
    # (4) mag_centers [K_a, S2] = [1, 4] (S2 = l_h * l_h = 4)
    # 假设幅度中心向量为 [1, 2, 3, 4]
    mag_centers = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    
    # (5) scale_vec [S3] = [4] (S3 = l_f * l_f = 4)
    # 锚点位置是 2 (index 2, value 3.0 in mag_centers)
    # 目标锚点值是 [3, 6, 9, 12]
    scale_vec = torch.tensor([3.0, 6.0, 9.0, 12.0])
    
    # (6) V 矩阵 [L, D] = [8, 4]，用两个I_4 堆叠
    v = torch.cat([torch.eye(4), torch.eye(4)], dim=0)
    
    # (7) Cluster Info
    # vec_assign_map [S2, S3] = [4, 4]
    # 模拟 "v0, v1, v0, v1" 的模式
    # 横向f增加，纵向h增加
    vec_assign_map = torch.tensor([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ], dtype=torch.int16)
    
    # mag_assign_map [S3] = [4]
    # 所有列都使用幅度聚类 0
    mag_assign_map = torch.zeros(4, dtype=torch.uint8)
    
    # mag_medoid_max_pos [K_a] = [1]
    # 锚点位置在 index 2
    mag_medoid_max_pos = torch.tensor([2], dtype=torch.int16)
    
    # 其他 info 占位
    vec_medoid_indices = torch.zeros((K_v, 2))
    vec_medoid_max_pos = torch.zeros(K_v)
    mag_medoid_indices = torch.zeros(K_a)
    norm_mode = 'sub' # 使用减法模式
    
    cluster_info = (
        vec_medoid_indices, vec_assign_map, vec_medoid_max_pos,
        mag_medoid_indices, mag_assign_map, mag_medoid_max_pos,
        norm_mode, scheme, shape_meta
    )
    
    # 3. 执行测试
    recover = ClusterRecover(".", recover_enabled=True, preload_all=False, parallelism=4, mag_en=False)
    
    block_scale = torch.tensor([
        [1, 4, 7, 10],
        [2, 5, 8, 11],
        [3, 6, 9, 12],
        [4, 7, 10, 13],
    ], dtype=torch.float32)
    output = recover.recover_matrix(
        partial_s, row_max_val, sum_exp, mag_centers, scale_vec, block_scale,
        v, cluster_info, l_h, l_w, l_f
    )
    
    print(f"Output shape: {output.shape}") # Should be [8, 4]
    print("Output sample (first 2 rows):")
    print(output)
    
    real_A = torch.tensor([
                        [1, 0, 1, 1, 4, 3, 4, 4],
                        [0, 0, 1, 2, 3, 3, 4, 5],
                        [3, 2, 3, 3, 6, 5, 6, 6],
                        [2, 2, 3, 4, 5, 5, 6, 7],
                        [7, 6, 7, 7, 10, 9, 10, 10],
                        [6, 6, 7, 8, 9, 9, 10, 11],
                        [9, 8, 9, 9, 12, 11, 12, 12],
                        [8, 8, 9, 10, 11, 11, 12, 13],], dtype=torch.float32)
    real_S = torch.nn.functional.softmax(real_A, dim=-1)
    print("Real S ")
    print(real_S)
    real_output = torch.matmul(real_S, v)
    print("Real Output ")
    print(real_output)