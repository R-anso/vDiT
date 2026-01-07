import os
import json
import traceback
import numpy as np
import torch

class Attn_Info:
    """记录基础注意力位置信息"""

    def __init__(
        self,
        key: str = "cond",
        iter_idx: int = -1,
        layer_idx: int = -1,
        head_idx: list[int] | None = None,
        B_hw: int = 0,
        out_dir: str = "./attn_score/default",
        out_fmt: str = "npy",
        rope_order: str | None = None,
        enable_map: dict[str, bool] | None = None,
    ):
        self.key = key
        self.iter_idx = iter_idx
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.B_hw = B_hw
        self.out_dir = out_dir
        self.out_fmt = out_fmt
        self.rope_order = rope_order
        self.enable_map = self._normalize_enable_map(enable_map)

    @staticmethod
    def _normalize_enable_map(enable_map: dict[str, bool] | None) -> dict[str, bool]:
        if not enable_map:
            return {}
        return {str(k).lower(): bool(v) for k, v in enable_map.items() if k is not None}

    def should_save(self, target: str) -> bool:
        return self.enable_map.get(target.lower(), False)

    def check_save_en(self, head_idx: int) -> bool:
        return self.head_idx is not None and head_idx in self.head_idx

    def get_real_out_dir(self) -> str:
        return f"{self.out_dir}/{self.key}"

    def get_fname(self, this_head, name: str = "score") -> str:
        return f"{name}_It{self.iter_idx}_L{self.layer_idx}_H{this_head}"

    def get_rope_order(self) -> str | None:
        return self.rope_order


class Attn_Save_Cfg:
    """注意力保存配置容器"""

    _SUPPORTED_FORMATS = {"npy", "txt", "json", "pt", "pth"}

    def __init__(
        self,
        enable: dict[str, bool] | bool | None = None,
        key: list[str] | None = None,
        iter_idx: list[int] | None = None,
        layer_idx: list[int] | None = None,
        head_idx: list[int] | None = None,
        B_hw: int = 0,
        out_dir: str = "./attn_mask/default",
        out_fmt: str = "npy",
        rope_order: str | None = None,
    ):
        self.enable_map = self._normalize_enable_map(enable)
        self.enable = self.enable_map  # 兼容旧属性名
        self.key = list(key or [])
        self.B_hw = B_hw
        self.out_dir = out_dir
        self.out_fmt = self._normalize_out_fmt(out_fmt)
        self.iter_idx = list(iter_idx or [])
        self.layer_idx = list(layer_idx or [])
        self.head_idx = list(head_idx or [])
        self.rope_order = self._sanitize_rope_order(rope_order)

    @classmethod
    def _normalize_out_fmt(cls, out_fmt: str) -> str:
        fmt = (out_fmt or "").lower()
        if fmt not in cls._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported out_fmt '{out_fmt}'. Expected one of {sorted(cls._SUPPORTED_FORMATS)}."
            )
        return fmt

    @staticmethod
    def _sanitize_rope_order(rope_order: str | None) -> str | None:
        if not rope_order:
            return None
        tokens = [tok for tok in rope_order.replace(" ", "").lower().split("-") if tok]
        if len(tokens) != 3 or set(tokens) != {"f", "h", "w"}:
            raise ValueError(
                f"Invalid rope_order '{rope_order}'. Expected a permutation like 'f-h-w'."
            )
        return "-".join(tokens)

    @staticmethod
    def _normalize_enable_map(enable: dict[str, bool] | bool | None) -> dict[str, bool]:
        allowed = {"q", "k", "score"}
        if enable is None:
            return {k: False for k in allowed}
        if isinstance(enable, bool):
            return {k: bool(enable) for k in allowed}
        if not isinstance(enable, dict):
            raise TypeError("enable must be a bool or a dict with keys q/k/score.")
        normalized = {str(k).lower(): bool(v) for k, v in enable.items() if k}
        extra = set(normalized) - allowed
        if extra:
            raise ValueError(f"Unknown enable keys: {sorted(extra)}")
        for k in allowed:
            normalized.setdefault(k, False)
        return normalized

    def any_enabled(self) -> bool:
        return any(self.enable_map.values())

    def should_save(self, target: str) -> bool:
        return self.enable_map.get(target.lower(), False)

    def get_enable_map(self) -> dict[str, bool]:
        return dict(self.enable_map)

    def check_save_en(self, key, iter_idx, layer_idx) -> bool:
        return (
            self.any_enabled()
            and key in self.key
            and iter_idx in self.iter_idx
            and layer_idx in self.layer_idx
        )

    def get_head_list(self) -> list[int]:
        return self.head_idx

    def get_rope_order(self) -> str | None:
        return self.rope_order


def save_tensor(
    scores: torch.Tensor | np.ndarray,
    out_dir: str,
    out_fmt: str,
    fname_template: str,
    rope_order: str | None = None,
    component_dims: tuple[int, ...] | None = None,
):
    out_dir = out_dir or "./attn_mask/default"
    fmt = (out_fmt or "npy").lower()
    if fmt not in {"npy", "txt", "json", "pt", "pth"}:
        fmt = "npy"

    def _infer_base_prefix(name: str) -> str:
        stem = name.rsplit(".", 1)[0] if "." in name else name
        stem = stem.split("_", 1)[0]
        return stem or "tensor"

    use_components = bool(rope_order)
    tokens: list[str] = []
    component_map: dict[str, torch.Tensor] | None = None
    base_prefix = _infer_base_prefix(fname_template)

    if use_components:
        order_clean = rope_order.replace(" ", "").lower() if rope_order else ""
        tokens = [tok for tok in order_clean.split("-") if tok]
        if len(tokens) != 3 or set(tokens) != {"f", "h", "w"}:
            raise ValueError(
                f"Invalid rope_order '{rope_order}'. Expected a permutation like 'f-h-w'."
            )

        base_tensor = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores)

        if component_dims is not None:
            if len(component_dims) != 3:
                 raise ValueError("component_dims must have 3 elements for f-h-w.")
            
            parts = torch.split(base_tensor, list(component_dims), dim=-1)
            canonical_parts = dict(zip(("f", "h", "w"), parts))
            
            alias = {key: f"{base_prefix}_{key}" for key in ("f", "h", "w")}
            component_map = {}
            for token in tokens:
                comp_tensor = canonical_parts[token]
                component_map[alias[token]] = comp_tensor.detach().cpu().float()
        
        else:
            if base_tensor.ndim == 0 or base_tensor.shape[0] < 3:
                raise ValueError(
                    "rope_order provided but scores do not contain stacked components at dim 0."
                )

            alias = {key: f"{base_prefix}_{key}" for key in ("f", "h", "w")}
            component_map = {}
            for idx, token in enumerate(tokens):
                comp_tensor = base_tensor[idx]
                if not isinstance(comp_tensor, torch.Tensor):
                    comp_tensor = torch.as_tensor(comp_tensor)
                component_map[alias[token]] = comp_tensor.detach().cpu().float()

    try:
        os.makedirs(out_dir, exist_ok=True)

        score_np = None
        if fmt not in {"pt", "pth"}:
            if use_components and component_map is not None:
                try:
                    canonical = [f"{base_prefix}_{key}" for key in ("f", "h", "w")]
                    score_np = np.stack(
                        [component_map[name].numpy() for name in canonical],
                        axis=0,
                    )
                except ValueError:
                     canonical = [f"{base_prefix}_{key}" for key in ("f", "h", "w")]
                     score_np = np.array([component_map[name].numpy() for name in canonical], dtype=object)

            else:
                if isinstance(scores, torch.Tensor):
                    score_np = scores.detach().cpu().numpy()
                else:
                    score_np = np.asarray(scores)

        fname = fname_template
        if not fname.lower().endswith("." + fmt):
            if "." in fname:
                fname = fname.rsplit(".", 1)[0] + "." + fmt
            else:
                fname = f"{fname}.{fmt}"

        out_path = os.path.join(out_dir, fname)
        tmp_path = out_path + ".tmp"

        try:
            if fmt == "npy":
                with open(tmp_path, "wb") as f:
                    np.save(f, score_np)
            elif fmt == "txt":
                with open(tmp_path, "w", encoding="utf-8") as f:
                    if score_np.dtype == object:
                         f.write("Cannot save object array (variable dimensions) to txt.\n")
                    elif score_np.ndim <= 1:
                        f.write(" ".join(f"{float(v):.8f}" for v in score_np) + "\n")
                    else:
                        for row in score_np.reshape(score_np.shape[0], -1):
                            f.write(" ".join(f"{float(v):.8f}" for v in row) + "\n")
            elif fmt == "json":
                if isinstance(score_np, np.ndarray) and score_np.dtype == object:
                     scores_list = [item.tolist() for item in score_np]
                else:
                     scores_list = score_np.tolist()

                payload = {
                    "shape": score_np.shape,
                    "dtype": str(score_np.dtype),
                    "scores": scores_list,
                    "rope_order": rope_order if use_components else None,
                }
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            elif fmt in {"pt", "pth"}:
                if use_components and component_map is not None:
                    torch.save(component_map, tmp_path)
                else:
                    if isinstance(scores, np.ndarray):
                        scores_to_save = torch.from_numpy(scores)
                    else:
                        scores_to_save = scores
                    torch.save(scores_to_save, tmp_path)

            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as exc:
        print(f"[save_tensor] unexpected error: {exc}\n{traceback.format_exc()}", flush=True)