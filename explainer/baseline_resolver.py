import csv
import copy
import os
import os.path as osp
import subprocess
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import InstanceNorm, global_mean_pool
from torch_geometric.utils import is_undirected, sort_edge_index

from explainer.explainer_utils import create_edge_embeds, sample_graph


ALLOWED_BASE_EXPLAINERS = {
    "confexplainer",
    "eigsearch",
    "goat",
    "pgexplainer",
    "proxyexplainer",
    "mixupexplainer",
    "gsat",
}

_TUNING_SKIP_KEYS = {
    "best_val",
    "combo_slug",
    "elapsed_sec",
    "final_test_auc",
    "gpu",
    "last_val",
    "returncode",
    "seed",
    "status",
    "test_auc",
}


def _repo_root() -> str:
    return osp.abspath(osp.join(osp.dirname(__file__), ".."))


def _baseline_root() -> str:
    return osp.join(_repo_root(), "ICLR_Baselines")


def _abs_path(path: str) -> str:
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(_repo_root(), path))


def _maybe_number(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _score_from_row(row: Dict[str, Any]) -> Optional[float]:
    for key in ("final_test_auc", "test_auc", "best_val", "last_val"):
        value = _maybe_number(row.get(key))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def _load_baseline_defaults() -> Dict[str, Any]:
    cfg_path = osp.join(_baseline_root(), "configs", "config.py")
    defaults: Dict[str, Any] = {}
    namespace: Dict[str, Any] = {}
    argv_backup = list(sys.argv)
    try:
        sys.argv = [cfg_path]
        with open(cfg_path, "r") as f:
            code = compile(f.read(), cfg_path, "exec")
            exec(code, namespace)
    finally:
        sys.argv = argv_backup
    parser_args = namespace["args"]
    defaults.update(vars(parser_args))
    return defaults


def _default_checkpoint_path(dataset: str, base_explainer: str, seed: int) -> str:
    seed_dir = osp.join(_baseline_root(), "param", dataset, base_explainer, str(seed))
    filename = f"{dataset}_{base_explainer}_best_model.pth"
    return osp.join(seed_dir, filename)


def _default_metadata_path(dataset: str, base_explainer: str, seed: int) -> str:
    seed_dir = osp.join(_baseline_root(), "param", dataset, base_explainer, str(seed))
    return osp.join(seed_dir, f"{dataset}_{base_explainer}_resolved_meta.yaml")


def _normalize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in sorted((mapping or {}).items()):
        normalized[key] = _maybe_number(value)
    return normalized


def _metadata_matches(spec: Dict[str, Any]) -> bool:
    metadata_path = spec.get("metadata_path")
    if not metadata_path or not osp.exists(metadata_path):
        return False
    try:
        current = _load_yaml(metadata_path)
    except Exception:
        return False
    expected = {
        "dataset": spec["dataset"],
        "base_explainer": spec["base_explainer"],
        "seed": int(spec["seed"]),
        "model": str(spec.get("model") or ""),
        "hyperparameters": _normalize_mapping(spec.get("hyperparameters") or {}),
        "source": spec.get("source"),
    }
    observed = {
        "dataset": current.get("dataset"),
        "base_explainer": current.get("base_explainer"),
        "seed": _maybe_number(current.get("seed")),
        "model": str(current.get("model") or ""),
        "hyperparameters": _normalize_mapping(current.get("hyperparameters") or {}),
        "source": current.get("source"),
    }
    return observed == expected


def _write_metadata(spec: Dict[str, Any]) -> None:
    metadata_path = spec.get("metadata_path")
    if not metadata_path:
        return
    os.makedirs(osp.dirname(metadata_path), exist_ok=True)
    payload = {
        "dataset": spec["dataset"],
        "base_explainer": spec["base_explainer"],
        "seed": int(spec["seed"]),
        "model": spec.get("model"),
        "hyperparameters": _normalize_mapping(spec.get("hyperparameters") or {}),
        "source": spec.get("source"),
        "checkpoint_path": spec.get("checkpoint_path"),
    }
    with open(metadata_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def _manifest_entry(manifest_path: str, dataset: str, base_explainer: str) -> Dict[str, Any]:
    manifest_abs = _abs_path(manifest_path)
    if not osp.exists(manifest_abs):
        return {}
    data = _load_yaml(manifest_abs)
    overrides = data.get("overrides", data)
    entry = (overrides.get(dataset, {}) or {}).get(base_explainer, {}) or {}
    if not entry or entry.get("enabled", True) is False:
        return {}
    return entry


def _resolve_from_manifest(manifest_path: str, dataset: str, base_explainer: str, seed: int) -> Optional[Dict[str, Any]]:
    entry = _manifest_entry(manifest_path, dataset, base_explainer)
    if not entry:
        return None

    hyperparameters = dict(entry.get("hyperparameters", {}) or {})
    defaults = _load_baseline_defaults()
    model_name = entry.get("model") or hyperparameters.get("model") or defaults.get("model", "GCN")
    checkpoint_path = entry.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = _abs_path(str(checkpoint_path))
    else:
        checkpoint_path = _default_checkpoint_path(dataset, base_explainer, seed)

    return {
        "model": model_name,
        "hyperparameters": hyperparameters,
        "checkpoint_path": checkpoint_path,
        "run_dir": None,
        "source": "manifest",
    }


def _latest_tuning_runs(dataset: str, base_explainer: str) -> list[str]:
    tuning_root = osp.join(_baseline_root(), "runs", "tuning")
    if not osp.isdir(tuning_root):
        return []
    prefix = f"{dataset}_{base_explainer}_"
    return sorted(
        entry for entry in os.listdir(tuning_root)
        if entry.startswith(prefix) and osp.isdir(osp.join(tuning_root, entry))
    )


def _resolve_from_latest_tuning(dataset: str, base_explainer: str, seed: int) -> Optional[Dict[str, Any]]:
    tuning_root = osp.join(_baseline_root(), "runs", "tuning")
    candidates = _latest_tuning_runs(dataset, base_explainer)
    if not candidates:
        return None

    for entry in reversed(candidates):
        run_dir = osp.join(tuning_root, entry)
        csv_path = osp.join(run_dir, "results.csv")
        if not osp.exists(csv_path):
            continue

        best_row = None
        best_score = None
        with open(csv_path, "r", newline="", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = str(row.get("status", "")).strip().lower()
                returncode = str(row.get("returncode", "")).strip()
                if status != "ok":
                    continue
                if returncode not in {"", "0"}:
                    continue
                score = _score_from_row(row)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_row = row

        if best_row is None:
            continue

        hyperparameters = {}
        for key, value in best_row.items():
            if key in _TUNING_SKIP_KEYS:
                continue
            parsed = _maybe_number(value)
            if parsed is not None:
                hyperparameters[key] = parsed

        defaults = _load_baseline_defaults()
        model_name = hyperparameters.get("model") or defaults.get("model", "GCN")

        return {
            "model": str(model_name),
            "hyperparameters": hyperparameters,
            "checkpoint_path": _default_checkpoint_path(dataset, base_explainer, seed),
            "run_dir": run_dir,
            "source": "latest_tuning",
        }

    return None


def _resolve_from_dataset_defaults(dataset: str, base_explainer: str, seed: int) -> Dict[str, Any]:
    defaults = _load_baseline_defaults()
    model_name = defaults.get("model", "GCN")
    return {
        "model": str(model_name),
        "hyperparameters": {},
        "checkpoint_path": _default_checkpoint_path(dataset, base_explainer, seed),
        "run_dir": None,
        "source": "defaults",
    }


def resolve_baseline_spec(
    dataset: str,
    base_explainer: str,
    seed: int,
    manifest_path: str = "configs/beep_baselines.yaml",
    baseline_source: str = "latest_tuning",
) -> Dict[str, Any]:
    base_explainer = str(base_explainer).lower()
    if base_explainer not in ALLOWED_BASE_EXPLAINERS:
        raise ValueError(f"Unsupported base explainer: {base_explainer}")

    spec = _resolve_from_manifest(manifest_path, dataset, base_explainer, seed)
    if spec is None:
        if baseline_source != "latest_tuning":
            raise ValueError(f"Unsupported baseline_source: {baseline_source}")
        spec = _resolve_from_latest_tuning(dataset, base_explainer, seed)
    if spec is None:
        spec = _resolve_from_dataset_defaults(dataset, base_explainer, seed)

    if spec is None:
        raise FileNotFoundError(
            f"Could not resolve baseline for dataset={dataset}, base_explainer={base_explainer}. "
            f"Checked manifest and latest tuning results."
        )

    spec = dict(spec)
    spec["dataset"] = dataset
    spec["base_explainer"] = base_explainer
    spec["seed"] = int(seed)
    spec["metadata_path"] = _default_metadata_path(dataset, base_explainer, seed)
    if spec["source"] in {"latest_tuning", "defaults"}:
        spec["checkpoint_exists"] = osp.exists(spec["checkpoint_path"]) and _metadata_matches(spec)
    else:
        spec["checkpoint_exists"] = osp.exists(spec["checkpoint_path"])
    return spec


def ensure_baseline_checkpoint(spec: Dict[str, Any], gpu: int) -> str:
    checkpoint_path = spec["checkpoint_path"]
    if spec.get("checkpoint_exists"):
        return checkpoint_path

    cmd = [
        sys.executable,
        "main.py",
        "--dataset", str(spec["dataset"]),
        "--explainer_name", str(spec["base_explainer"]),
        "--seed", str(spec["seed"]),
        "--gpu", str(gpu),
    ]

    model_name = spec.get("model")
    if model_name:
        cmd.extend(["--model", str(model_name)])

    for key, value in sorted((spec.get("hyperparameters") or {}).items()):
        if key in {"model"}:
            continue
        if value is None:
            continue
        cmd.extend([f"--{key}", str(value)])

    print("[baseline] Checkpoint missing. Auto-training baseline via:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=_baseline_root(), check=True)

    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Baseline auto-train finished but checkpoint still missing: {checkpoint_path}"
        )
    _write_metadata(spec)
    return checkpoint_path


def score_base_explainer_batch(batch, model, explainer, device):
    if hasattr(explainer, "score_batch"):
        return explainer.score_batch(batch, model, device=device)

    _, _, node_embeds = model(batch)
    edge_embeds = create_edge_embeds(batch.edge_index, node_embeds).unsqueeze(dim=0)
    sampling_weights = explainer(edge_embeds)
    return sample_graph(sampling_weights, device, training=False).squeeze()


class _TrainingFreeEdgeScorer(nn.Module):
    """
    Stateless score adapter for baselines such as EiG-Search and GOAt.
    It emits the same [num_edges] continuous score tensor expected from learned
    explainers, so BEEP can use it for pseudo-labeling without a learned module.
    """

    def __init__(self, dataset: str, base_explainer: str, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.dataset = dataset
        self.base_explainer = str(base_explainer).lower()
        self.params = dict(params or {})
        self.register_buffer("_training_free_adapter", torch.ones(1))

    def forward(self, edge_embeds):
        raise RuntimeError(f"{self.base_explainer} is training-free; call score_batch instead.")

    def load_state_dict(self, state_dict, strict: bool = True):
        marker = state_dict.get("_training_free_adapter") if isinstance(state_dict, dict) else None
        if isinstance(marker, torch.Tensor):
            return super().load_state_dict({"_training_free_adapter": marker}, strict=False)
        return super().load_state_dict({"_training_free_adapter": self._training_free_adapter}, strict=False)

    @staticmethod
    def _ensure_single_graph_batch(data, device):
        data = data.to(device)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        return data

    @staticmethod
    def _normalize(scores: torch.Tensor) -> torch.Tensor:
        if scores.numel() == 0:
            return scores
        scores = torch.nan_to_num(scores.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = scores - scores.min()
        max_val = scores.max()
        if max_val > 0:
            scores = scores / max_val
        return scores.clamp(0.0, 1.0)

    @staticmethod
    def _target_class(data, logits: torch.Tensor) -> int:
        try:
            y = int(getattr(data, "y", None).view(-1)[0].item())
            if 0 <= y < logits.size(-1):
                return y
        except Exception:
            pass
        return int(logits.argmax(dim=-1)[0].item())

    @staticmethod
    def _edge_groups(edge_index: torch.Tensor, undirected: bool) -> list[list[int]]:
        groups: Dict[tuple[int, int], list[int]] = {}
        src, dst = edge_index
        for idx, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            key = (u, v) if not undirected else tuple(sorted((u, v)))
            groups.setdefault(key, []).append(idx)
        return list(groups.values())

    @staticmethod
    def _edge_weights(data, device) -> torch.Tensor:
        if hasattr(data, "edge_weights") and data.edge_weights is not None:
            return data.edge_weights.float().to(device)
        return torch.ones(data.edge_index.size(1), device=device)

    def _gradient_scores(self, data, model, device, goat_transform: bool = False) -> torch.Tensor:
        data = self._ensure_single_graph_batch(data, device)
        num_edges = int(data.edge_index.size(1))
        if num_edges == 0:
            return torch.empty(0, device=device)
        base_weights = self._edge_weights(data, device)
        with torch.enable_grad():
            edge_weights = base_weights.clone().detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            _, logits, _ = model(data, edge_weights=edge_weights)
            target_class = self._target_class(data, logits)
            target_logit = logits[0, target_class]
            grads = torch.autograd.grad(target_logit, edge_weights, retain_graph=False, create_graph=False)[0]
        if goat_transform:
            return self._postprocess_goat_scores(grads, fallback=True)
        return self._normalize(grads.abs())

    def _postprocess_goat_scores(self, scores: torch.Tensor, fallback: bool = False) -> torch.Tensor:
        transform = str(self.params.get("goat_score_transform", "default")).strip().lower()
        if transform == "default":
            scores = scores.abs() if fallback else scores
        elif transform == "raw":
            pass
        elif transform == "abs":
            scores = scores.abs()
        elif transform == "positive":
            scores = scores.clamp_min(0.0)
        elif transform == "negative":
            scores = (-scores).clamp_min(0.0)
        else:
            raise ValueError(f"Unsupported goat_score_transform: {transform}")
        return self._normalize(scores)

    def _eig_scores(self, data, model, device) -> torch.Tensor:
        data = self._ensure_single_graph_batch(data, device)
        num_edges = int(data.edge_index.size(1))
        if num_edges == 0:
            return torch.empty(0, device=device)

        epsilon = float(self.params.get("eig_epsilon", 0.0))
        sparsity = self.params.get("eig_sparsity", 0.7)
        max_edges = self.params.get("eig_max_edges", 200)
        search_max_edges = self.params.get("eig_search_max_edges", 200)
        enable_search = bool(self.params.get("eig_enable_search", True))
        undirected = bool(self.params.get("is_undirected", True))

        if max_edges is not None and num_edges > int(max_edges):
            return self._gradient_scores(data, model, device)

        base_weights = self._edge_weights(data, device)
        with torch.no_grad():
            _, base_logits, _ = model(data, edge_weights=base_weights)
        target_class = self._target_class(data, base_logits)

        raw_scores = torch.zeros(num_edges, device=device)
        for idxs in self._edge_groups(data.edge_index, undirected):
            perturbed = base_weights.clone()
            perturbed[idxs] = epsilon
            with torch.no_grad():
                _, pert_logits, _ = model(data, edge_weights=perturbed)
            denom = torch.linalg.norm(base_weights[idxs] - perturbed[idxs], ord=2).clamp_min(1e-12)
            raw_scores[idxs] = (base_logits - pert_logits)[0, target_class] / denom

        scores = self._normalize(raw_scores)
        best_k = None
        if enable_search and (search_max_edges is None or num_edges <= int(search_max_edges)):
            best_k = self._eig_best_k(data, model, base_logits.detach(), target_class, base_weights, scores, epsilon, sparsity)
        return self._truncate_scores(scores, best_k, sparsity)

    def _eig_best_k(self, data, model, base_logits, target_class, base_weights, scores, epsilon, sparsity):
        num_edges = int(scores.numel())
        if num_edges < 3:
            return None
        ranked = torch.argsort(scores, descending=True)
        if sparsity is not None:
            keep = max(2, int(round(num_edges * (1.0 - float(sparsity)))))
            ranked = ranked[: min(num_edges, keep)]
        if ranked.numel() < 2:
            return int(ranked.numel())

        best_score = float("-inf")
        best_k = None
        for k in range(2, int(ranked.numel()) + 1):
            candidate = ranked[:k]
            minus_weights = base_weights.clone()
            minus_weights[candidate] = epsilon
            with torch.no_grad():
                _, minus_logits, _ = model(data, edge_weights=minus_weights)

            only_weights = torch.zeros_like(base_weights)
            only_weights[candidate] = base_weights[candidate]
            with torch.no_grad():
                _, only_logits, _ = model(data, edge_weights=only_weights)

            fidelity_pos = (base_logits[0, target_class] - minus_logits[0, target_class]).item()
            fidelity_neg = (base_logits[0, target_class] - only_logits[0, target_class]).item()
            score = fidelity_pos - fidelity_neg
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    @staticmethod
    def _truncate_scores(scores: torch.Tensor, best_k: Optional[int], sparsity) -> torch.Tensor:
        num_edges = int(scores.numel())
        if num_edges == 0:
            return scores
        if best_k is not None:
            k = max(1, min(num_edges, int(best_k)))
        elif sparsity is not None:
            k = max(1, min(num_edges, int(round(num_edges * (1.0 - float(sparsity))))))
        else:
            return scores
        if k >= num_edges:
            return scores
        top_idx = torch.topk(scores, k).indices
        pruned = torch.zeros_like(scores)
        pruned[top_idx] = scores[top_idx]
        return pruned

    @staticmethod
    def _goat_run_with_hooks(model, data):
        activations: Dict[str, torch.Tensor] = {}
        handles = []
        target_prefixes = ("conv1", "convs.", "lin1", "lin2")

        def make_hook(name):
            def hook(_, __, output):
                if torch.is_tensor(output):
                    activations[name] = output.detach()
            return hook

        for name, module in model.named_modules():
            if name.startswith(target_prefixes):
                handles.append(module.register_forward_hook(make_hook(name)))

        out = model(data)
        for handle in handles:
            handle.remove()
        return out, activations

    @staticmethod
    def _goat_edge2key(a, b, n):
        return int(a * n + b)

    @staticmethod
    def _goat_next_hop_neigh(edge_index, all_lefts=None, all_rights=None):
        if all_rights is None:
            all_next = []
            for left in all_lefts:
                all_next += torch.nonzero(edge_index[0] == left).view(-1).tolist()
        else:
            all_next = []
            for right in all_rights:
                all_next += torch.nonzero(edge_index[1] == right).view(-1).tolist()
        return edge_index[:, all_next]

    @staticmethod
    def _goat_layer_prop(loc, hidden, propagate, edge_index, x, weights, denoms, act_p):
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.float64)
        if isinstance(hidden, tuple):
            if loc == 0:
                a, b = hidden
                tmp_h1_bef = F.linear(x[b, :].view(1, -1), weights[0])
                tmp_h1 = torch.zeros_like(act_p[0], device=x.device)
                tmp_h1[a] = tmp_h1_bef * act_p[0][a] / denoms[0][a]
                tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight), weights[1])
                tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
                tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight), weights[2])
                tmp_h3 = tmp_h3_bef * act_p[2]
            elif loc == 1:
                tmp_h1_bef = F.linear(propagate(edge_index, x=x, edge_weight=edge_weight), weights[0])
                tmp_h1 = tmp_h1_bef * act_p[0] / denoms[0]
                a, b = hidden
                tmp_h2_bef = F.linear(tmp_h1[b, :].view(1, -1), weights[1])
                tmp_h2 = torch.zeros_like(act_p[1], device=x.device)
                tmp_h2[a] = tmp_h2_bef * act_p[1][a] / denoms[1][a]
                tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight), weights[2])
                tmp_h3 = tmp_h3_bef * act_p[2]
            elif loc == 2:
                tmp_h1_bef = F.linear(propagate(edge_index, x=x, edge_weight=edge_weight), weights[0])
                tmp_h1 = tmp_h1_bef * act_p[0] / denoms[0]
                tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight), weights[1])
                tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
                a, b = hidden
                tmp_h3_bef = F.linear(tmp_h2[b, :].view(1, -1), weights[2])
                tmp_h3 = torch.zeros_like(act_p[2], device=x.device)
                tmp_h3[a] = tmp_h3_bef * act_p[2][a]
        else:
            tmp_h1_bef = F.linear(propagate(edge_index, x=x.double(), edge_weight=edge_weight), weights[0])
            tmp_h1 = tmp_h1_bef * act_p[0] / denoms[0]
            tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight), weights[1])
            tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
            tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight), weights[2])
            tmp_h3 = tmp_h3_bef * act_p[2]

        batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        tmp_pool = global_mean_pool(tmp_h3, batch)
        tmplin1_bef = F.linear(tmp_pool, weights[3])
        tmplin1 = tmplin1_bef * act_p[3]
        tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
        tmplin2 = F.linear(tmplin1, weights[4])
        return [tmp_h1_bef, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]

    @staticmethod
    def _goat_bias_prop(layer, loc, hidden, propagate, edge_index, x, weights, bias, denoms, act_p):
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.float64)
        if layer == 0:
            tmp_h1_bef = bias[0].repeat(x.shape[0], 1)
            tmp_h1 = tmp_h1_bef * act_p[0] / denoms[0]
            if isinstance(hidden, tuple):
                a, b = hidden
                if loc == 1:
                    tmp_h2_bef = F.linear(tmp_h1[b, :].view(1, -1), weights[1])
                    tmp_h2 = torch.zeros_like(act_p[1], device=x.device)
                    tmp_h2[a] = tmp_h2_bef * act_p[1][a] / denoms[1][a]
                    tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight), weights[2])
                    tmp_h3 = tmp_h3_bef * act_p[2]
                elif loc == 2:
                    tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight), weights[1])
                    tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
                    tmp_h3_bef = F.linear(tmp_h2[b, :].view(1, -1), weights[2])
                    tmp_h3 = torch.zeros_like(act_p[2], device=x.device)
                    tmp_h3[a] = tmp_h3_bef * act_p[2][a]
                else:
                    raise RuntimeError("Unexpected loc for GOAt layer 0 bias propagation")
            else:
                tmp_h2_bef = F.linear(propagate(edge_index, x=tmp_h1, edge_weight=edge_weight), weights[1])
                tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
                tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight), weights[2])
                tmp_h3 = tmp_h3_bef * act_p[2]
            batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool, weights[3])
            tmplin1 = tmplin1_bef * act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1, weights[4])
            return [tmp_h1_bef, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]

        if layer == 1:
            if isinstance(hidden, tuple):
                a, b = hidden
                if loc != 2:
                    raise RuntimeError("Unexpected loc for GOAt layer 1 bias propagation")
                tmp_h2_bef = bias[1].repeat(x.shape[0], 1)
                tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
                tmp_h3_bef = F.linear(tmp_h2[b, :].view(1, -1), weights[2])
                tmp_h3 = torch.zeros_like(act_p[2], device=x.device)
                tmp_h3[a] = tmp_h3_bef * act_p[2][a]
            else:
                tmp_h2_bef = bias[1].repeat(x.shape[0], 1)
                tmp_h2 = tmp_h2_bef * act_p[1] / denoms[1]
                tmp_h3_bef = F.linear(propagate(edge_index, x=tmp_h2, edge_weight=edge_weight), weights[2])
                tmp_h3 = tmp_h3_bef * act_p[2]
            batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool, weights[3])
            tmplin1 = tmplin1_bef * act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1, weights[4])
            return [None, tmp_h2_bef, tmp_h3_bef, tmplin1_bef, tmplin2]

        if layer == 2:
            tmp_h3_bef = bias[2].repeat(x.shape[0], 1)
            tmp_h3 = tmp_h3_bef * act_p[2]
            batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            tmp_pool = global_mean_pool(tmp_h3, batch)
            tmplin1_bef = F.linear(tmp_pool, weights[3])
            tmplin1 = tmplin1_bef * act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1, weights[4])
            return [None, None, tmp_h3_bef, tmplin1_bef, tmplin2]

        if layer == 3:
            tmplin1_bef = bias[3].view(1, -1)
            tmplin1 = tmplin1_bef * act_p[3]
            tmplin1 = F.dropout(tmplin1, p=0.5, training=False)
            tmplin2 = F.linear(tmplin1, weights[4])
            return [None, None, None, tmplin1_bef, tmplin2]

        raise RuntimeError("Unsupported GOAt bias layer")

    def _goat_scores(self, data, model, device) -> Optional[torch.Tensor]:
        data = self._ensure_single_graph_batch(data, device)
        num_edges = int(data.edge_index.size(1))
        if num_edges == 0:
            return torch.empty(0, device=device)
        if getattr(model, "use_jk", False):
            return None
        if not hasattr(model, "conv1") or not hasattr(model, "convs") or len(model.convs) != 2:
            return None
        if not hasattr(model, "lin1") or not hasattr(model, "lin2"):
            return None

        required = [
            "conv1.lin.weight",
            "convs.0.lin.weight",
            "convs.1.lin.weight",
            "lin1.weight",
            "lin2.weight",
            "conv1.bias",
            "convs.0.bias",
            "convs.1.bias",
            "lin1.bias",
            "lin2.bias",
        ]
        params = {name: p.data.double() for name, p in model.named_parameters() if p.requires_grad}
        if any(name not in params for name in required):
            return None

        with torch.no_grad():
            (_, logits, _), activation = self._goat_run_with_hooks(model, data)
        pred_cls = int(logits.argmax(dim=-1)[0].item())
        target_class = self._target_class(data, logits)
        if pred_cls != target_class:
            return None
        needed_activations = ("conv1", "convs.0", "convs.1", "lin1", "lin2")
        if any(name not in activation for name in needed_activations):
            return None

        weights = [
            params["conv1.lin.weight"],
            params["convs.0.lin.weight"],
            params["convs.1.lin.weight"],
            params["lin1.weight"],
            params["lin2.weight"],
        ]
        bias = [
            params["conv1.bias"],
            params["convs.0.bias"],
            params["convs.1.bias"],
            params["lin1.bias"],
            params["lin2.bias"],
        ]

        h1 = activation["conv1"].double()
        h2 = activation["convs.0"].double()
        h3 = activation["convs.1"].double()
        lin1out = activation["lin1"].double()
        lin2out = activation["lin2"].double()
        h1_a = (h1 > 0).double()
        h2_a = (h2 > 0).double()
        h3_a = (h3 > 0).double()
        lin1out_a = (lin1out > 0).double()

        eps = 1e-12
        denom_1 = (h1 * h1_a).norm(2, dim=-1, keepdim=True).clamp_min(eps).expand_as(h1 * h1_a)
        denom_2 = (h2 * h2_a).norm(2, dim=-1, keepdim=True).clamp_min(eps).expand_as(h2 * h2_a)
        denoms = [denom_1.double(), denom_2.double()]
        act_p = [h1_a, h2_a, h3_a, lin1out_a]

        off_data = copy.copy(data)
        off_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        if hasattr(off_data, "edge_weights"):
            off_data.edge_weights = torch.empty(0, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, off_activation = self._goat_run_with_hooks(model, off_data)
        if "convs.1" not in off_activation or "lin1" not in off_activation:
            return None
        off_act_p = [
            (off_activation["convs.1"] > 0).double(),
            (off_activation["lin1"] > 0).double(),
        ]

        edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        first_hop_edges = {
            a: data.edge_index[:, (data.edge_index[0] == a).nonzero().view(-1)]
            for a in range(data.x.size(0))
        }
        second_hop_edges, third_hop_edges = {}, {}
        for a in range(data.x.size(0)):
            all_b = list(set(first_hop_edges[a][1].view(-1).tolist()))
            second_hop_edges[a] = self._goat_next_hop_neigh(data.edge_index, all_lefts=all_b)
            all_c = list(set(second_hop_edges[a][1].view(-1).tolist()))
            third_hop_edges[a] = self._goat_next_hop_neigh(data.edge_index, all_lefts=all_c)
        nhop_edges = [first_hop_edges, second_hop_edges, third_hop_edges]

        layer_count = 3
        x = data.x.double()
        edge_overall: Dict[int, torch.Tensor] = {
            self._goat_edge2key(a, b, data.x.size(0)): torch.zeros_like(lin2out)
            for a, b in edges
        }

        for a, b in edges:
            ekey = self._goat_edge2key(a, b, data.x.size(0))
            for loc in range(layer_count):
                out = self._goat_layer_prop(loc, (a, b), model.conv1.propagate, data.edge_index, x, weights, denoms, act_p)
                edge_overall[ekey] += out[-1] / (2 * layer_count + 1)
                for k in range(layer_count - 1):
                    if loc > k:
                        out = self._goat_bias_prop(k, loc, (a, b), model.conv1.propagate, data.edge_index, x, weights, bias, denoms, act_p)
                        edge_overall[ekey] += out[-1] / max(1, 2 * (layer_count - k))

        for act_id in range(layer_count):
            for a in range(data.x.size(0)):
                tmp_act = torch.zeros(act_p[act_id].shape, device=device).double()
                tmp_act[a] = act_p[act_id][a]
                ap = copy.deepcopy(act_p)
                ap[act_id] = tmp_act
                out = self._goat_layer_prop(None, None, model.conv1.propagate, data.edge_index, x, weights, denoms, ap)
                all_edges = []
                for hop in range(act_id + 1):
                    all_edges += torch.transpose(nhop_edges[hop][a].cpu(), 0, 1).tolist()
                for left, right in all_edges:
                    ekey = self._goat_edge2key(left, right, data.x.size(0))
                    edge_overall[ekey] += out[-1] / (2 * layer_count + 1) / max(1, len(all_edges))
                for k in range(layer_count):
                    if act_id >= k:
                        out = self._goat_bias_prop(k, None, None, model.conv1.propagate, data.edge_index, x, weights, bias, denoms, ap)
                        for left, right in all_edges:
                            ekey = self._goat_edge2key(left, right, data.x.size(0))
                            edge_overall[ekey] += out[-1] / max(1, 2 * (layer_count - k)) / max(1, len(all_edges))

        for act_id in range(layer_count, len(act_p)):
            out = self._goat_layer_prop(None, None, model.conv1.propagate, data.edge_index, x, weights, denoms, act_p)
            for ekey in edge_overall:
                edge_overall[ekey] += out[-1] / (2 * layer_count + 1) / max(1, num_edges)
            for k in range(len(act_p)):
                if act_id >= k:
                    out = self._goat_bias_prop(k, None, None, model.conv1.propagate, data.edge_index, x, weights, bias, denoms, act_p)
                    for ekey in edge_overall:
                        edge_overall[ekey] += out[-1] / max(1, 2 * (layer_count - k)) / max(1, num_edges)

        act_id = layer_count - 1
        k = layer_count - 1
        for a in range(data.x.size(0)):
            tmp_act = torch.zeros(off_act_p[0].shape, device=device).double()
            tmp_act[a] = off_act_p[0][a]
            ap = copy.deepcopy(act_p)
            ap[layer_count - 1] = tmp_act
            ap[layer_count] = copy.deepcopy(off_act_p[1])
            all_edges = []
            for hop in range(act_id + 1):
                all_edges += torch.transpose(nhop_edges[hop][a].cpu(), 0, 1).tolist()
            if act_id >= k and all_edges:
                out = self._goat_bias_prop(k, None, None, model.conv1.propagate, data.edge_index, x, weights, bias, denoms, ap)
                for left, right in all_edges:
                    ekey = self._goat_edge2key(left, right, data.x.size(0))
                    edge_overall[ekey] -= out[-1] / max(1, 2 * (layer_count - k)) / len(all_edges)

        for act_id in range(layer_count, len(act_p)):
            ap = copy.deepcopy(act_p)
            ap[layer_count - 1] = copy.deepcopy(off_act_p[0])
            ap[layer_count] = copy.deepcopy(off_act_p[1])
            for k in range(layer_count - 1, len(act_p)):
                if act_id >= k:
                    out = self._goat_bias_prop(k, None, None, model.conv1.propagate, data.edge_index, x, weights, bias, denoms, ap)
                    for ekey in edge_overall:
                        edge_overall[ekey] -= out[-1] / max(1, 2 * (layer_count - k)) / max(1, num_edges)

        overall_score = torch.stack([
            edge_overall[self._goat_edge2key(int(left), int(right), data.x.size(0))].view(-1)
            for left, right in edges
        ])
        if bool(self.params.get("is_undirected", True)):
            aggr = str(self.params.get("goat_undirected_aggr", "mean")).strip().lower()
            undirected_score = overall_score.clone()
            for edge_idx in range(num_edges):
                left = int(data.edge_index[0, edge_idx])
                right = int(data.edge_index[1, edge_idx])
                reverse = (torch.logical_and(data.edge_index[0] == right, data.edge_index[1] == left)).nonzero()
                if reverse.numel() > 0:
                    reverse_idx = int(reverse.view(-1)[0].item())
                    if aggr == "mean":
                        undirected_score[edge_idx] = 0.5 * (overall_score[edge_idx] + overall_score[reverse_idx])
                    elif aggr == "max":
                        undirected_score[edge_idx] = torch.maximum(overall_score[edge_idx], overall_score[reverse_idx])
                    elif aggr == "sum":
                        undirected_score[edge_idx] = overall_score[edge_idx] + overall_score[reverse_idx]
                    else:
                        raise ValueError(f"Unsupported goat_undirected_aggr: {aggr}")
            overall_score = undirected_score
        return self._postprocess_goat_scores(overall_score[:, target_class].to(device), fallback=False)

    def score_batch(self, batch, model, device):
        model.eval()
        data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
        scores = []
        for data in data_list:
            if self.base_explainer == "eigsearch":
                scores.append(self._eig_scores(data, model, device).detach())
            elif self.base_explainer == "goat":
                try:
                    goat_scores = self._goat_scores(data, model, device)
                except Exception:
                    goat_scores = None
                if goat_scores is None:
                    goat_scores = self._gradient_scores(data, model, device, goat_transform=True)
                scores.append(goat_scores.detach())
            else:
                raise ValueError(f"Unsupported training-free base explainer: {self.base_explainer}")
        if not scores:
            return torch.empty(0, device=device)
        return torch.cat(scores, dim=0).to(device)


class _LinearMaskAdapter(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.explainer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, edge_embeds):
        return self.explainer(edge_embeds)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.explainer.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.explainer.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def score_batch(self, batch, model, device):
        _, _, node_embeds = model(batch)
        edge_embeds = create_edge_embeds(batch.edge_index, node_embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer(edge_embeds)
        return sample_graph(sampling_weights, device, training=False).squeeze()


class _GSATExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, width_multiplier: int):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim * width_multiplier),
            InstanceNorm(hidden_dim * width_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * width_multiplier, hidden_dim),
            InstanceNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ])

    def forward(self, embeds: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if embeds.dim() == 3:
            embeds = embeds.squeeze(0)
        if batch is None:
            batch = embeds.new_zeros(embeds.size(0), dtype=torch.long)

        x = embeds
        for module in self.net:
            if isinstance(module, InstanceNorm):
                x = module(x, batch)
            else:
                x = module(x)
        return x


class _GSATAdapter(nn.Module):
    def __init__(self, input_dim: int, extractor_hidden: int, width_multiplier: int, learn_edge_att: bool):
        super().__init__()
        self.learn_edge_att = bool(learn_edge_att)
        self.explainer = _GSATExtractor(
            input_dim=input_dim,
            hidden_dim=extractor_hidden,
            dropout=0.0,
            width_multiplier=width_multiplier,
        )

    def forward(self, embeds, batch=None):
        return self.explainer(embeds, batch)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.explainer.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.explainer.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    @staticmethod
    def _reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        if to_edge_index.numel() == 0:
            return values
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("Edges in from_edge_index and to_edge_index are different.")
        return values[ranking]

    def _edge_attention(self, edge_index, att):
        if is_undirected(edge_index):
            reverse_att = self._reorder_like(edge_index.flip(0), edge_index, att)
            return (att + reverse_att) / 2.0
        return att

    @staticmethod
    def _lift_node_att_to_edge_att(node_att, edge_index):
        return node_att[edge_index[0]] * node_att[edge_index[1]]

    def score_batch(self, batch, model, device):
        _, _, node_embeds = model(batch)
        if self.learn_edge_att:
            edge_embeds = create_edge_embeds(batch.edge_index, node_embeds.detach())
            edge_batch = batch.batch[batch.edge_index[0]]
            att_logits = self.explainer(edge_embeds, edge_batch).view(-1)
            return self._edge_attention(batch.edge_index, att_logits.sigmoid()).view(-1)

        node_logits = self.explainer(node_embeds.detach(), batch.batch).view(-1)
        node_att = node_logits.sigmoid()
        return self._lift_node_att_to_edge_att(node_att, batch.edge_index).view(-1)


class _ConfEdgeExplainer(nn.Module):
    def __init__(self, edge_dim: int, explainer_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, explainer_hidden),
            nn.ReLU(),
            nn.Linear(explainer_hidden, 1),
        )

    def forward(self, edge_embeds):
        return self.net(edge_embeds)


class _ConfExplainerAdapter(nn.Module):
    def __init__(self, hidden_dim: int, explainer_hidden: int, conf_hidden: int):
        super().__init__()
        edge_dim = hidden_dim * 2
        self.explainer = _ConfEdgeExplainer(edge_dim, explainer_hidden)
        self.confidence_model = nn.ModuleDict({
            "edge_encoder": nn.Sequential(
                nn.ReLU(),
                nn.Linear(edge_dim, conf_hidden),
                nn.ReLU(),
            ),
            "score_head": nn.Sequential(
                nn.Linear(conf_hidden + 1, 1),
                nn.Sigmoid(),
            ),
        })

    def forward(self, edge_embeds):
        return self.explainer(edge_embeds)

    def load_state_dict(self, state_dict, strict: bool = True):
        if "explainer" in state_dict:
            explainer_state = state_dict["explainer"]
            if "0.weight" in explainer_state:
                explainer_state = {f"net.{k}": v for k, v in explainer_state.items()}
            explainer_result = self.explainer.load_state_dict(explainer_state, strict=strict)
            if "confidence_model" in state_dict:
                self.confidence_model.load_state_dict(state_dict["confidence_model"], strict=False)
            return explainer_result
        if "0.weight" in state_dict:
            state_dict = {f"net.{k}": v for k, v in state_dict.items()}
        return self.explainer.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return {
            "explainer": self.explainer.state_dict(),
            "confidence_model": self.confidence_model.state_dict(),
        }

    def score_batch(self, batch, model, device):
        _, _, node_embeds = model(batch)
        edge_embeds = create_edge_embeds(batch.edge_index, node_embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer(edge_embeds).squeeze(-1)
        return sample_graph(sampling_weights, device, training=False).view(-1)


def _extract_checkpoint_state(base_explainer: str, checkpoint_obj: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint_obj, dict):
        raise RuntimeError(
            f"Unsupported checkpoint payload for base explainer '{base_explainer}': "
            f"expected dict, got {type(checkpoint_obj).__name__}"
        )

    if base_explainer in {"eigsearch", "goat"}:
        return checkpoint_obj

    if base_explainer == "confexplainer":
        if "explainer" in checkpoint_obj:
            return checkpoint_obj
        return checkpoint_obj

    return checkpoint_obj


def build_base_explainer_adapter(
    dataset: str,
    base_explainer: str,
    checkpoint_path: Optional[str],
    hidden_dim: int,
    in_dim: int,
    num_cls: int,
    use_jk: bool,
    device,
    runtime_args: Optional[Any] = None,
):
    base_explainer = str(base_explainer).lower()

    checkpoint_obj: Dict[str, Any] = {}
    if checkpoint_path:
        checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_checkpoint_state(base_explainer, checkpoint_obj or {})

    if base_explainer in {"eigsearch", "goat"}:
        params = {}
        if isinstance(checkpoint_obj, dict):
            params.update(checkpoint_obj.get("hyperparameters", {}) or {})
        if runtime_args is not None:
            for key in (
                "eig_epsilon",
                "eig_sparsity",
                "eig_max_edges",
                "eig_search_max_edges",
                "eig_enable_search",
                "is_undirected",
                "goat_score_transform",
                "goat_undirected_aggr",
            ):
                value = getattr(runtime_args, key, None)
                if value is not None:
                    params[key] = value
        explainer = _TrainingFreeEdgeScorer(dataset, base_explainer, params=params).to(device)
        explainer.eval()
        return explainer, state_dict

    if base_explainer in {"pgexplainer", "proxyexplainer", "mixupexplainer"}:
        inferred_hidden = int(state_dict["0.weight"].shape[1] // 2)
        explainer = _LinearMaskAdapter(inferred_hidden).to(device)
    elif base_explainer == "gsat":
        first_weight = state_dict.get("net.0.weight")
        second_weight = state_dict.get("net.4.weight")
        if first_weight is None or second_weight is None:
            raise RuntimeError("Unsupported GSAT checkpoint: missing net.0.weight or net.4.weight.")
        input_dim = int(first_weight.shape[1])
        extractor_hidden = int(second_weight.shape[0])
        width_multiplier = int(first_weight.shape[0] // extractor_hidden)
        if width_multiplier < 1 or first_weight.shape[0] % extractor_hidden != 0:
            raise RuntimeError("Unsupported GSAT checkpoint: inconsistent extractor layer shapes.")
        learn_edge_att = input_dim != extractor_hidden
        explainer = _GSATAdapter(
            input_dim=input_dim,
            extractor_hidden=extractor_hidden,
            width_multiplier=width_multiplier,
            learn_edge_att=learn_edge_att,
        ).to(device)
    elif base_explainer == "confexplainer":
        explainer_state = state_dict.get("explainer", state_dict)
        first_weight = explainer_state.get("net.0.weight", explainer_state.get("0.weight"))
        if first_weight is None:
            raise RuntimeError("Unsupported ConfExplainer checkpoint: missing first explainer layer weight.")
        inferred_hidden = int(first_weight.shape[1] // 2)
        explainer_hidden = int(first_weight.shape[0])
        confidence_state = state_dict.get("confidence_model", {})
        edge_encoder_weight = confidence_state.get("edge_encoder.1.weight")
        conf_hidden = int(edge_encoder_weight.shape[0]) if isinstance(edge_encoder_weight, torch.Tensor) else 4
        explainer = _ConfExplainerAdapter(
            hidden_dim=inferred_hidden,
            explainer_hidden=explainer_hidden,
            conf_hidden=conf_hidden,
        ).to(device)
    else:
        raise ValueError(f"Unsupported base explainer: {base_explainer}")

    explainer.load_state_dict(state_dict)
    explainer.eval()
    return explainer, state_dict
