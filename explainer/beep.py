import os
import os.path as osp
import hashlib
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from gnn.GCN import GCN
from gnn.GIN import GIN
from evaluation.evaluate_auc import evaluate_auc_class
from utils import get_dataset
from classifier import GuidedExplainer, GuidedExplainerGIN
from classification_loss import polar_loss

from .baseline_resolver import (
    build_base_explainer_adapter,
    score_base_explainer_batch,
)

from evaluation.evaluate_f1 import (
    calculate_f1_fixed_threshold,
    compute_bimodality_coefficient,
    compute_polarity,
    evaluate_fidelity,
)

def reset_parameters(module, dataset):
    if isinstance(module, nn.Linear):
        if dataset == "BA3":
            nn.init.xavier_normal_(module.weight)
        else:
            nn.init.xavier_normal_(module.weight, gain=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class beep():
    def __init__(self, args, device):
        self.dataset = args.dataset
        self.device = device
        self._load_dataset_config()
        self.sample_bias = 0.0
        self.explainer = None
        self.base_explainer_name = None
        self._base_score_cache_enabled = False
        self._base_score_cache_path = None
        self._base_score_cache = None

    def _load_dataset_config(self):
        config_path = os.path.join("configs", f"{self.dataset}.yaml")
        with open(config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self.in_dim = dataset_config['in_dim']
        self.num_cls = dataset_config['num_cls']
        self.pool_type = dataset_config['pool_type']
        self.use_jk = dataset_config['use_jk']

    def _score_base_batch(self, batch, model):
        if self._base_score_cache_enabled:
            return self._score_base_batch_with_cache(batch, model)
        return score_base_explainer_batch(batch, model, self.explainer, self.device)

    def _score_base_batch_with_cache(self, batch, model):
        data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
        scores = []
        misses = 0
        for data in data_list:
            graph_key = self._graph_cache_key(data)
            cached = self._base_score_cache["scores"].get(graph_key)
            if cached is None:
                score = score_base_explainer_batch(data.to(self.device), model, self.explainer, self.device)
                cached = score.detach().cpu().float().view(-1)
                self._base_score_cache["scores"][graph_key] = cached
                misses += 1
            scores.append(cached.view(-1).to(self.device))
        if misses:
            self._save_base_score_cache()
        if not scores:
            return torch.empty(0, device=self.device)
        return torch.cat(scores, dim=0)

    @staticmethod
    def _hash_tensor(hasher, tensor):
        if tensor is None:
            hasher.update(b"<none>")
            return
        if not torch.is_tensor(tensor):
            hasher.update(repr(tensor).encode("utf-8"))
            return
        tensor_cpu = tensor.detach().cpu().contiguous()
        hasher.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
        hasher.update(str(tensor_cpu.dtype).encode("utf-8"))
        hasher.update(tensor_cpu.numpy().tobytes())

    def _graph_cache_key(self, data):
        hasher = hashlib.sha256()
        for attr in ("x", "edge_index", "edge_weights", "y"):
            self._hash_tensor(hasher, getattr(data, attr, None))
        return hasher.hexdigest()

    @staticmethod
    def _file_fingerprint(path):
        if not path or not osp.isfile(path):
            return None
        stat = os.stat(path)
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return {"size": int(stat.st_size), "sha256": hasher.hexdigest()}

    def _dataset_processed_fingerprints(self, data_path):
        processed_dir = osp.join(data_path, "processed")
        if not osp.isdir(processed_dir):
            return {}
        fingerprints = {}
        for filename in sorted(os.listdir(processed_dir)):
            if not filename.endswith(".pt"):
                continue
            fingerprints[filename] = self._file_fingerprint(osp.join(processed_dir, filename))
        return fingerprints

    def _cache_param_metadata(self, args, model_name, hidden, nlayers, dropout, model_ckpt, only_pos_flag, data_path):
        cache_seed = 42 if self.base_explainer_name in {"eigsearch", "goat"} else int(args.seed)
        return {
            "cache_version": 2,
            "dataset": self.dataset,
            "dataset_processed_fingerprints": self._dataset_processed_fingerprints(data_path),
            "seed": cache_seed,
            "explainer_name": "beep" if args.explainer_name == "beep_nongib" else args.explainer_name,
            "base_explainer": self.base_explainer_name,
            "model": model_name,
            "hidden": int(hidden),
            "nlayers": int(nlayers),
            "dropout": float(dropout),
            "pool_type": self.pool_type,
            "use_jk": bool(self.use_jk),
            "model_ckpt": model_ckpt,
            "model_ckpt_fingerprint": self._file_fingerprint(model_ckpt),
            "only_pos": bool(only_pos_flag),
            "eig_epsilon": None if getattr(args, "eig_epsilon", None) is None else float(args.eig_epsilon),
            "eig_sparsity": None if getattr(args, "eig_sparsity", None) is None else float(args.eig_sparsity),
            "eig_max_edges": getattr(args, "eig_max_edges", None),
            "eig_search_max_edges": getattr(args, "eig_search_max_edges", None),
            "eig_enable_search": bool(getattr(args, "eig_enable_search", True)),
            "is_undirected": bool(getattr(args, "is_undirected", True)),
            "goat_score_transform": getattr(args, "goat_score_transform", "default"),
            "goat_undirected_aggr": getattr(args, "goat_undirected_aggr", "mean"),
        }

    def _base_score_cache_filename(self, metadata):
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return (
            f"{metadata['dataset']}_{metadata['base_explainer']}_{metadata['model']}_"
            f"seed{metadata['seed']}_{digest}.pt"
        )

    def _base_score_cache_metadata_variants(self, metadata):
        variants = [metadata]
        if metadata.get("explainer_name") == "beep":
            legacy = dict(metadata)
            legacy["explainer_name"] = "beep_nongib"
            variants.append(legacy)
        return variants

    def _load_compatible_base_score_cache(self, cache_dir, metadata):
        cache_root = osp.join(cache_dir, self.dataset, self.base_explainer_name)
        metadata_variants = self._base_score_cache_metadata_variants(metadata)
        candidate_paths = []
        for candidate_metadata in metadata_variants:
            candidate_paths.append(osp.join(cache_root, self._base_score_cache_filename(candidate_metadata)))

        prefix = (
            f"{metadata['dataset']}_{metadata['base_explainer']}_{metadata['model']}_"
            f"seed{metadata['seed']}_"
        )
        if osp.isdir(cache_root):
            for filename in sorted(os.listdir(cache_root)):
                if filename.startswith(prefix) and filename.endswith(".pt"):
                    candidate_paths.append(osp.join(cache_root, filename))

        seen = set()
        for cache_path in candidate_paths:
            if cache_path in seen or not osp.isfile(cache_path):
                continue
            seen.add(cache_path)
            try:
                loaded = torch.load(cache_path, map_location="cpu")
            except Exception as exc:
                print(f"[base-score-cache] Ignoring unreadable cache {cache_path}: {exc}")
                continue

            if not isinstance(loaded, dict) or not isinstance(loaded.get("scores"), dict):
                print(f"[base-score-cache] Ignoring incompatible cache: {cache_path}")
                continue

            loaded_metadata = loaded.get("metadata")
            if loaded_metadata in metadata_variants:
                return cache_path, loaded

            print(f"[base-score-cache] Ignoring incompatible cache: {cache_path}")

        return None, None

    def _setup_base_score_cache(self, args, model_name, hidden, nlayers, dropout, model_ckpt, only_pos_flag, data_path):
        self._base_score_cache_enabled = False
        self._base_score_cache_path = None
        self._base_score_cache = None

        if self.base_explainer_name not in {"eigsearch", "goat"} or bool(getattr(args, "disable_base_score_cache", False)):
            return

        metadata = self._cache_param_metadata(args, model_name, hidden, nlayers, dropout, model_ckpt, only_pos_flag, data_path)
        cache_dir = getattr(args, "base_score_cache_dir", None)
        if cache_dir in (None, ""):
            cache_dir = "edge_score_cache"
            setattr(args, "base_score_cache_dir", cache_dir)

        cache_path = osp.join(cache_dir, self.dataset, self.base_explainer_name, self._base_score_cache_filename(metadata))
        cache = {"metadata": metadata, "scores": {}}

        loaded_path, loaded_cache = self._load_compatible_base_score_cache(cache_dir, metadata)
        if loaded_cache is not None:
            cache_path = loaded_path
            cache = loaded_cache
            print(f"[base-score-cache] Loaded {len(cache['scores'])} graphs from {cache_path}")

        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        self._base_score_cache_enabled = True
        self._base_score_cache_path = cache_path
        self._base_score_cache = cache
        print(f"[base-score-cache] Enabled for {self.base_explainer_name} at {cache_path}")

    def _save_base_score_cache(self):
        if not self._base_score_cache_enabled or not self._base_score_cache_path:
            return
        tmp_path = f"{self._base_score_cache_path}.tmp"
        torch.save(self._base_score_cache, tmp_path)
        os.replace(tmp_path, self._base_score_cache_path)

    def _evaluate_base_auc(self, loader, model):
        all_gt = []
        all_pred = []
        with torch.no_grad():
            self.explainer.eval()
            model.eval()
            for batch in loader:
                batch = batch.to(self.device)
                scores = self._score_base_batch(batch, model)
                edge_gt = batch.edge_gt.squeeze()
                all_gt.extend(edge_gt.detach().cpu().numpy())
                all_pred.extend(scores.detach().cpu().numpy())
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(all_gt, all_pred)

    def _collect_base_scores(self, loader, model):
        scores = []
        labels = []
        with torch.no_grad():
            self.explainer.eval()
            model.eval()
            for batch in loader:
                batch = batch.to(self.device)
                batch_scores = self._score_base_batch(batch, model)
                edge_gt = batch.edge_gt.squeeze()
                scores.extend(batch_scores.detach().cpu().numpy())
                labels.extend(edge_gt.detach().cpu().numpy())
        return np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)

    def _collect_guided_scores(self, loader, guided_explainer):
        scores = []
        labels = []
        with torch.no_grad():
            guided_explainer.eval()
            for batch in loader:
                batch = batch.to(self.device)
                batch_scores = guided_explainer(batch).squeeze()
                if batch_scores.dim() > 1 and batch_scores.size(-1) == 2:
                    batch_scores = torch.softmax(batch_scores, dim=-1)[:, 1]
                edge_gt = batch.edge_gt.squeeze()
                scores.extend(batch_scores.detach().cpu().numpy())
                labels.extend(edge_gt.detach().cpu().numpy())
        return np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)

    @staticmethod
    def _safe_auc(labels, scores):
        from sklearn.metrics import roc_auc_score
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    @staticmethod
    def _safe_f1(scores, labels, threshold):
        if len(labels) == 0:
            return float("nan")
        return float(calculate_f1_fixed_threshold(scores, labels, threshold=threshold))

    @staticmethod
    def _fmt_metric(value):
        return "nan" if value != value else f"{value:.4f}"

    def _evaluate_explainer_metrics(self, test_loader, model, explainer, label, explainer_type):
        if explainer_type == "base":
            scores, gt_labels = self._collect_base_scores(test_loader, model)
        else:
            scores, gt_labels = self._collect_guided_scores(test_loader, explainer)

        valid_mask = np.isin(gt_labels, [0.0, 1.0])
        scores = scores[valid_mask]
        gt_labels = gt_labels[valid_mask]
        threshold = float(np.mean(scores)) if len(scores) > 0 else 0.5
        metrics = {
            "auc": self._safe_auc(gt_labels, scores),
            "f1": self._safe_f1(scores, gt_labels, threshold),
            "bimodality": float(compute_bimodality_coefficient(scores)),
            "binarization": float(compute_polarity(scores)),
            "threshold": threshold,
        }

        try:
            base_score_fn = (
                self._score_base_batch
                if self.base_explainer_name in {"eigsearch", "goat"} and self._base_score_cache_enabled
                else None
            )
            metrics["fidelity"] = float(
                evaluate_fidelity(
                    test_loader,
                    model,
                    self.explainer,
                    explainer,
                    self.device,
                    self.dataset,
                    threshold=threshold,
                    explainer_type=explainer_type,
                    base_score_fn=base_score_fn,
                )
            )
        except ZeroDivisionError:
            metrics["fidelity"] = float("nan")

        self._print_metric_block(label, metrics)
        return metrics

    def _print_metric_block(self, label, metrics):
        print(f"\n===== {label} Metrics =====")
        print(f"{label}_AUC: {self._fmt_metric(metrics['auc'])}")
        print(f"{label}_F1_Score: {self._fmt_metric(metrics['f1'])}")
        print(f"{label}_Bimodality: {self._fmt_metric(metrics['bimodality'])}")
        print(f"{label}_Binarization: {self._fmt_metric(metrics['binarization'])}")
        print(f"{label}_Fidelity: {self._fmt_metric(metrics['fidelity'])}")

        if label == "Base":
            print(f"Base Explainer F1 Score: {self._fmt_metric(metrics['f1'])}")
            print(f"Base_Bimodality: {self._fmt_metric(metrics['bimodality'])}")
            print(f"Base_Polarity: {self._fmt_metric(metrics['binarization'])}")
            print(f"Base Explainer Fidelity: {self._fmt_metric(metrics['fidelity'])}")
        else:
            print(f"{label}_Guided Explainer F1 Score: {self._fmt_metric(metrics['f1'])}")
            print(f"{label}_Bimodality: {self._fmt_metric(metrics['bimodality'])}")
            print(f"{label}_Polarity: {self._fmt_metric(metrics['binarization'])}")
            print(f"{label}_Guided Explainer Fidelity: {self._fmt_metric(metrics['fidelity'])}")

    def _print_metrics_summary(self, metrics_by_stage):
        print("\n===== Final Metrics Summary =====")
        header = f"{'Stage':<8} {'AUC':>8} {'F1':>8} {'Bimod':>8} {'Binar':>8} {'Fidelity':>9}"
        print(header)
        print("-" * len(header))
        for label, metrics in metrics_by_stage:
            print(
                f"{label:<8} "
                f"{self._fmt_metric(metrics['auc']):>8} "
                f"{self._fmt_metric(metrics['f1']):>8} "
                f"{self._fmt_metric(metrics['bimodality']):>8} "
                f"{self._fmt_metric(metrics['binarization']):>8} "
                f"{self._fmt_metric(metrics['fidelity']):>9}"
            )

    def _base_checkpoint_filenames(self, dataset, base_explainer):
        return [
            f"{dataset}_{base_explainer}_best.pth",
            f"{dataset}_{base_explainer}_best_model.pth",
        ]

    def _resolve_required_base_checkpoint(self, dataset, base_explainer, seed, explicit_path=None):
        if explicit_path:
            if osp.isfile(explicit_path):
                return explicit_path
            raise FileNotFoundError(f"Explicit base checkpoint path does not exist: {explicit_path}")

        filenames = self._base_checkpoint_filenames(dataset, base_explainer)
        candidates = []
        for filename in filenames:
            candidates.append(osp.join("best_base_param", dataset, base_explainer, str(seed), filename))
            candidates.append(osp.join("best_base_param", dataset, base_explainer, filename))
            candidates.append(osp.join("ICLR_Baselines", "best_params", dataset, base_explainer, str(seed), filename))
            candidates.append(osp.join("ICLR_Baselines", "param", dataset, base_explainer, str(seed), filename))
            candidates.append(osp.join("param", dataset, base_explainer, str(seed), filename))

        for path in candidates:
            if osp.isfile(path):
                return path

        if base_explainer in {"goat", "eigsearch"}:
            print(
                f"[base] No {base_explainer} checkpoint artifact found; "
                "using stateless training-free adapter."
            )
            return None

        expected = "\n  - ".join(candidates)
        raise FileNotFoundError(
            "beep requires a pre-existing base explainer checkpoint. "
            "Automatic training/resolution from ICLR_Baselines is disabled.\n"
            f"Expected filenames: {', '.join(filenames)}\n"
            f"Checked:\n  - {expected}"
        )

    def _write_norm_metadata(self, output_dir, args, base_explainer_name, base_ckpt_path, model_ckpt):
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "explainer_name": args.explainer_name,
            "base_explainer": base_explainer_name,
            "seed": int(args.seed),
            "base_checkpoint": base_ckpt_path,
            "gnn_checkpoint": model_ckpt,
            "args": dict(sorted(vars(args).items())),
        }
        metadata_path = osp.join(output_dir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=True)
        print(f"[Visualization] Metadata saved to {metadata_path}")
    
    def train_test(self, args):
        batch_size = args.batch_size
        model_name = args.model
        nlayers = args.nlayers
        hidden = args.hidden
        dropout = args.dropout
        
        round1_epochs = args.round1_epochs
        round1_lr = args.round1_lr
        
        print("Class hyperparameter information")
        print(args.__dict__)
        
        dataset = self.dataset
        data_path = f"data/{dataset}"
        dataset_tag = dataset
        train_dataset, val_dataset, test_dataset, num_cls = get_dataset(
            data_path,
            dataset,
        )
        total_training_time = 0.0
        total_inference_time = 0.0
        total_pseudo_time = 0.0
        round1_train_time = 0.0
        round2_train_time = 0.0
        training_steps = 0
        inference_steps = 0
        pseudo_steps = 0

        # Optional: keep only graphs with label 1 when configured
        only_pos_flag = getattr(args, 'only_pos', None)
        if only_pos_flag is None:
            if dataset == "FC":
                only_pos_flag = bool(getattr(args, 'fc_only_pos', False))
            else:
                only_pos_flag = False
        else:
            only_pos_flag = bool(only_pos_flag)

        if dataset == "FC" and only_pos_flag:
            def _is_pos_graph(g):
                y = getattr(g, 'y', None)
                if y is None:
                    return False
                if isinstance(y, torch.Tensor):
                    if y.numel() == 0:
                        return False
                    return int(y.view(-1)[0].item()) == 1
                try:
                    return int(y) == 1
                except Exception:
                    return False

            orig_tv = (len(train_dataset), len(val_dataset), len(test_dataset))
            train_dataset = [g for g in train_dataset if _is_pos_graph(g)]
            val_dataset = [g for g in val_dataset if _is_pos_graph(g)]
            test_dataset = [g for g in test_dataset if _is_pos_graph(g)]
            print(
                f"[{dataset}-only-pos] Using only y==1 graphs. sizes train/val/test: "
                f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} "
                f"(from {orig_tv[0]}/{orig_tv[1]}/{orig_tv[2]})"
            )
        
        if model_name == 'GCN':
            model = GCN(train_dataset[0].x.shape[1], self.num_cls, nlayers, hidden, dropout, self.pool_type, self.use_jk)
        elif model_name == 'GIN':
            model = GIN(train_dataset[0].x.shape[1], self.num_cls, nlayers, hidden, dropout, self.pool_type)
            
        model_path = f"param/{model_name}"
        model_ckpt = osp.join(model_path, f"{dataset_tag}_{model_name}_best_val.pth")
        model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=len(val_dataset), shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)
        
        base_explainer_name = str(getattr(args, "base_explainer", "pgexplainer")).lower()
        self.base_explainer_name = base_explainer_name
        base_ckpt_path = self._resolve_required_base_checkpoint(
            dataset=dataset_tag,
            base_explainer=base_explainer_name,
            seed=args.seed,
            explicit_path=getattr(args, "base_checkpoint_path", None),
        )
        if base_ckpt_path is None:
            print(f"[base] Initializing training-free base explainer: {base_explainer_name}")
        else:
            print(f"[base] Loading required base explainer checkpoint from {base_ckpt_path}")

        self.explainer, _ = build_base_explainer_adapter(
            dataset=dataset_tag,
            base_explainer=base_explainer_name,
            checkpoint_path=base_ckpt_path,
            hidden_dim=hidden,
            in_dim=self.in_dim,
            num_cls=self.num_cls,
            use_jk=self.use_jk,
            device=self.device,
            runtime_args=args,
        )
        self._setup_base_score_cache(
            args=args,
            model_name=model_name,
            hidden=hidden,
            nlayers=nlayers,
            dropout=dropout,
            model_ckpt=model_ckpt,
            only_pos_flag=only_pos_flag,
            data_path=data_path,
        )

        base_val_auc = self._evaluate_base_auc(val_loader, model)
        print(f"Loaded base explainer validation AUC: {base_val_auc:.4f}")
        print(f"==> [base End] Base Explainer restored (Val AUC={base_val_auc:.4f})")

        base_test_start = time.time()
        base_metrics = self._evaluate_explainer_metrics(
            test_loader, model, self.explainer, "Base", explainer_type="base"
        )
        base_test_inference_time = time.time() - base_test_start
        print(f"Base AUC of test dataset: {self._fmt_metric(base_metrics['auc'])}")
        metrics_history = [("Base", base_metrics)]

        save_path = osp.join("param", dataset_tag, args.explainer_name, base_explainer_name)
        os.makedirs(save_path, exist_ok=True)
        seed_dir = osp.join(save_path, str(args.seed))
        os.makedirs(seed_dir, exist_ok=True)
        phase1_ckpt_path = osp.join(seed_dir, f"{dataset_tag}_{args.explainer_name}_phase1_explainer.pth")
        torch.save(self.explainer.state_dict(), phase1_ckpt_path)
        print(f"[base] Saved phase1 explainer checkpoint to {phase1_ckpt_path}")
                
        # Set quantile pseudo-labeling hyperparameters from args
        try:
            self.pseudo_alpha0 = float(getattr(args, 'alpha0', None) or 0.10)
            self.pseudo_skew_c = float(getattr(args, 'c', None) or 0.50)
        except Exception:
            self.pseudo_alpha0 = 0.10
            self.pseudo_skew_c = 0.50
        print(f"Quantile pseudo-labeling params: alpha0={self.pseudo_alpha0}, c={self.pseudo_skew_c}")
        
        if self.dataset == "FC":                   
            self.explainer_round1 = GuidedExplainerGIN(self.in_dim, hidden_dim = hidden, out_channels = 2).to(self.device)
        else:
            self.explainer_round1 = GuidedExplainer(self.in_dim, hidden_dim = hidden, out_channels = 2).to(self.device)
        self.explainer_round1.apply(lambda module: reset_parameters(module, self.dataset))

        self.optimizer_explainer_round1 = torch.optim.Adam(self.explainer_round1.parameters(), lr=round1_lr)

        best_round1_auc = 0.0
        best_round1_state = None
        patience = 30
        epochs_no_improve = 0

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        overall_training_start = time.time()

        pseudo_label_start = time.time()
        pseudo_labeled_data_list = self._generate_pseudo_labeled_dataset(train_loader, model)
        pseudo_label_duration = time.time() - pseudo_label_start
        print(f"[round1][pseudo] generation_time={pseudo_label_duration:.4f}s for {len(pseudo_labeled_data_list)} graphs")
        total_pseudo_time += pseudo_label_duration
        pseudo_steps += 1
        new_train_loader = DataLoader( pseudo_labeled_data_list, batch_size=args.batch_size, shuffle=False)

        best_round1_auc = 0.0
        best_round1_state = None
        epochs_no_improve = 0
        ### Round 1: Guided Explainer with pseudo-labels from base explainer
        for epoch in range(round1_epochs):
            train_start_r1 = time.time()
            self.explainer_round1.train()
            epoch_loss_r1 = 0.0

            for new_batch in new_train_loader:
                new_batch = new_batch.to(self.device)
                prob = self.explainer_round1(new_batch)

                labeled_edge_mask = (new_batch.edge_gt.squeeze() != -1.0)
                if labeled_edge_mask.sum() == 0:
                    continue

                loss = polar_loss(prob[labeled_edge_mask], new_batch.edge_gt[labeled_edge_mask], pos_weight=args.w)
                self.optimizer_explainer_round1.zero_grad()
                loss.backward()
                self.optimizer_explainer_round1.step()

                epoch_loss_r1 += loss.item()

            training_duration_r1 = time.time() - train_start_r1

            val_start_r1 = time.time()
            val_auc_r1 = evaluate_auc_class(val_loader, self.explainer_round1, device=self.device)
            val_duration_r1 = time.time() - val_start_r1

            print(
                f"[round1 | epoch {epoch}/{round1_epochs}] loss={epoch_loss_r1:.4f}, "
                f"ValAUC={val_auc_r1:.4f}, training_time={training_duration_r1:.4f}s, "
                f"inference_time={val_duration_r1:.4f}s"
            )
            total_training_time += training_duration_r1
            round1_train_time += training_duration_r1
            training_steps += 1
            total_inference_time += val_duration_r1
            inference_steps += 1

            if val_auc_r1 > best_round1_auc:
                best_round1_auc = val_auc_r1
                best_round1_state = {k: v.clone() for k, v in self.explainer_round1.state_dict().items()}

                ckpt_dir = seed_dir
                ckpt_path = osp.join(ckpt_dir, f"{dataset_tag}_{args.explainer_name}_best_model.pth")
                torch.save(best_round1_state, ckpt_path)
                print(f"  >> [round1] Best Explainer updated (AUC={best_round1_auc:.4f}) - saved to {ckpt_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  >> Early stopping triggered (no improvement for {patience} epochs).")
                    break

        if best_round1_state is not None:
            self.explainer_round1.load_state_dict(best_round1_state)
            print(f"==> [round1 End] Best restored (Val AUC={best_round1_auc:.4f})")

        test_start_r1 = time.time()
        round1_metrics = self._evaluate_explainer_metrics(
            test_loader, model, self.explainer_round1, "Round_1", explainer_type="guided"
        )
        test_inference_time_r1 = time.time() - test_start_r1
        metrics_history.append(("Round_1", round1_metrics))
        print(f"AUC of test dataset: {self._fmt_metric(round1_metrics['auc'])}")
        
        print("\n===== round 2: Second Guided Explainer (fresh pseudo-labels from Guided scores) =====")

        # 1) Collect edge scores from the trained guided explainer (round 1)
        print("Collecting guided edge score distribution from round 1 model...")
        guided_scores = []
        with torch.no_grad():
            self.explainer_round1.eval()
            for batch in DataLoader(train_dataset, batch_size=16, shuffle=False):
                batch = batch.to(self.device)
                prob = self.explainer_round1(batch)  # [E]
                guided_scores.extend(prob.detach().cpu().numpy().tolist())

        if len(guided_scores) == 0:
            print("[round 2] No guided scores collected; skipping second guided stage.")
            self._print_metrics_summary(metrics_history)
            return round1_metrics["auc"]

        guided_scores = np.array(guided_scores, dtype=np.float64)
        self.pseudo_alpha0 = float(getattr(args, 'alpha0', self.pseudo_alpha0))
        self.pseudo_skew_c = float(getattr(args, 'c', self.pseudo_skew_c))
        print(f"[round 2] Quantile params: alpha0={self.pseudo_alpha0}, c={self.pseudo_skew_c}")

        round2_label_start = time.time()
        pseudo_labeled_round2 = self._generate_pseudo_labeled_dataset_from_guided(
            train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        )
        round2_label_duration = time.time() - round2_label_start
        print(f"[round2][pseudo] generation_time={round2_label_duration:.4f}s for {len(pseudo_labeled_round2)} graphs")
        total_pseudo_time += round2_label_duration
        pseudo_steps += 1

        round2_loader = DataLoader(pseudo_labeled_round2, batch_size=args.batch_size, shuffle=False)

        if self.dataset == "FC":
            self.explainer_round2 = GuidedExplainerGIN(self.in_dim, hidden_dim=hidden, out_channels=2).to(self.device)
        else:
            self.explainer_round2 = GuidedExplainer(self.in_dim, hidden_dim=hidden, out_channels=2).to(self.device)

        if best_round1_state is not None:
            self.explainer_round2.load_state_dict(best_round1_state)
        else:
            self.explainer_round2.apply(lambda module: reset_parameters(module, self.dataset))

        round2_lr = float(getattr(args, 'second_phase_lr', None) or round1_lr)
        round2_epochs = int(getattr(args, 'second_phase_epochs', None) or round1_epochs)
        w2 = float(args.w)
        optimizer_round2 = torch.optim.Adam(self.explainer_round2.parameters(), lr=round2_lr)

        best_round2_auc = 0.0
        best_round2_state = None
        patience3 = 30
        epochs_no_improve3 = 0

        print(f"[round 2] Training with lr={round2_lr}, epochs={round2_epochs}, w={w2}")
        for epoch in range(round2_epochs):
            train_start_r2 = time.time()
            self.explainer_round2.train()
            epoch_loss_r2 = 0.0

            for batch3 in round2_loader:
                batch3 = batch3.to(self.device)
                prob3 = self.explainer_round2(batch3)
                labeled_edge_mask3 = (batch3.edge_gt.squeeze() != -1.0)
                if labeled_edge_mask3.sum() == 0:
                    continue

                loss3 = polar_loss(prob3[labeled_edge_mask3], batch3.edge_gt[labeled_edge_mask3], pos_weight=w2)

                optimizer_round2.zero_grad()
                loss3.backward()
                optimizer_round2.step()
                epoch_loss_r2 += loss3.item()

            training_duration_r2 = time.time() - train_start_r2

            val_start_r2 = time.time()
            val_auc_r2 = evaluate_auc_class(val_loader, self.explainer_round2, device=self.device)
            val_duration_r2 = time.time() - val_start_r2

            print(
                f"[round2 | epoch {epoch}/{round2_epochs}] loss={epoch_loss_r2:.4f}, "
                f"ValAUC={val_auc_r2:.4f}, training_time={training_duration_r2:.4f}s, "
                f"inference_time={val_duration_r2:.4f}s"
            )
            total_training_time += training_duration_r2
            round2_train_time += training_duration_r2
            training_steps += 1
            total_inference_time += val_duration_r2
            inference_steps += 1

            if val_auc_r2 > best_round2_auc:
                best_round2_auc = val_auc_r2
                best_round2_state = {k: v.clone() for k, v in self.explainer_round2.state_dict().items()}
                ckpt_dir = seed_dir
                ckpt_path3 = osp.join(ckpt_dir, f"{dataset_tag}_{args.explainer_name}_best_model_second.pth")
                torch.save(best_round2_state, ckpt_path3)
                print(f"  >> [round2] Best Explainer updated (AUC={best_round2_auc:.4f}) - saved to {ckpt_path3}")
                epochs_no_improve3 = 0
            else:
                epochs_no_improve3 += 1
                if epochs_no_improve3 >= patience3:
                    print(f"  >> [round2] Early stopping (no improvement for {patience3} epochs).")
                    break

        if best_round2_state is not None:
            self.explainer_round2.load_state_dict(best_round2_state)
            print(f"==> [round2 End] Best restored (Val AUC={best_round2_auc:.4f})")

        overall_training_time = time.time() - overall_training_start

        test_start_r2 = time.time()
        round2_metrics = self._evaluate_explainer_metrics(
            test_loader, model, self.explainer_round2, "Round_2", explainer_type="guided"
        )
        test_inference_time_r2 = time.time() - test_start_r2
        metrics_history.append(("Round_2", round2_metrics))
        print(f"AUC of test dataset (second guided): {self._fmt_metric(round2_metrics['auc'])}")

        if training_steps > 0:
            avg_training_time = total_training_time / training_steps
        else:
            avg_training_time = 0.0
        if inference_steps > 0:
            avg_inference_time = total_inference_time / inference_steps
        else:
            avg_inference_time = 0.0
        if pseudo_steps > 0:
            avg_pseudo_time = total_pseudo_time / pseudo_steps
        else:
            avg_pseudo_time = 0.0

        end_to_end_training_time = overall_training_time
        total_test_inference_time = base_test_inference_time + test_inference_time_r1 + test_inference_time_r2

        print(
            f"\n[Timing Summary] "
            f"end_to_end_training_time={end_to_end_training_time:.4f}s "
            f"(pseudo={total_pseudo_time:.4f}s, round1_train={round1_train_time:.4f}s, round2_train={round2_train_time:.4f}s); "
            f"avg_training_time={avg_training_time:.4f}s over {training_steps} steps; "
            f"avg_inference_time={avg_inference_time:.4f}s over {inference_steps} steps; "
            f"avg_pseudo_time={avg_pseudo_time:.4f}s over {pseudo_steps} steps; "
            f"test_inference_base={base_test_inference_time:.4f}s, "
            f"test_inference_round1={test_inference_time_r1:.4f}s, "
            f"test_inference_round2={test_inference_time_r2:.4f}s, "
            f"total_test_inference={total_test_inference_time:.4f}s"
        )

        self._print_metrics_summary(metrics_history)
        
        return round2_metrics["auc"]
    
    def _generate_pseudo_labeled_dataset(self, train_loader, model):
        # Hyperparameters with safe defaults, can be overridden by attributes if present
        alpha0 = float(getattr(self, 'pseudo_alpha0', 0.10))  # base cutoff in (0, 0.5)
        skew_c = float(getattr(self, 'pseudo_skew_c', 0.50))  # skewness adjustment factor > 0

        pseudo_labeled_data_list = []
        self.explainer.eval()
        model.eval()

        # Pass 1: collect all base mask scores to estimate quantiles and skewness
        all_scores = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                mask_scores = self._score_base_batch(batch, model)
                all_scores.append(mask_scores.detach().cpu())

        if len(all_scores) == 0:
            print("[Pseudo-labeling] No scores collected; returning empty list.")
            return []

        all_scores = torch.cat(all_scores, dim=0).float()
        # Clamp into [0,1] just in case of numerical drift
        all_scores = torch.clamp(all_scores, 0.0, 1.0)

        # Skewness computation (population moments style)
        mu = float(all_scores.mean().item())
        sigma = float(all_scores.std(unbiased=False).item())
        if sigma < 1e-12:
            g1 = 0.0
        else:
            centered = all_scores - mu
            m3 = float(torch.mean(centered ** 3).item())
            g1 = m3 / (sigma ** 3 + 1e-12)

        # Asymmetric quantile levels
        alpha_pos = alpha0 * (1.0 + skew_c * max(g1, 0.0))
        alpha_neg = alpha0 * (1.0 + skew_c * max(-g1, 0.0))
        # Keep within sensible bounds
        alpha_pos = float(min(max(alpha_pos, 1e-4), 0.49))
        alpha_neg = float(min(max(alpha_neg, 1e-4), 0.49))

        # Thresholds via quantiles
        t0 = float(torch.quantile(all_scores, q=alpha_neg).item())
        t1 = float(torch.quantile(all_scores, q=1.0 - alpha_pos).item())
        # Numerical safety
        t0 = max(0.0, min(1.0, t0))
        t1 = max(0.0, min(1.0, t1))

        print("=== Pseudo-labeling Strategy (Quantiles + Skewness) ===")
        print(f"  g1 (skewness): {g1:.4f}  |  mean={mu:.4f}, std={sigma:.4f}")
        print(f"  alpha0={alpha0:.4f}, c={skew_c:.4f}  ->  alpha_neg={alpha_neg:.4f}, alpha_pos={alpha_pos:.4f}")
        print(f"  thresholds: t0=Q_{alpha_neg:.4f}={t0:.4f},  t1=Q_{1-alpha_pos:.4f}={t1:.4f}")
        print(f"  Pseudo-negative: s <= {t0:.4f}")
        print(f"  Unlabeled: {t0:.4f} < s < {t1:.4f}")
        print(f"  Pseudo-positive: s >= {t1:.4f}")

        # Pass 2: assign labels using the computed thresholds
        total_edges = 0
        pseudo_positive = 0
        pseudo_negative = 0
        unlabeled_edges = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                mask_scores = self._score_base_batch(batch, model)

                # Create pseudo-labels: -1 for unlabeled, 0 for negative, 1 for positive
                pseudo_labels = torch.full_like(mask_scores, -1.0, dtype=torch.float)
                labeled_mask = torch.zeros_like(mask_scores, dtype=torch.bool)

                # Apply quantile thresholds
                positive_mask = mask_scores >= t1
                negative_mask = mask_scores <= t0

                pseudo_labels[positive_mask] = 1.0
                labeled_mask[positive_mask] = True
                pseudo_positive += int(positive_mask.sum().item())

                pseudo_labels[negative_mask] = 0.0
                labeled_mask[negative_mask] = True
                pseudo_negative += int(negative_mask.sum().item())

                total_edges += int(mask_scores.numel())
                unlabeled_edges += int((~labeled_mask).sum().item())

                new_data = Data(
                    x=batch.x.to(self.device),
                    edge_index=batch.edge_index.to(self.device),
                    edge_gt=pseudo_labels.unsqueeze(1).to(self.device),
                    labeled_mask=labeled_mask.to(self.device),
                    y=batch.y.to(self.device) if hasattr(batch, 'y') else None
                )

                pseudo_labeled_data_list.append(new_data)

        print("\nPseudo-labeling statistics:")
        print(f"  Total edges: {total_edges}")
        print(f"  Pseudo-positive (>= t1): {pseudo_positive} ({(pseudo_positive/max(total_edges,1))*100:.1f}%)")
        print(f"  Pseudo-negative (<= t0): {pseudo_negative} ({(pseudo_negative/max(total_edges,1))*100:.1f}%)")
        print(f"  Unlabeled (excluded): {unlabeled_edges} ({(unlabeled_edges/max(total_edges,1))*100:.1f}%)")
        print(f"  Labeled ratio: {((pseudo_positive + pseudo_negative)/max(total_edges,1))*100:.1f}%")

        return pseudo_labeled_data_list

    def _generate_pseudo_labeled_dataset_from_guided(self, train_loader):
        pseudo_labeled = []
        self.explainer_round1.eval()

        # Hyperparameters (same defaults as base pseudo-labeling)
        alpha0 = float(getattr(self, 'pseudo_alpha0', 0.10))
        skew_c = float(getattr(self, 'pseudo_skew_c', 0.50))

        # Pass 1: collect all guided scores to compute quantiles/skewness
        guided_scores = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                prob = self.explainer_round1(batch).detach().cpu().float().squeeze()
                guided_scores.append(prob)

        if len(guided_scores) == 0:
            print("[round 2] No guided scores; returning empty list.")
            return []

        guided_scores = torch.cat(guided_scores, dim=0)
        guided_scores = torch.clamp(guided_scores, 0.0, 1.0)

        mu = float(guided_scores.mean().item())
        sigma = float(guided_scores.std(unbiased=False).item())
        if sigma < 1e-12:
            g1 = 0.0
        else:
            centered = guided_scores - mu
            m3 = float(torch.mean(centered ** 3).item())
            g1 = m3 / (sigma ** 3 + 1e-12)

        alpha_pos = alpha0 * (1.0 + skew_c * max(g1, 0.0))
        alpha_neg = alpha0 * (1.0 + skew_c * max(-g1, 0.0))
        alpha_pos = float(min(max(alpha_pos, 1e-4), 0.49))
        alpha_neg = float(min(max(alpha_neg, 1e-4), 0.49))

        t0 = float(torch.quantile(guided_scores, q=alpha_neg).item())
        t1 = float(torch.quantile(guided_scores, q=1.0 - alpha_pos).item())
        t0 = max(0.0, min(1.0, t0))
        t1 = max(0.0, min(1.0, t1))

        print("=== [round 2] Pseudo-labeling (Quantiles + Skewness) ===")
        print(f"  g1 (skewness): {g1:.4f}  |  mean={mu:.4f}, std={sigma:.4f}")
        print(f"  alpha0={alpha0:.4f}, c={skew_c:.4f}  ->  alpha_neg={alpha_neg:.4f}, alpha_pos={alpha_pos:.4f}")
        print(f"  thresholds: t0=Q_{alpha_neg:.4f}={t0:.4f},  t1=Q_{1-alpha_pos:.4f}={t1:.4f}")

        total_edges = 0
        pos_edges = 0
        neg_edges = 0
        unlabeled = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                prob = self.explainer_round1(batch)  # [E]

                pseudo = torch.full_like(prob, -1.0, dtype=torch.float)
                labeled = torch.zeros_like(prob, dtype=torch.bool)

                pos_mask = prob >= t1
                neg_mask = prob <= t0
                pseudo[pos_mask] = 1.0
                pseudo[neg_mask] = 0.0
                labeled[pos_mask | neg_mask] = True

                total_edges += int(prob.numel())
                pos_edges += int(pos_mask.sum().item())
                neg_edges += int(neg_mask.sum().item())
                unlabeled += int((~(pos_mask | neg_mask)).sum().item())

                new_data = Data(
                    x=batch.x.to(self.device),
                    edge_index=batch.edge_index.to(self.device),
                    edge_gt=pseudo.unsqueeze(1).to(self.device),
                    labeled_mask=labeled.to(self.device),
                    y=batch.y.to(self.device) if hasattr(batch, 'y') else None
                )
                pseudo_labeled.append(new_data)

        print(
            f"[round 2] Pseudo-label stats: total={total_edges}, "
            f"pos={pos_edges} ({(pos_edges/max(total_edges,1))*100:.1f}%), "
            f"neg={neg_edges} ({(neg_edges/max(total_edges,1))*100:.1f}%), "
            f"unlabeled={unlabeled} ({(unlabeled/max(total_edges,1))*100:.1f}%)"
        )
        return pseudo_labeled
