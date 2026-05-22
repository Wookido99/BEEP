import os
import os.path as osp
import argparse
import yaml
from typing import Dict, List, Optional, Tuple


_MPL_CACHE_DIR = osp.abspath(osp.join('plot', '.matplotlib_cache_eval'))
os.environ.setdefault('MPLCONFIGDIR', _MPL_CACHE_DIR)
os.environ.setdefault('MPLBACKEND', 'Agg')
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from evaluation.evaluate_f1 import (
    compute_polarity,
    compute_bimodality_coefficient,
)
from evaluation.evaluate_auc import evaluate_auc, evaluate_auc_class

from utils import get_dataset
from gnn.GCN import GCN
from gnn.GIN import GIN
from classifier import GuidedExplainer, GuidedExplainerGIN
from explainer.explainer_utils import create_edge_embeds, sample_graph
from explainer.baseline_resolver import build_base_explainer_adapter, score_base_explainer_batch


CONFIG_ALIASES = {
    'beep_nongib': 'beep',
}

METRIC_NAMES = [
    'base_f1@0.5',
    'guided_p2_f1@0.5',
    'guided_p3_f1@0.5',
    'orig_acc',
    'base_acc',
    'guided_p2_acc',
    'guided_p3_acc',
    'base_auc',
    'guided_p2_auc',
    'guided_p3_auc',
    'base_vs_orig',
    'guided_p2_vs_orig',
    'guided_p3_vs_orig',
    'guided_p2_vs_base',
    'guided_p3_vs_base',
    'p1_polarity',
    'p1_bimodality',
    'p2_polarity',
    'p2_bimodality',
    'p3_polarity',
    'p3_bimodality',
]


def load_yaml_config(dataset: str) -> Dict:
    path = osp.join('configs', f'{dataset}.yaml')
    if not osp.exists(path):
        raise FileNotFoundError(f'Missing YAML: {path}')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_gnn(dataset: str, cfg: Dict, device: torch.device, explainer_name: str):
    in_dim = int(cfg.get('in_dim'))
    num_cls = int(cfg.get('num_cls'))
    pool_type = cfg.get('pool_type', 'sum')
    use_jk = bool(cfg.get('use_jk', False))

    config_name = CONFIG_ALIASES.get(explainer_name, explainer_name)
    expl_cfg = cfg.get(config_name, cfg.get('beep', {}))
    hidden = int(expl_cfg.get('hidden', 64))
    nlayers = int(expl_cfg.get('nlayers', 4))
    dropout = float(expl_cfg.get('dropout', 0.5))
    model_name = str(expl_cfg.get('model', 'GCN')).upper()

    if model_name == 'GCN':
        model = GCN(in_dim, num_cls, nlayers, hidden, dropout, pool_type, use_jk)
    elif model_name == 'GIN':
        model = GIN(in_dim, num_cls, nlayers, hidden, dropout, pool_type)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    ckpt = osp.join('param', model_name, f'{dataset}_{model_name}_best_val.pth')
    if not osp.exists(ckpt):
        raise FileNotFoundError(f'Missing GNN checkpoint: {ckpt}')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, hidden, in_dim, num_cls


def build_explainers(dataset: str, hidden: int, in_dim: int, device: torch.device):
    guided_cls = GuidedExplainerGIN if dataset == 'FC' else GuidedExplainer
    guided_p2 = guided_cls(in_dim, hidden_dim=hidden, out_channels=2).to(device)
    guided_p3 = guided_cls(in_dim, hidden_dim=hidden, out_channels=2).to(device)
    return guided_p2, guided_p3


def test_loader(dataset: str):
    root = osp.join('data', dataset)
    _, _, test_ds, _ = get_dataset(root, dataset)
    return DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)


def compute_f1_05(test_loader, model, base_exp, guided_exp, device) -> Tuple[float, float]:
    base_scores: List[float] = []
    guided_scores: List[float] = []
    gts: List[int] = []

    model.eval(); base_exp.eval(); guided_exp.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            p1 = score_base_explainer_batch(batch, model, base_exp, device)
            # guided
            p2 = guided_exp(batch).squeeze()
            if p2.dim() > 1 and p2.size(-1) == 2:
                p2 = torch.softmax(p2, dim=-1)[:, 1]
            # gt
            gt = batch.edge_gt.squeeze()
            base_scores.extend(p1.detach().cpu().numpy().tolist())
            guided_scores.extend(p2.detach().cpu().numpy().tolist())
            gts.extend(gt.detach().cpu().numpy().tolist())

    gts_arr = np.asarray(gts, dtype=int)
    base_arr = (np.asarray(base_scores, dtype=float) >= 0.5).astype(int)
    guided_arr = (np.asarray(guided_scores, dtype=float) >= 0.5).astype(int)
    base_f1 = f1_score(gts_arr, base_arr, zero_division=0)
    guided_f1 = f1_score(gts_arr, guided_arr, zero_division=0)
    return float(base_f1), float(guided_f1)


def compute_acc_with_masks(test_loader, model, base_thr, guided_thr, base_exp, guided_exp, device, dataset: str):
    model.eval(); base_exp.eval(); guided_exp.eval()
    orig_correct = base_correct = guided_correct = total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, orig_logits, node_embeds = model(batch)
            orig_cls = torch.argmax(orig_logits, dim=1)
            true_y = batch.y

            # base mask
            p1 = score_base_explainer_batch(batch, model, base_exp, device)
            w1 = (p1 >= base_thr).float()
            # w1 = (p1).float()

            # guided mask
            p2 = guided_exp(batch).squeeze()
            if p2.dim() > 1 and p2.size(-1) == 2:
                p2 = torch.softmax(p2, dim=-1)[:, 1]
            w2 = (p2 >= guided_thr).float()
            # w2 = (p2).float()
                
            _, base_logits, _ = model(batch, edge_weights=w1)
            _, guided_logits, _ = model(batch, edge_weights=w2)
            base_cls = torch.argmax(base_logits, dim=1)
            guided_cls = torch.argmax(guided_logits, dim=1)

            include = (true_y == 1) if dataset == 'FC' else torch.ones_like(true_y, dtype=torch.bool)
            n = int(include.sum().item())
            total += n
            if n > 0:
                orig_correct += int((orig_cls[include] == true_y[include]).sum().item())
                base_correct += int((base_cls[include] == true_y[include]).sum().item())
                guided_correct += int((guided_cls[include] == true_y[include]).sum().item())
            if w1 is None:
                base_correct +=  0
            if w2 is None:
                guided_correct += 0
                
    denom = max(total, 1)
    return float(orig_correct/denom), float(base_correct/denom), float(guided_correct/denom)


def collect_phase1_scores(test_loader, model, base_exp, device):
    """Collect Phase 1 mask scores and ground-truth across the test set."""
    p1_scores = []
    gts = []
    model.eval(); base_exp.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            p1 = score_base_explainer_batch(batch, model, base_exp, device)
            gt = batch.edge_gt.squeeze()
            p1_scores.extend(p1.detach().cpu().numpy().tolist())
            gts.extend(gt.detach().cpu().numpy().tolist())
    return np.asarray(p1_scores, dtype=float), np.asarray(gts, dtype=int)


def collect_guided_scores(test_loader, model, guided_exp, device):
    """Collect Guided (Phase 2 or 3) mask scores across the test set."""
    scores = []
    model.eval(); guided_exp.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            p = guided_exp(batch).squeeze()
            if p.dim() > 1 and p.size(-1) == 2:
                p = torch.softmax(p, dim=-1)[:, 1]
            scores.extend(p.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def seed_dirs(root: str) -> List[str]:
    if not osp.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if d.isdigit() and osp.isdir(osp.join(root, d))], key=lambda x: int(x))


def resolve_eval_root(dataset: str, explainer_name: str, base_explainer: Optional[str]) -> str:
    if explainer_name in {'beep', 'beep_nongib'} and base_explainer:
        candidate = osp.join('param', dataset, explainer_name, base_explainer)
        if osp.isdir(candidate):
            return candidate
    return osp.join('param', dataset, explainer_name)


def main():
    ap = argparse.ArgumentParser(description='Evaluate saved explainers: base + guided P2/P3 @ thr=0.5')
    ap.add_argument('--dataset', required=True, choices=['MUTAG', 'BA3', 'FC', 'MNIST'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--explainer_name', default='beep')
    ap.add_argument('--base_explainer', default=None)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)

    cfg = load_yaml_config(args.dataset)
    explainer_name = args.explainer_name
    model, hidden, in_dim, _ = build_gnn(args.dataset, cfg, device, explainer_name)
    loader = test_loader(args.dataset)

    root = resolve_eval_root(args.dataset, explainer_name, args.base_explainer)
    seeds = [str(args.seed)] if args.seed is not None else seed_dirs(root)
    if not seeds:
        print(f'No seeds under {root}')
        return

    explainer_label = explainer_name if not args.base_explainer else f'{explainer_name}/{args.base_explainer}'
    print(f'Dataset={args.dataset} | Explainer={explainer_label} | Seeds={" ".join(seeds)} | Device={device}')
    print('seed, base_f1@0.5, guided_p2_f1@0.5, guided_p3_f1@0.5, '
          'orig_acc, base_acc, guided_p2_acc, guided_p3_acc, '
          'base_auc, guided_p2_auc, guided_p3_auc, '
          'base_vs_orig, guided_p2_vs_orig, guided_p3_vs_orig, guided_p2_vs_base, guided_p3_vs_base, '
          'p1_polarity, p1_bimodality, p2_polarity, p2_bimodality, p3_polarity, p3_bimodality')

    rows = []
    for s in seeds:
        prefix = f'{args.dataset}_{explainer_name}'
        base_ckpt = osp.join(root, s, f'{prefix}_phase1_explainer.pth')
        p2_ckpt   = osp.join(root, s, f'{prefix}_best_model.pth')
        p3_ckpt   = osp.join(root, s, f'{prefix}_best_model_second.pth')

        missing = [p for p in [base_ckpt, p2_ckpt, p3_ckpt] if not osp.exists(p)]
        if missing:
            print(f'[seed {s}] Missing checkpoints: {missing} → skip')
            continue

        p2_state = torch.load(p2_ckpt, map_location=device)
        p3_state = torch.load(p3_ckpt, map_location=device)
        base_state = torch.load(base_ckpt, map_location=device)
        resolved_base = args.base_explainer or detect_explainer_family(base_state)
        if resolved_base == 'pg':
            resolved_base = 'pgexplainer'
        base_exp, _ = build_base_explainer_adapter(
            dataset=args.dataset,
            base_explainer=resolved_base,
            checkpoint_path=base_ckpt,
            hidden_dim=hidden,
            in_dim=in_dim,
            num_cls=int(cfg.get('num_cls')),
            use_jk=bool(cfg.get('use_jk', False)),
            device=device,
        )
        guided_p2, guided_p3 = build_explainers(args.dataset, hidden, in_dim, device)
        guided_p2.load_state_dict(p2_state)
        guided_p3.load_state_dict(p3_state)
        base_exp.eval(); guided_p2.eval(); guided_p3.eval()

        base_f1, p2_f1 = compute_f1_05(loader, model, base_exp, guided_p2, device)
        _,      p3_f1 = compute_f1_05(loader, model, base_exp, guided_p3, device)

        orig_acc, base_acc, p2_acc = compute_acc_with_masks(
            loader, model, 0.5, 0.5, base_exp, guided_p2, device, args.dataset
        )
        _, _, p3_acc = compute_acc_with_masks(
            loader, model, 0.5, 0.5, base_exp, guided_p3, device, args.dataset
        )

        base_auc = evaluate_auc(loader, base_exp, model, device, training=False)
        p2_auc = evaluate_auc_class(loader, guided_p2, device=device)
        p3_auc = evaluate_auc_class(loader, guided_p3, device=device)

        bvso = base_acc - orig_acc
        p2_vo = p2_acc - orig_acc
        p3_vo = p3_acc - orig_acc
        p2_vb = p2_acc - base_acc
        p3_vb = p3_acc - base_acc

        # Per-phase distribution metrics
        p1_scores, _gts = collect_phase1_scores(loader, model, base_exp, device)
        p2_scores = collect_guided_scores(loader, model, guided_p2, device)
        p3_scores = collect_guided_scores(loader, model, guided_p3, device)

        p1_pol = compute_polarity(p1_scores)
        p1_bim = compute_bimodality_coefficient(p1_scores)
        p2_pol = compute_polarity(p2_scores)
        p2_bim = compute_bimodality_coefficient(p2_scores)
        p3_pol = compute_polarity(p3_scores)
        p3_bim = compute_bimodality_coefficient(p3_scores)

        rows.append((int(s), base_f1, p2_f1, p3_f1, orig_acc, base_acc, p2_acc, p3_acc, base_auc, p2_auc, p3_auc,
                     bvso, p2_vo, p3_vo, p2_vb, p3_vb,
                     p1_pol, p1_bim, p2_pol, p2_bim, p3_pol, p3_bim))
        print(f'{s}, {base_f1:.4f}, {p2_f1:.4f}, {p3_f1:.4f}, '
              f'{orig_acc:.4f}, {base_acc:.4f}, {p2_acc:.4f}, {p3_acc:.4f}, '
              f'{base_auc:.4f}, {p2_auc:.4f}, {p3_auc:.4f}, '
              f'{bvso:+.4f}, {p2_vo:+.4f}, {p3_vo:+.4f}, {p2_vb:+.4f}, {p3_vb:+.4f}, '
              f'{p1_pol:.4f}, {p1_bim:.4f}, {p2_pol:.4f}, {p2_bim:.4f}, {p3_pol:.4f}, {p3_bim:.4f}')

    if rows:
        arr = np.array(rows, dtype=float)
        # 0 seed | 1 base_f1 | 2 p2_f1 | 3 p3_f1 | 4 orig | 5 base | 6 p2 | 7 p3 | 8 base_auc | 9 p2_auc | 10 p3_auc | 11 bvso | 12 p2_vo | 13 p3_vo | 14 p2_vb | 15 p3_vb | 16 p1_pol | 17 p1_bim | 18 p2_pol | 19 p2_bim | 20 p3_pol | 21 p3_bim
        values = arr[:, 1:]
        means = np.mean(values, axis=0)
        stds = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1])
        avg_items = list(zip(METRIC_NAMES, means))
        summary_items = list(zip(METRIC_NAMES, means, stds))
        avg_metrics = dict(avg_items)
        std_metrics = dict(zip(METRIC_NAMES, stds))
        print('\nAverages across seeds:')
        print(f'  base_f1@0.5:    {avg_metrics["base_f1@0.5"]:.6f}')
        print(f'  guided_p2_f1@0.5:{avg_metrics["guided_p2_f1@0.5"]:.6f}')
        print(f'  guided_p3_f1@0.5:{avg_metrics["guided_p3_f1@0.5"]:.6f}')
        print(f'  orig_acc:       {avg_metrics["orig_acc"]:.6f}')
        print(f'  base_acc:       {avg_metrics["base_acc"]:.6f}')
        print(f'  guided_p2_acc:  {avg_metrics["guided_p2_acc"]:.6f}')
        print(f'  guided_p3_acc:  {avg_metrics["guided_p3_acc"]:.6f}')
        print(f'  base_auc:       {avg_metrics["base_auc"]:.6f}')
        print(f'  guided_p2_auc:  {avg_metrics["guided_p2_auc"]:.6f}')
        print(f'  guided_p3_auc:  {avg_metrics["guided_p3_auc"]:.6f}')
        print(f'  base_vs_orig:   {avg_metrics["base_vs_orig"]:+.6f}')
        print(f'  guided_p2_vs_orig:{avg_metrics["guided_p2_vs_orig"]:+.6f}')
        print(f'  guided_p3_vs_orig:{avg_metrics["guided_p3_vs_orig"]:+.6f}')
        print(f'  guided_p2_vs_base:{avg_metrics["guided_p2_vs_base"]:+.6f}')
        print(f'  guided_p3_vs_base:{avg_metrics["guided_p3_vs_base"]:+.6f}')
        print(f'  p1_polarity:    {avg_metrics["p1_polarity"]:.6f}')
        print(f'  p1_bimodality:  {avg_metrics["p1_bimodality"]:.6f}')
        print(f'  p2_polarity:    {avg_metrics["p2_polarity"]:.6f}')
        print(f'  p2_bimodality:  {avg_metrics["p2_bimodality"]:.6f}')
        print(f'  p3_polarity:    {avg_metrics["p3_polarity"]:.6f}')
        print(f'  p3_bimodality:  {avg_metrics["p3_bimodality"]:.6f}')

        print('\nMean +/- std across seeds:')
        for metric, mean, std in summary_items:
            print(f'  {metric}: {mean:.6f} +/- {std:.6f}')

        csv_dir = osp.join('results', args.dataset)
        os.makedirs(csv_dir, exist_ok=True)
        csv_stem = explainer_name if not args.base_explainer else f'{explainer_name}_{args.base_explainer}'

        per_seed_name = f'{args.dataset}_{csv_stem}_per_seed.csv'
        per_seed_path = osp.join(csv_dir, per_seed_name)
        with open(per_seed_path, 'w') as f:
            headers = ['seed'] + METRIC_NAMES
            f.write(','.join(headers) + '\n')
            for row in rows:
                f.write(','.join([str(int(row[0]))] + [f'{value:.6f}' for value in row[1:]]) + '\n')
        print(f'Wrote per-seed CSV to {per_seed_path}')

        csv_name = f'{args.dataset}_{csv_stem}_averages.csv'
        csv_path = osp.join(csv_dir, csv_name)
        with open(csv_path, 'w') as f:
            headers = ['metric', 'value']
            f.write(','.join(headers) + '\n')
            for metric, value in avg_items:
                f.write(f'{metric},{value:.6f}\n')
        print(f'Wrote averages CSV to {csv_path}')

        summary_name = f'{args.dataset}_{csv_stem}_mean_std.csv'
        summary_path = osp.join(csv_dir, summary_name)
        with open(summary_path, 'w') as f:
            headers = ['metric', 'mean', 'std', 'n']
            f.write(','.join(headers) + '\n')
            for metric, mean, std in summary_items:
                f.write(f'{metric},{mean:.6f},{std:.6f},{values.shape[0]}\n')
        print(f'Wrote mean/std CSV to {summary_path}')


if __name__ == '__main__':
    main()
