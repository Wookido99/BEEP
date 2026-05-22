import os
import argparse
import torch
import numpy as np
import random
import yaml

from configs.config import args as config_defaults, parse_bool


EXPLAINER_REGISTRY = {
    "beep": ("explainer.beep", "beep"),
}

def load_explainer_class(explainer_name):
    if explainer_name not in EXPLAINER_REGISTRY:
        supported = ", ".join(sorted(EXPLAINER_REGISTRY))
        raise ValueError(f"Unsupported explainer_name={explainer_name!r}. Supported: {supported}")

    module_name, class_name = EXPLAINER_REGISTRY[explainer_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")

    ### GNN hyperparameters ###
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'BA3', 'FC', 'MNIST', 'AC'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, choices=['GCN', 'GIN'], help='Model name')
    parser.add_argument('--hidden', type=int, help='Number of hidden units')
    parser.add_argument('--nlayers', type=int, help='Number of hidden layers')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--dropout', type=float, help='Dropout ratio')
    parser.add_argument('--pool_type', type=str, choices=['mean', 'sum', 'max'], help='Pooling type')
    parser.add_argument('--norm_type', type=str, choices=['batch', 'layer', 'instance'], help='Normalization type')
    parser.add_argument('--only_pos', dest='only_pos', action='store_true',
                        help='Use only positive-labeled graphs for applicable datasets')
    parser.add_argument('--no_only_pos', dest='only_pos', action='store_false',
                        help='Use all labels even if configs enable only_pos by default')
    parser.set_defaults(only_pos=None)

    ### Explainer hyperparameters ###
    parser.add_argument('--explainer_name', type=str, choices=['beep'], help='Explainer to be used')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    
    # BEEP-specific arguments
    parser.add_argument('--guided_epochs', type=int, help='Number of guided explainer epochs')
    parser.add_argument('--guided_lr', type=float, help='Guided explainer learning rate')
    parser.add_argument('--round', type=int, help='Number of guided explainer rounds')
    parser.add_argument('--guided_patience', type=int, help='Early stopping patience per guided round')
    parser.add_argument('--base_explainer', type=str,
                        choices=['pgexplainer', 'proxyexplainer', 'mixupexplainer', 'gsat', 'confexplainer', 'goat', 'eigsearch'],
                        help='Base explainer type')
    parser.add_argument('--baseline_manifest', type=str, help='Baseline override manifest path')
    parser.add_argument('--baseline_source', type=str, choices=['latest_tuning'],
                        help='Source of baseline hyperparameters/checkpoints')
    parser.add_argument('--base_checkpoint_path', type=str,
                        help='Explicit pre-trained base explainer checkpoint path for BEEP')
    parser.add_argument('--gen_epochs', type=int, help='Generator/VAE training epochs')
    parser.add_argument('--gen_lr', type=float, help='Generator/VAE learning rate')
    parser.add_argument('--gen_hidden', type=int, help='Generator hidden dimension')
    parser.add_argument('--gen_latent', type=int, help='Generator latent dimension')
    parser.add_argument('--guided_gnn', type=str, help='Guided GNN type')
    
    parser.add_argument('--tau', type=float, help='expl_threshold')
    parser.add_argument('--vae_thre', type=float, help='vae_thre')
    parser.add_argument('--flip_p', type=float, help='dropedge rate')

    parser.add_argument('--std_multiplier', type=float, default=1.0, help='standard deviation multiplier for tau margin')
    parser.add_argument('--smoothing_factor', type=float, default=0.1, help='smoothing factor for loss')
    parser.add_argument('--w', type=float, help='weight for loss')

    # Quantile pseudo-labeling (skewness-adjusted) hyperparameters
    parser.add_argument('--alpha0', type=float, help='Base quantile cutoff (0, 0.5) for pseudo-labeling')
    parser.add_argument('--c', type=float, help='Skewness adjustment factor (>0) for asymmetric quantiles')
    parser.add_argument('--pseudo_noise_ratio', type=float, help='Label noise ratio for pseudo-labels')
    parser.add_argument('--threshold_strategy', type=str,
                        help='Pseudo-label thresholding strategy (skewness_adjusted, symmetric_quantile, mean)')

    # Training-free base explainer controls (GOAt / EiG-Search)
    parser.add_argument('--is_undirected', type=parse_bool,
                        help='Treat reverse directed edges as one undirected edge for GOAt/EiG-Search')
    parser.add_argument('--goat_score_transform', type=str,
                        choices=['default', 'raw', 'abs', 'positive', 'negative'],
                        help='GOAt score post-processing before normalization')
    parser.add_argument('--goat_undirected_aggr', type=str,
                        choices=['mean', 'max', 'sum'],
                        help='GOAt reverse-edge aggregation when is_undirected is true')
    parser.add_argument('--eig_epsilon', type=float,
                        help='EiG-Search edge weight used to ablate selected edges')
    parser.add_argument('--eig_sparsity', type=float,
                        help='EiG-Search fraction of edges to drop before/after search')
    parser.add_argument('--eig_max_edges', type=int,
                        help='EiG-Search exact scoring edge limit; larger graphs use gradient saliency')
    parser.add_argument('--eig_search_max_edges', type=int,
                        help='EiG-Search linear search edge limit')
    parser.add_argument('--eig_enable_search', type=parse_bool,
                        help='Run EiG-Search linear prefix search after edge ranking')
    parser.add_argument('--base_score_cache_dir', type=str,
                        help='Directory for cached training-free base edge scores')
    parser.add_argument('--disable_base_score_cache', action='store_true', default=None,
                        help='Disable cached training-free base edge scores')

    parser.add_argument('--round1_epochs', type=int, help='Phase 2 training epochs')
    parser.add_argument('--round1_lr', type=float, help='Phase 2 learning rate')
    
    # Optional second guided stage (Phase 3) controls
    parser.add_argument('--enable_second_guided', action='store_true', default=None, help='Enable second-stage guided explainer (Phase 3)')
    parser.add_argument('--second_phase_epochs', type=int, help='Phase 3 training epochs (defaults to round1_epochs)')
    parser.add_argument('--second_phase_lr', type=float, help='Phase 3 learning rate (defaults to round1_lr)')
    parser.add_argument('--second_std_multiplier', type=float, help='Std multiplier for Phase 3 (defaults to std_multiplier)')
    
    # Self-guidance hyperparameters
    parser.add_argument('--self_guidance_high', type=float, help='High threshold for self-guided relabeling')
    parser.add_argument('--self_guidance_low', type=float, help='Low threshold for self-guided relabeling')
    parser.add_argument('--self_guidance_rounds', type=int, help='Number of self-guided rounds')
    parser.add_argument('--self_guidance_reinit', action='store_true', default=None, help='Reinitialize guided model each round')

    # Contrastive separation learning specific arguments
    parser.add_argument('--use_contrastive_separation', action='store_true', default=None, help='Use contrastive separation learning task')
    parser.add_argument('--contrastive_epochs', type=int, default=50, help='Contrastive learning epochs')
    parser.add_argument('--contrastive_lr', type=float, default=0.001, help='Contrastive learning rate')
    parser.add_argument('--separation_weight', type=float, default=1.0, help='Weight for inter-cluster separation loss')
    parser.add_argument('--compactness_weight', type=float, default=0.1, help='Weight for intra-cluster compactness loss')


    ### etc ###
    parser.add_argument('--ckpt_path', type=str, help='Location for saving checkpoints')
    parser.add_argument('--gpu', type=int, help='GPU device id to use')
    parser.add_argument('--seed', type=int, help='Random seed')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fill_missing_args(args, yaml_config, config_defaults):
    print("=== Argument Priority Check ===")
    terminal_args = {k for k, v in vars(args).items() if v is not None}
    print(f"Terminal args: {sorted(terminal_args)}")
    print(f"YAML config keys: {list(yaml_config.keys())}")
    print(f"Config defaults keys: {list(vars(config_defaults).keys())}")

    base_hparams = yaml_config.get("base_hparams", {})
    if base_hparams is None:
        base_hparams = {}
    
    # Apply YAML config for missing args
    yaml_applied = []
    for key, value in yaml_config.items():
        if key == "base_hparams":
            continue
        if getattr(args, key, None) is None:
            setattr(args, key, value)
            yaml_applied.append(key)
    print(f"Applied from YAML: {yaml_applied}")

    # Apply base-explainer-specific first-stage hparams after base_explainer is resolved.
    base_applied = []
    if args.explainer_name == "beep" and isinstance(base_hparams, dict):
        selected_base = getattr(args, "base_explainer", None)
        selected_hparams = base_hparams.get(selected_base, {})
        if isinstance(selected_hparams, dict):
            for key, value in selected_hparams.items():
                if key not in terminal_args:
                    setattr(args, key, value)
                    base_applied.append(key)
    print(f"Applied from base_hparams: {base_applied}")

    # Apply config defaults for still missing args
    config_applied = []
    for key, value in vars(config_defaults).items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
            config_applied.append(key)
    print(f"Applied from config.py: {config_applied}")

    print("=" * 40)


if __name__ == "__main__":
    args = parse_args()

    yaml_config = {}
    if args.dataset: 
        config_path = os.path.join("configs", f"{args.dataset}.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                yaml_config = yaml_data.get(args.explainer_name, {})

    fill_missing_args(args, yaml_config, config_defaults)

    set_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    explainer_cls = load_explainer_class(args.explainer_name)
    explainer = explainer_cls(args, device=device)
    explainer.train_test(args)
