import argparse
import torch

parser = argparse.ArgumentParser()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

### GNN hyperparameters ###
parser.add_argument('--dataset', type=str, default='MUTAG', choices=['MUTAG', 'BA3', 'FC', 'MNIST', 'AC'])
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN'])
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=4, help='Number of hidden layers.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout ratio.')
parser.add_argument('--pool_type', type=str, default='sum', choices=['mean', 'sum', 'max'], help='Pooling type.')
parser.add_argument('--use_jk', default=True, help='Use Jumping Knowledge.')
parser.add_argument('--only_pos', dest='only_pos', action='store_true',
                    help='Keep only positive-label graphs for datasets that support it.')
parser.add_argument('--no_only_pos', dest='only_pos', action='store_false',
                    help='Use all labels even if only_pos is enabled in configs.')
parser.set_defaults(only_pos=None)

### Explainer hyperparameters ###
parser.add_argument('--explainer_name', type=str, default='pgexplainer', help='explainer to be used.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--size_reg', type=float, default=1.0, help='Size regularization coefficient.')
parser.add_argument('--ent_reg', type=float, default=1.0, help='Entropy regularization coefficient.')
parser.add_argument('--temp_0', type=float, default=1.0, help='Temperature 0')
parser.add_argument('--temp_1', type=float, default=1.0, help='Temperature 1')

parser.add_argument('--vae_lr', type=float, default=0.001, help='Vae learning rate')
parser.add_argument('--round1_lr', type=float, default = 0.0001, help='Phase 2 lr')
parser.add_argument('--round1_epochs', type=int, default = 100, help='Phase 2 epochs')
parser.add_argument('--round', type=int, default=5, help='Number of guided explainer rounds')
parser.add_argument('--guided_patience', type=int, default=30, help='Early stopping patience per guided round')
parser.add_argument('--base_explainer', type=str, default='pgexplainer',
                    choices=['pgexplainer', 'proxyexplainer', 'mixupexplainer', 'gsat', 'confexplainer', 'goat', 'eigsearch'],
                    help='Base explainer type for beep-style methods')
parser.add_argument('--baseline_manifest', type=str, default='configs/beep_baselines.yaml',
                    help='Override manifest for resolved base explainers')
parser.add_argument('--baseline_source', type=str, default='latest_tuning',
                    choices=['latest_tuning'], help='Source of resolved base explainer settings')
parser.add_argument('--base_checkpoint_path', type=str, default=None,
                    help='Explicit pre-trained base explainer checkpoint path for BEEP')

# Optional second guided stage (Phase 3) defaults
parser.add_argument('--enable_second_guided', action='store_true', help='Enable second-stage guided explainer (Phase 3)')
parser.add_argument('--second_phase_epochs', type=int, default=100, help='Phase 3 epochs (defaults to round1_epochs)')
parser.add_argument('--second_phase_lr', type=float, default=None, help='Phase 3 learning rate (defaults to round1_lr)')
parser.add_argument('--second_std_multiplier', type=float, default=None, help='Std multiplier for Phase 3 (defaults to std_multiplier)')

# Self-guidance hyperparameters (defaults align with YAML if present)
parser.add_argument('--self_guidance_high', type=float, default=None, help='High threshold for self-guided relabeling')
parser.add_argument('--self_guidance_low', type=float, default=None, help='Low threshold for self-guided relabeling')
parser.add_argument('--self_guidance_rounds', type=int, default=None, help='Number of self-guided rounds')
parser.add_argument('--self_guidance_reinit', action='store_true', help='Reinitialize guided model each round')

# Contrastive separation learning hyperparameters
parser.add_argument('--use_contrastive_separation', action='store_true', help='Use contrastive separation learning task')
parser.add_argument('--contrastive_epochs', type=int, default=50, help='Contrastive learning epochs')
parser.add_argument('--contrastive_lr', type=float, default=0.001, help='Contrastive learning rate')
parser.add_argument('--separation_weight', type=float, default=1.0, help='Weight for inter-cluster separation loss')
parser.add_argument('--compactness_weight', type=float, default=0.1, help='Weight for intra-cluster compactness loss')

parser.add_argument('--tau', type=float, default=0.8, help='expl_threshold')
parser.add_argument('--std_multiplier', type=float, default=1.0, help='standard deviation multiplier for tau margin')
# Quantile pseudo-labeling (skewness-adjusted) defaults
parser.add_argument('--alpha0', type=float, default=0.10, help='Base quantile cutoff (0, 0.5) for pseudo-labeling')
parser.add_argument('--c', type=float, default=0.50, help='Skewness adjustment factor (>0) for asymmetric quantiles')
parser.add_argument('--smoothing_factor', type=float, default=0.1, help='smoothing factor for loss')
parser.add_argument('--vae_thre', type=float, default=0.7, help='vae_thre')
parser.add_argument('--flip_p', default=0.2, type=float, help='dropedge rate')
parser.add_argument('--pseudo_noise_ratio', type=float, default=0.0, help='Label noise ratio for pseudo-labels')
parser.add_argument('--threshold_strategy', type=str, default=None,
                    help='Pseudo-label thresholding strategy (skewness_adjusted, symmetric_quantile, mean)')
parser.add_argument('--w', default=10.0, type=float, help='weight for loss')

# Training-free base explainer controls (GOAt / EiG-Search)
parser.add_argument('--is_undirected', type=parse_bool, default=True,
                    help='Treat reverse directed edges as one undirected edge for GOAt/EiG-Search')
parser.add_argument('--goat_score_transform', type=str, default='default',
                    choices=['default', 'raw', 'abs', 'positive', 'negative'],
                    help='GOAt score post-processing before normalization')
parser.add_argument('--goat_undirected_aggr', type=str, default='mean',
                    choices=['mean', 'max', 'sum'],
                    help='GOAt reverse-edge aggregation when is_undirected is true')
parser.add_argument('--eig_epsilon', type=float, default=0.0,
                    help='EiG-Search edge weight used to ablate selected edges')
parser.add_argument('--eig_sparsity', type=float, default=0.7,
                    help='EiG-Search fraction of edges to drop before/after search')
parser.add_argument('--eig_max_edges', type=int, default=200,
                    help='EiG-Search exact scoring edge limit; larger graphs use gradient saliency')
parser.add_argument('--eig_search_max_edges', type=int, default=200,
                    help='EiG-Search linear search edge limit')
parser.add_argument('--eig_enable_search', type=parse_bool, default=True,
                    help='Run EiG-Search linear prefix search after edge ranking')
parser.add_argument('--base_score_cache_dir', type=str, default='edge_score_cache',
                    help='Directory for cached training-free base edge scores')
parser.add_argument('--disable_base_score_cache', action='store_true', default=False,
                    help='Disable cached training-free base edge scores')

### etc ###
parser.add_argument('--ckpt_path', type=str, default='test/ckpts/', help='Location for saving checkpoints')
parser.add_argument('--gpu', type=int, default=7, help='GPU device id to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
