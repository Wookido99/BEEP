Before running any experiments, make sure to install the required packages:
```
pip install -r requirements.txt
```

## Data and pretrained artifacts

The dataset files are not included in this GitHub repository. Please download
the datasets separately and place them under `data/` before running the
experiments.

Base explainers have already been pretrained, and their artifacts are included
in this repository either as parameters or cached scores:

- `best_base_param/`: pretrained base explainer parameters
- `edge_score_cache_eigsearch/`: cached `eigsearch` edge scores
- `edge_score_cache_goat/`: cached `goat` edge scores
- `edge_score_cache_goat_best/`: cached `goat` edge scores used for the best
  configuration

For smoother reproduction, pretrained BEEP parameters are also included:

- `param/`: pretrained model and BEEP parameters

After downloading the datasets, the expected project structure is:

```
BEEP/
├── data/
├── param/
├── best_base_param/
├── edge_score_cache_eigsearch/
├── edge_score_cache_goat/
└── edge_score_cache_goat_best/
```

- dataset: name of the dataset (MUTAG, BA3, FC, MNIST)
- explainer_name: name of the explainer module (beep)
- base_explainer: name of the base explainer used to generate guidance scores (pgexplainer, proxyexplainer, mixupexplainer, gsat, confexplainer, goat, eigsearch)

If you want to run the experiment with a specific random seed:
```
python main.py --dataset MUTAG --explainer_name beep --base_explainer pgexplainer --seed 42
```
