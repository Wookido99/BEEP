Before running any experiments, make sure to install the required packages:
```
pip install -r requirements.txt
```

## External files

The following directories are not included in this GitHub repository because
they contain datasets, cached edge scores, pretrained parameters, or other
large experiment artifacts:

- `best_base_param/`
- `data/`
- `edge_score_cache_eigsearch/`
- `edge_score_cache_goat/`
- `edge_score_cache_goat_best/`
- `param/`

These files will be provided separately as a downloadable archive. After
downloading the archive, extract it into the project root so that the directory
structure matches the paths above:

```
BEEP/
├── data/
├── param/
├── best_base_param/
├── edge_score_cache_eigsearch/
├── edge_score_cache_goat/
└── edge_score_cache_goat_best/
```

Download link: `TODO: add external file archive link`

- dataset: name of the dataset (MUTAG, BA3, FC, MNIST)
- explainer_name: name of the explainer module (beep)
- base_explainer: name of the base explainer used to generate guidance scores (pgexplainer, proxyexplainer, mixupexplainer, gsat, confexplainer, goat, eigsearch)

If you want to run the experiment with a specific random seed:
```
python main.py --dataset MUTAG --explainer_name beep --base_explainer pgexplainer --seed 42
```
