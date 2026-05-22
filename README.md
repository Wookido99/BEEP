Before running any experiments, make sure to install the required packages:
```
pip install -r requirements.txt
```

- DATASET: name of the dataset (MUTAG, BA3, FC, MNIST)
- EXPLAINER: name of the explainer module (beep)

If you want to run the experiment with a specific random seed:
```
python main.py --dataset MUTAG --explainer_name beep --base_explainer pgexplainer --seed 42
```
