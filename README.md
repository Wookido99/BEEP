We provide preprocessed datasets and pretrained GNN models to simplify the setup process.\
Note: For the MNIST dataset, you must manually unzip the processed graph file.

Before running any experiments, make sure to install the required packages:
```
pip install -r requirements.txt
```

To run the experiments, use the provided shell script explain.sh.
```
bash explain.sh
```
The script runs the explainer module over the specified dataset using 10 different random seeds (from 0 to 9), and logs the results to a designated directory.\
Before executing the script, edit the following variables inside explain.sh:
- DATASET: name of the dataset (MUTAG, BA3, FC, MNIST)
- EXPLAINER: name of the explainer module (beep)

If you want to run the experiment with a specific random seed:
```
python main.py --dataset MUTAG --explainer_name beep --seed 1
```
