# Tetramesa Prediction

Image-based detection of new species within the Tetramesa genus.



## Installation

From source 
```
git clone https://github.com/JoshVStaden/tetramesa_predict.git
cd tetramesa_predict
pip install -e .
```

## To Load the Data

Currently, the dataset still needs to be hosted somewhere. Otherwise, the .pkl files have already been included in the repository.

When it is loaded, run

```
python save_as_pkl.py --train /path/to/train_set --test /path/to/test_set
```

This should output two files: train.pkl and test.pkl.


## To Evaluate a Model Predictor

First, ensure that there is a train.pkl and test.pkl. Then, run

```
python train.py
```