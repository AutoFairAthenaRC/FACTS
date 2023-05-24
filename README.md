# Fairness Aware Counterfactuals for Subgroups

This repository is the implementation of the paper Fairness Aware Counterfactuals for Subgroups.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

----> lets say the first image of the paper (if-thens).

[Comparative Subgroup Counterfactuals]()

## Requirements

All experiments were run on the [Anaconda](https://www.anaconda.com/) platform. Creating a conda environment is generally recommended to avoid package version collisions. You can do that with:

```setup
conda create --name facts 
```

and then activate it with

```setup
conda activate facts
```

 To install requirements:

```setup
conda install -c conda-forge numpy pandas scikit-learn matplotlib mlxtend omnixai xgboost notebook
```

Optional Dependency on IBM Fairness 360 if you would like to run on the exact same "COMPAS" dataset as we did:

```setup
conda install -c conda-forge aif360
```


## Model Training

A model is needed for auditing purposes. We trained, in our paper, a logistic regression classifier. Any other classification model could be used.

Specifically, our method expects a model with the form of `facts.models.ModelAPI`, which means simply any python object with a `predict` method which takes as input a DataFrame containing the dataset and outputs a 1-dimensional prediction array of 0s and 1s.



## Pre-trained Models / Pre-computed results

In the scope of precomputed results, we provide:
- full sets of precomputed rules for each dataset (with a frequent itemset minimum support of 1%).
- only for the Adult dataset with 'race' as the protected attribute, we have also provided a file which, in addition to the rules, contains the actual model and the test data we used, which are the main inputs required by our framework. Note: for reading this file correctly, you will probably need to have `scikit-learn` version 1.0.2 installed.
- finally, all experiments we ran use 131313 as the value of the random_state parameter, wherever applicable (notice that, for example, the `LogisticRegression` model with default parameters is deterministic).

The above assets can be found in the directory [Pre-computed Results](facts/Pre-computed Results/). The commands required to read them are displayed in the Jupyter notebooks we provide (more on this in the following sections).

## Auditing for Fairness

We provide notebooks with examples and detailed instructions of how our framework can be used in order to audit fairness. All these reside in the [Notebooks](facts/Νotebooks/) directory (for now, only for the Adult Dataset).

## Results ?

We have implemented the rules for definitions of fairness ...

According to these, we rank the groups and the first 

-- All in paper: examples, table from main

 The framework allows to query with sugroup id and with rank and perform comparative evaluation / summary for the results, e.g. ....

For more information, see jupyter notebook ...

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing ?

>📋  Pick a licence and describe how to contribute to your code repository. 
