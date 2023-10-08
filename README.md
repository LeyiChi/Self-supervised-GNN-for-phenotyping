# Self-supervised GNN for phenotyping
This repository implements a self-supervised graph neural network for identifying temporal phenotypes from longitudinal electronic health records.


## Usage

### Clone the repository:<br />

```
git clone https://github.com/LeyiChi/Self-supervised-GNN-for-phenotyping.git
```

### Requirement
1. Python 3.6
2. torch 1.10.1
3. pandas 1.1.5
4. sklearn 0.24.2
5. R 4.3.1
6. shap 0.41.0
7. survival 3.5.5
Some other libraries (find what you miss when running the code)


### Data availability
Protected Health Information restrictions apply to the availability of the clinical data here, which were used under IRB approval for use only in the current study. As a result, this dataset is not publicly available.

### Data preparation
1. Data extraction are done using sql from the electronic health record system. The codes are not provided as the dataset is not publicly available.
2. Run 1-data preprocess.py for data preprocessing to screen included patients and visits.

### Subphenotype derivation using self-supervised graph neural network
1. Get initial embeddings for nodes
```
python 2-initial embeddings for nodes.py
```
2. Visit hierarchy network construction
```
python 3-Visit hierarchy network.py
```
3. Autoencoder pretraining
```
python 4.1-autoencoder-pretraining.py
```
4. The self-supervised graph clustering model training
```
python 4.2-Self supervised GNN model training.py
```
5. Model validation on the testing data and patient nodes visualization
```
python 5-Validation and visualization.py
```

### Subphenotype analysis
1. data preparing for subphenotype analysis
```
python 6.1-data preparing for subphenotype analysis.py
```
2. data cleaning 
```
python 6.2-data cleaning.py
```
3. data distribution and statistics
```
python 6.3-data distribution and statistics.py
```
KM curves and chord diagram are in R scripts.
```
7.1-km_curve.Rmd
7.2-chord_diagram.Rmd
```
4. LDA model for disease status generation
```
python 6.4 LDA model for disease status generation.py
```
5. To get insights into the disease status and visit behaviors, run
```
python 6.5-visit-stats-1.py
python 6.6-visit-stats-2.py
python 6.7-visit-stats-3.py
```
6. Subphenotype prediction model construction and evaluation
```
python 6.8-subphenotype prediction model.py
```
7. Prognosis prediction model
The prognosis prediction model was built using Cox regression. Model construction and evaluation were implemented in the following script.
```
8-cox_regression.Rmd
```
8. Important interventions discovery
```
9-psm-train.Rmd
```

