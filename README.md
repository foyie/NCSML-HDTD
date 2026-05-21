# NCSML-HDTD: Network Centrality and Sequence-Based Machine Learning for COVID-19 Drug Target Discovery

A machine learning methodology that combines network centrality measures with protein sequence features to identify potential COVID-19 drug targets.
Paper Link: https://doi.org/10.1007/978-981-99-2680-0_45 


## Overview

NCSML-HDTD is a computational approach designed to identify druggable human proteins relevant to COVID-19 by leveraging protein-protein interaction (PPI) networks and sequence-based features. This methodology enables rapid, cost-effective drug target discovery compared to traditional biological experiments like RNA interference.

### Key Innovation

Unlike traditional centrality-based approaches, NCSML-HDTD combines:
- **Network centrality measures** (12 different metrics) from PPI networks
- **Protein sequence features** (atom type and amino acid composition)
- **Multiple machine learning models** for robust predictions

This dual-feature approach allows the methodology to identify targets even for proteins with limited network interactions by leveraging their sequence properties.

## Features

### Network Centrality Measures (12 metrics)
1. Degree Centrality (DC)
2. Betweenness Centrality (BC)
3. Eigenvector Centrality (EVC)
4. Pagerank Prestige (PRP)
5. Proximity Prestige (PP)
6. Influence Range Closeness Centrality (IRCC)
7. Power Centrality (PC)
8. Katz Centrality
9. HITS Hubs Centrality
10. HITS Authorities Centrality
11. Load Centrality
12. Harmonic Centrality

### Protein Sequence Features
- **Amino Acid Composition (AAC)**: 20 descriptors for standard amino acids
- **Atom Type Composition (ATC)**: 5 descriptors (C, H, N, O, S)

### Machine Learning Models
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Classification (SVC)
- AdaBoost Classifier
- XGBoost Classifier

## Performance

The Random Forest model achieved the best results:
- **Accuracy**: 0.714
- **Precision**: 0.65
- **Recall**: 0.812
- **F1-Score**: 0.722
- **AUC**: 0.787

### Results
- **Detected targets**: 1,135 proteins (1,075 unique genes)
- **Validated overlap**: 569 targets with existing COVID-19 drug target databases
- **Novel targets identified**: 506 previously unidentified potential COVID-19 drug targets

## Methodology

### Workflow


![flowchart2](https://github.com/foyie/Drug-Target-Identification-Using-Biological-Information-and-Node2vec/assets/89987028/9d962b8b-18df-4418-aa65-edf153ddd50a)


### Data

- **Source**: UniProt (reviewed human proteome)
- **Proteins**: 20,461 total
- **Positive samples**: 87 known COVID-19 drug targets (from Gordon et al.)
- **Negative samples**: 87 randomly selected non-targets
- **Dataset**: Balanced, preprocessed

## Installation

### Requirements
- Python 3.7+
- pandas
- scikit-learn
- numpy
- openpyxl (for Excel file handling)

### Quick Start

```bash
pip install pandas scikit-learn numpy openpyxl
```

## Usage

The implementation is provided as a Jupyter notebook (`identification.ipynb`) that can be run in Google Colab or locally.

### Running the Code

#### 1. **Load and Prepare Training Data**

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load the shuffled feature dataset with labels (target vs non-target)
data = pd.read_excel('Shuffled Final Target non target Features1.xlsx')

# Display first few rows
print(data.head())

# Data structure:
# - Node: Protein identifier
# - Entry_Names: UniProt entry names
# - Network features: DC, BC, EVC, PRP, PP, IRCC, PC, Katz Cent, etc. (12 features)
# - Sequence features: AAC_A through AAC_Y (20 amino acids) and ATC features (5 atoms)
# - Label: 1 for drug target, 0 for non-target
```

#### 2. **Data Preprocessing**

```python
# Separate features and labels
X = data.drop(['Node', 'Entry_Names', 'Label'], axis=1)  # 38 features total
y = data['Label']

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize features to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
```

#### 3. **Train Machine Learning Models**

```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train Random Forest (best performing model)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate on test set
y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_pred):.3f}")
```

#### 4. **Predict Novel COVID-19 Drug Targets**

```python
# Load unlabeled proteins for prediction
unlabeled_data = pd.read_excel('Predict.xlsx')
protein_ids = unlabeled_data[['Node', 'Entry_Names']].copy()

# Prepare features
X_unlabeled = unlabeled_data.drop(['Node', 'Entry_Names'], axis=1)

# Handle missing values and normalize
X_unlabeled_imputed = pd.DataFrame(imputer.transform(X_unlabeled), columns=X_unlabeled.columns)
X_unlabeled_normalized = pd.DataFrame(scaler.transform(X_unlabeled_imputed), columns=X_unlabeled.columns)

# Make predictions
predictions = rf.predict(X_unlabeled_normalized)
probabilities = rf.predict_proba(X_unlabeled_normalized)

# Combine results
results = pd.concat([
    protein_ids.reset_index(drop=True),
    X_unlabeled_normalized.reset_index(drop=True),
    pd.DataFrame(predictions, columns=['Prediction']),
    pd.DataFrame(probabilities, columns=['Probability_NonTarget', 'Probability_Target'])
], axis=1)

# Filter predicted drug targets (prediction == 1)
predicted_targets = results[results['Prediction'] == 1]

# Save results
results.to_excel('prediction_results.xlsx', index=False)
predicted_targets.to_excel('predicted_drug_targets.xlsx', index=False)

print(f"Total proteins analyzed: {len(results)}")
print(f"Predicted drug targets: {len(predicted_targets)}")
```

### Example Workflow

```python
# Complete workflow example
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load data
data = pd.read_excel('Shuffled Final Target non target Features1.xlsx')

# 2. Prepare training set
X = data.drop(['Node', 'Entry_Names', 'Label'], axis=1)
y = data['Label']

# 3. Preprocess
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# 5. Train model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# 6. Evaluate
test_score = rf.score(X_test, y_test)
print(f"Model Accuracy: {test_score:.3f}")

# 7. Predict on new data
new_data = pd.read_excel('Predict.xlsx')
X_new = scaler.transform(imputer.transform(new_data.drop(['Node', 'Entry_Names'], axis=1)))
predictions = rf.predict(X_new)

# 8. Save results
results = pd.DataFrame({'Prediction': predictions})
results.to_csv('predictions.csv', index=False)
```

## Data Sources

- **UniProt**: https://www.uniprot.org/
- **Known COVID-19 targets**: 
  - Gordon et al. (2020) - SARS-CoV-2 protein interaction map
  - Saha et al. (2022)
  - Barman et al. (2022)
  - Tehrani et al. (2020)

## Data Format

### Training Dataset Structure
The training data should be in Excel format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Node | int | Protein identifier |
| Entry_Names | str | UniProt entry name |
| DC | float | Degree Centrality |
| BC | float | Betweenness Centrality |
| EVC | float | Eigenvector Centrality |
| PRP | float | Pagerank Prestige |
| PP | float | Proximity Prestige |
| IRCC | float | Influence Range Closeness Centrality |
| PC | float | Power Centrality |
| Katz Cent | float | Katz Centrality |
| HITS Hubs | float | HITS Hubs Centrality |
| HITS Auth | float | HITS Authorities Centrality |
| Load Cent | float | Load Centrality |
| Harmonic Cent | float | Harmonic Centrality |
| AAC_A to AAC_Y | float | Amino Acid Composition (20 features) |
| Label | int | 1 = drug target, 0 = non-target |

**Total Features: 38** (12 network + 26 sequence)

### Prediction Dataset Structure
For predicting novel targets, provide the same feature columns (without Label):

```
Node | Entry_Names | DC | BC | ... | AAC_Y
```

## Project Structure

```
NCSML-HDTD/
├── README.md
├── identification.ipynb           # Main notebook (Jupyter)
├── data/
│   ├── Shuffled Final Target non target Features1.xlsx  # Training data
│   ├── Predict.xlsx                                     # Data for prediction
│   └── results/
│       ├── prediction_results.xlsx      # Full results with probabilities
│       └── predicted_drug_targets.xlsx  # Filtered targets only
└── output/
    └── out1.csv                   # Normalized feature matrix
```

## Colab Integration

The notebook is optimized for Google Colab with Google Drive integration:

```python
from google.colab import drive
drive.mount('/content/drive')

# Access your files via /content/drive/MyDrive/...
```

## Data Processing Pipeline

```
Raw Excel Data
    ↓
Drop protein identifiers
    ↓
Impute missing values (mean strategy)
    ↓
Normalize to [0, 1] range (MinMaxScaler)
    ↓
Train/Test split (80/20)
    ↓
Train ML models
    ↓
Evaluate and select best model (RF)
    ↓
Predict on new data
    ↓
Save results to Excel
```

## Citation

If you use NCSML-HDTD in your research, please cite:

```bibtex
@inproceedings{jha2023ncsml,
  title={NCSML-HDTD: Network Centrality and Sequence-Based Machine Learning Methodology for Human Drug Targets Discovery of COVID-19},
  author={Jha, Shalini and Das, Chandrima and Saha, Sovan},
  booktitle={Proceedings of International Conference on Frontiers in Computing and Systems},
  pages={515--523},
  year={2023},
  publisher={Springer Nature Singapore Pte Ltd}
}
```

## Authors

- **Shalini Jha** - Institute of Engineering and Management, Kolkata
- **Chandrima Das** - Institute of Engineering and Management, Kolkata
- **Sovan Saha** - Techno Main Salt Lake, Kolkata (corresponding author)

Contact: sovansaha12@gmail.com

## License

This project is provided under the license specified in the LICENSE file.

## Acknowledgments

This work extends existing methodologies in essential protein prediction and applies them to COVID-19 drug target discovery. We acknowledge the foundational work of:
- Min et al. (2011) - Local average connectivity-based methods
- Jeong et al. (2005) - Central lethal rule
- Gordon et al. (2020) - SARS-CoV-2 protein interaction map

## Related Work & References

Key literature referenced in this work:

1. **Drug Target Identification**: Bakheet & Doig (2009) - Properties and identification of human protein drug targets
2. **Network Analysis**: Hahn & Kern (2005) - Comparative genomics of centrality and essentiality
3. **Machine Learning**: El-Behery et al. (2021) - Efficient ML model for drug-target interactions
4. **Deep Learning**: Zhang et al. (2020) - DeepHE for essential genes prediction
5. **COVID-19 Interactions**: Gordon et al. (2020) - SARS-CoV-2 protein interaction map

For a complete reference list, see the original paper.

## Future Directions

- Extension to other diseases beyond COVID-19
- Integration of additional biological features (structural properties, expression data)
- Deep learning approaches for improved predictions
- Validation through wet-lab experiments
- Development of interactive visualization tools

## Model Details

### Random Forest Configuration
- **Algorithm**: Random Forest Classifier
- **Parameters**: n_estimators=100, random_state=42
- **Input Features**: 38 (12 network + 26 sequence)
- **Output**: Binary classification (0: non-target, 1: drug target)

### Alternative Models Implemented
1. **Logistic Regression (LR)**: Fast baseline model
2. **Decision Tree (DT)**: Interpretable results
3. **Support Vector Classification (SVC)**: Non-linear decision boundaries
4. **AdaBoost Classifier**: Ensemble method with boosting
5. **XGBoost Classifier**: Gradient boosting optimization

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Random Forest | 0.714 | 0.65 | 0.812 | 0.722 | 0.787 |


*Note: RF consistently outperforms other models in this application*


## Contributing

Improvements welcome! Consider:

1. **Model Optimization**
   - Hyperparameter tuning for each model
   - Cross-validation for robust evaluation
   - Feature importance analysis

2. **Feature Engineering**
   - Interaction terms between network and sequence features
   - Derived features from atomic/amino acid composition
   - Dimensionality reduction (PCA, t-SNE)

3. **Data Enhancement**
   - Integration with additional protein databases
   - Temporal analysis if time-series data available
   - Multi-label classification for proteins with multiple functions

4. **Reproducibility**
   - Document exact scikit-learn versions
   - Add random seed management
   - Create data validation pipelines

5. **Validation**
   - Experimental wet-lab validation of predictions
   - Comparison with other drug discovery databases
   - Cross-species validation

## Future Directions

- **Deep Learning**: Implement neural networks (RNN, CNN) for feature learning
- **Transfer Learning**: Leverage pre-trained protein models
- **Multi-disease**: Adapt methodology to other diseases
- **Explainability**: SHAP values for model interpretability
- **Interaction Prediction**: Predict drug-protein binding affinity
- **Visualization**: Interactive dashboard for results exploration



* This repository consists of the main code file and the data that was used.
* This paper got approved in 3rd International Conference on Frontiers in Computing and Systems (COMSYS-2022) held at IIT Ropar
* It was awarded the best paper in the conference
* The paper is currenty published in Proceedings of International Conference on Frontiers in Computing and Systems(Springer)
* Cite this paper:

PAPER LINK: Jha, S., Das, C., Saha, S. (2023). NCSML-HDTD: Network Centrality and Sequence-Based Machine Learning Methodology for Human Drug Targets Discovery of COVID-19. In: Sarkar, R., Pal, S., Basu, S., Plewczynski, D., Bhattacharjee, D. (eds) Proceedings of International Conference on Frontiers in Computing and Systems. COMSYS 2022. Lecture Notes in Networks and Systems, vol 690. Springer, Singapore. https://doi.org/10.1007/978-981-99-2680-0_45 
