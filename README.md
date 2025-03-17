# NLP Coursework - Abbreviation and Longform Detection

## Overview
This project focuses on **sequence classification** for detecting abbreviations and their long forms in scientific text. The task follows the **BIO tagging format** and utilizes the **PLOD dataset**, sourced from PLOS journal articles. The project involves **Named Entity Recognition (NER)** and sequence labeling using various **machine learning and deep learning models**.

### Coursework Details:
- **Module:** Natural Language Processing (COMM061)
- **Dataset:** PLOD-CW (Abbreviation and Longform Detection)
- **Task:** Sequence classification for entity recognition

## Problem Statement
The goal is to **detect abbreviations and their corresponding long forms** in scientific text. Abbreviations in different disciplines can vary, requiring accurate **classification and disambiguation**.

### BIO Labeling Schema:
- **B-AC**: Beginning of an abbreviation/acronym
- **B-LF**: Beginning of a long form
- **I-LF**: Inside a long form
- **B-O**: Other tokens (not abbreviations or long forms)

## Dataset
- **Source**: PLOS journal articles (PLOD dataset)
- **Size**: 50k labeled tokens
- **Splits**:
  - **Training**: 40,000 tokens
  - **Validation**: 5,000 tokens
  - **Test**: 5,000 tokens
- **Features**:
  - Word tokens
  - Part-of-Speech (POS) tags
  - Named Entity Recognition (NER) labels

## Pre-Processing
- **Tokenization**: Processed using standard NLP tokenizers
- **POS Tag Removal**: Only NER labels are retained
- **Data Splitting**: Maintained train-validation-test split

## Model Architectures
### Models Implemented:
- **RNN (Recurrent Neural Network)**
- **BiLSTM (Bidirectional Long Short-Term Memory)**
- **Word Embeddings**:
  - GloVe
  - FastText
- **Loss Functions & Optimizers**:
  - Cross Entropy Loss vs Multi-Class Hinge Loss
  - Adam vs SGD

## Experimentation
### Experiment 1: Comparing NLP Algorithms
- **Models Tested**: BiLSTM vs RNN
- **Findings**: BiLSTM outperformed RNN due to better long-range dependency capture.

### Experiment 2: Comparing Embedding Methods
- **Embeddings Used**:
  - GloVe (50-dimensional)
  - FastText (300-dimensional)
- **Findings**: GloVe embeddings achieved better classification performance than FastText.

### Experiment 3: Loss Function & Optimizer Comparison
- **Loss Functions**: Cross-Entropy vs Multi-Class Hinge Loss
- **Optimizers**: Adam vs SGD
- **Findings**: Cross-Entropy with Adam performed best, while Hinge Loss failed to distinguish entity classes properly.

### Experiment 4: Hyperparameter Optimization
- **Learning Rates Tested**: 0.003, 0.0003, 0.001, 0.01
- **Batch Sizes Tested**: 16, 32, 64
- **Findings**: Best results achieved with **learning rate 0.003** and **batch size 32**.

## Performance Evaluation
Key performance metrics:
- **Accuracy**
- **F1-score (macro-averaged)**
- **Confusion Matrix Analysis**

### Best Model Performance:
| Model  | Embedding | Optimizer | Loss Function | F1-score |
|--------|----------|-----------|--------------|----------|
| BiLSTM | GloVe    | Adam      | Cross-Entropy | **0.7614** |

## Limitations & Future Improvements
- **Dataset Imbalance**: Majority of tokens labeled as **B-O**, biasing predictions.
- **Unknown Tokens**: Handling **<UNK>** tokens remains a challenge.
- **Pretrained Transformer Models**: Implementing **BERT** or **RoBERTa** could improve performance.
- **Data Augmentation**: Expanding dataset with synthetic examples may improve generalization.

## Installation & Usage
### Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
jupyter notebook
# Open and run 'NLP_CW.ipynb'
```

---
Feel free to reach out.

