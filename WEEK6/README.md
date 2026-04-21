# Machine Learning - Classification (Week 5, 6, 7)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Algorithms](https://img.shields.io/badge/Algorithms-Classification-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

*A hands-on implementation of supervised learning classification algorithms including Naive Bayes and K-Nearest Neighbors, applied to real-world datasets such as Titanic.*

</div>

---

## Table of Contents
- [About This Project](#about-this-project)
- [Topics Covered](#topics-covered)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Algorithms Overview](#algorithms-overview)
- [Results & Analysis](#results--analysis)
- [References](#references)

---

## About This Project

This repository contains practical implementations and learning materials for **Supervised Learning - Classification**, focusing on fundamental machine learning algorithms.

The project explores how machines can learn patterns from labeled data and make predictions on unseen data. It includes both theoretical explanations and hands-on experiments using real datasets.

A key case study in this project is the **Titanic dataset**, where models are trained to predict passenger survival based on features such as age, class, and gender.

**Developed for:** Machine Learning Course  
**Focus:** Classification Algorithms (Naive Bayes, KNN)  
**Approach:** Theory + Hands-on Implementation  

---

## Topics Covered

- Introduction to Classification
- Supervised Learning Concepts
- Naive Bayes Algorithm
- K-Nearest Neighbors (KNN)
- Data Preprocessing & Cleaning
- Handling Missing Values (Imputation)
- Model Evaluation
- Hyperparameter Tuning (Grid Search)

---

## Repository Structure
WEEK6/
│
├── README.md
│
├── Week 5 - 7 - Introduction to classification.pptx
│ ← Lecture slides covering theory of classification
│
├── Week_5_6_7_Supervised_Learning_Hands_On_Classification_All_tanpa_tuning.ipynb
│ ← Main notebook (implementation & experiments)
│
├── titanic.csv
│ ← Dataset used for classification problem
│
├── cara kerja perhitungan naive bayes.xlsx
│ ← Step-by-step manual calculation of Naive Bayes
│
└── hasil_gridsearch_knn.xlsx
← Results of hyperparameter tuning for KNN


---

## Getting Started

### Prerequisites

Make sure you have Python installed (3.8+ recommended), then install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
Running the Project
Open Jupyter Notebook:
jupyter notebook
Run the main notebook:
Week_5_6_7_Supervised_Learning_Hands_On_Classification_All_tanpa_tuning.ipynb
Algorithms Overview
1. Naive Bayes
Based on Bayes Theorem
Assumes feature independence
Fast and efficient for classification problems

Used for:

Probability-based prediction
Understanding statistical classification
2. K-Nearest Neighbors (KNN)
Instance-based learning algorithm
Classifies data based on nearest neighbors
Sensitive to value of K

Used for:

Pattern recognition
Simple but powerful classification tasks
Results & Analysis
Dataset Used
Titanic Dataset
Key Steps:
Data cleaning (handling missing values)
Feature selection
Model training
Evaluation
Highlights:
Missing values in Age handled using imputation based on passenger class
KNN optimized using Grid Search
Comparison between model performances
References
Pattern Recognition and Machine Learning
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
Scikit-learn Documentation — https://scikit-learn.org/
Kaggle — Titanic Dataset
<div align="center"> <i>Machine Learning · Classification Study · Politeknik Negeri Batam</i> </div> ```