# Music Genre Classification from Audio Features
- **Goal:** classify music tracks into genres and analyze which audio features are most discriminative
- **Data:** Free Music Archive dataset (official repository): https://github.com/mdeff/fma
## Overview
This project aims to classify music genres using the Free Music Archive (FMA) dataset. It involves audio signal processing, feature engineering, and machine learning.
## Progress Tracking: What’s Done So Far? (FMA SMALL)
1. Data Preparation & Exploration
   - Metadata Integration: Successfully mapped audio file paths to the FMA metadata database.
   - Audio Processing: Explored the use of librosa for audio feature extraction.
2. Feature Engineering
   - Dimensionality Reduction: Implemented PCA (Principal Component Analysis) to handle the high-dimensional feature set while preserving significant variance.
   - Scaling: Standardized features to ensure optimal performance for distance-based and gradient-based algorithms.
3. Machine Learning Modeling
   - Baseline Models: Established initial benchmarks for classification accuracy.
   - Gradient Boosting: Implemented a Gradient Boosting Classifier, achieving promising results on the fma_small subset.
   - Performance Evaluation: Tracked training vs. test accuracy to monitor and mitigate overfitting.


Hyperparameters testing table: https://docs.google.com/spreadsheets/d/1zVdp_KwDR07jjUFOIKxyNs_meEN2bUFtcPnCqbhq_AA/edit?gid=0#gid=0
## Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, Librosa, Matplotlib, Seaborn
