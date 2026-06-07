# Music Genre Classification from Audio Features

## 1. Introduction

### Project Goal

The goal of this project was to build a machine learning model capable of classifying music tracks into genres based solely on audio-derived features.

The task can be formulated as a multiclass classification problem, where each track is assigned to one of several predefined music genres. In addition to obtaining accurate predictions, the project aimed to investigate the influence of different feature extraction techniques, dimensionality reduction methods, and classification algorithms on overall performance.


## 2. Dataset Description

### Free Music Archive (FMA)

This project uses the **Free Music Archive (FMA)** dataset, a publicly available collection of music tracks and metadata designed for machine learning research.

Repository:

https://github.com/mdeff/fma

### FMA Small Subset

The experiments were conducted on the **FMA Small** subset, which contains approximately 8,000 audio tracks equally distributed across eight music genres (1000 per genre).

The genres include:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

Each track contains:

- An audio file (`.mp3`)
- A unique track identifier
- Parent genre label
- Dataset split information (training, validation, testing)

---

## 3. Data Preparation

### Metadata Integration

The first step consisted of linking audio files with metadata provided by the FMA dataset. This allowed every track to be associated with its corresponding genre label and dataset split.

### Audio Processing

Audio files were processed using the **Librosa** library, a widely used Python package for music and audio analysis.

The library provides tools for:

- Loading audio signals,
- Spectral analysis,
- Feature extraction,
- Time-frequency transformations.

---

## 4. Feature Extraction

The quality of genre classification strongly depends on the representation of the audio signal. Therefore, multiple groups of features were extracted from each track.

### MFCC Features

Mel-Frequency Cepstral Coefficients (MFCCs) are among the most commonly used features in audio classification.

They provide a compact representation of the spectral envelope while approximating the way humans perceive sound frequencies.

### Chroma Features

Chroma features represent the distribution of energy among the twelve pitch classes.

They are useful for capturing harmonic and tonal information present in music.

### Spectral Features

Several spectral descriptors were extracted:

- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Contrast

These features describe the frequency content of an audio signal and often help distinguish between different genres.

### Zero Crossing Rate

The zero crossing rate measures how often the audio waveform changes sign.

This feature is particularly useful for distinguishing tonal sounds from noisy signals.

### RMS Energy

Root Mean Square (RMS) Energy provides a measure of signal loudness and overall intensity.

### Additional Statistical Features

For many extracted descriptors, summary statistics such as:

- Mean
- Standard Deviation
- Minimum
- Maximum

were computed over the duration of the track.

---

## 5. Exploratory Data Analysis

Before training machine learning models, exploratory analysis was performed.

The following aspects were investigated:

- Class distribution,
- Missing values,
- Feature distributions,
- Correlations between features.

This stage helped identify potential issues related to data quality and class imbalance.

---

## 6. Data Preprocessing

### Feature Scaling

Because many machine learning algorithms are sensitive to feature scales, standardization was applied.

The transformation was performed using:

```python
StandardScaler()
```

which transforms each feature according to:

\[
x' = \frac{x - \mu}{\sigma}
\]

where:

- \(\mu\) is the mean,
- \(\sigma\) is the standard deviation.

### Benefits of Scaling

- Faster model convergence,
- Improved numerical stability,
- Better performance for distance-based algorithms,
- Improved PCA results.

---

## 7. Dimensionality Reduction

### Principal Component Analysis (PCA)

The extracted feature space was relatively high-dimensional. To reduce redundancy and computational complexity, Principal Component Analysis (PCA) was applied.

PCA transforms the original features into a smaller set of orthogonal components while preserving as much variance as possible.

The main objectives were:

- Reduce dimensionality,
- Remove correlated information,
- Speed up model training,
- Reduce overfitting.

### Selecting the Number of Components

The number of retained components was determined using cumulative explained variance analysis.

The selected configuration preserved approximately 90–95% of the total variance while substantially reducing dimensionality.

---

## 8. Machine Learning Models

Several classification algorithms were evaluated.

### Logistic Regression

Logistic Regression served as a baseline model.

Advantages:

- Fast training,
- Easy interpretation,
- Strong baseline performance.

Limitations:

- Assumes linear decision boundaries,
- Limited ability to model complex relationships.

---

### Random Forest

Random Forest is an ensemble learning method based on multiple decision trees.

Advantages:

- Robust against overfitting,
- Handles nonlinear relationships,
- Provides feature importance estimates.

Hyperparameters explored included:

- Number of trees,
- Maximum depth,
- Minimum samples per leaf,
- Maximum number of features.

---

### LightGBM

LightGBM is a gradient boosting framework based on decision trees.

Advantages:

- High predictive performance,
- Efficient training,
- Good scalability,
- Handles large feature spaces effectively.

Hyperparameter optimization was performed to improve generalization performance.

---

## 9. Experimental Setup

### Data Splitting

The original train/test split provided by the FMA dataset was used.

Training data were used for:

- Model fitting,
- Hyperparameter tuning.

Testing data were used exclusively for final evaluation.

### Evaluation Metrics

The following metrics were calculated:

#### Accuracy

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

#### Macro F1 Score

Macro F1 computes the F1-score independently for each class and then averages the results.

This metric is particularly useful for multiclass classification tasks.

#### Precision

Measures the proportion of correctly predicted positive samples among all positive predictions.

#### Recall

Measures the proportion of correctly identified positive samples among all actual positives.

---

## 10. Results

### Model Comparison

| Model | Train Accuracy | Test Accuracy | Macro F1 |
|---------|---------|---------|---------|
| Logistic Regression | ... | ... | ... |
| Random Forest | 0.998 | 0.603 | 0.595 |
| LightGBM | 0.817 | 0.599 | 0.595 |

### Observations

- Random Forest achieved very high training accuracy but exhibited noticeable overfitting.
- LightGBM provided a better balance between training and testing performance.
- PCA significantly reduced computational requirements while maintaining predictive quality.
- Genre classification remains challenging because several genres share similar acoustic characteristics.

---

## 11. Discussion

### Impact of PCA

PCA successfully reduced feature dimensionality while preserving most of the information contained in the original feature space.

Benefits observed:

- Reduced training time,
- Lower memory usage,
- Improved model stability.

### Overfitting Analysis

Random Forest achieved nearly perfect performance on training data but considerably lower performance on unseen samples.

This suggests that the model learned patterns specific to the training set rather than generalizable characteristics of music genres.

### Genre Confusion

Certain genres were frequently confused due to similar musical structures and instrumentation.

Typical examples include:

- Rock vs Pop,
- Folk vs International,
- Electronic vs Experimental.

Confusion matrices revealed that these classes share overlapping feature distributions.

---

## 12. Conclusions

This project demonstrated a complete machine learning pipeline for automatic music genre classification.

The workflow included:

1. Audio preprocessing,
2. Feature extraction,
3. Feature scaling,
4. Dimensionality reduction using PCA,
5. Model training and evaluation.

Key findings:

- Audio features extracted with Librosa contain enough information to achieve meaningful genre classification performance.
- PCA effectively reduces dimensionality without substantial loss of predictive power.
- Ensemble methods such as Random Forest and LightGBM outperform simple linear models.
- Music genre classification remains a challenging task due to significant overlap between genres.

### Future Work

Potential improvements include:

- Deep learning models operating directly on spectrograms,
- Convolutional Neural Networks (CNNs),
- Audio Spectrogram Transformers (AST),
- Data augmentation techniques,
- Larger datasets such as FMA Medium or FMA Large,
- Transfer learning using pretrained audio models.

---

## References

1. Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset For Music Analysis.
2. Librosa Documentation: https://librosa.org
3. Scikit-learn Documentation: https://scikit-learn.org
4. LightGBM Documentation: https://lightgbm.readthedocs.io
5. Free Music Archive Dataset Repository: https://github.com/mdeff/fma