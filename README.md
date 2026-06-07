# Music Genre Classification from Audio Features

## 1. Introduction

This project explores automatic music genre classification on the Free Music Archive
(FMA) Small dataset. The goal is to predict one of eight parent genres from audio
content only, and to compare two feature-representation strategies:

1. **Hand-crafted audio descriptors extracted with librosa**
2. **Pretrained neural-network embeddings extracted from a pretrained audio model**

The main research question is whether a model trained on manually calculated
statistics such as MFCCs, chroma, and spectral descriptors can match the
performance of embeddings learned by a pretrained deep audio model.

## 2. Dataset

The experiments use the **FMA Small** subset from the Free Music Archive:

https://github.com/mdeff/fma

FMA Small contains approximately 8,000 tracks balanced across eight genres:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

The original FMA split information was used. In the embedding experiment, invalid
or unreadable audio files were skipped, resulting in 6,400 (6 of them were rejected, because they couldn't be processed with the current pipeline) training tracks and
1,600 validation/test tracks.

## 3. Experiment Overview

Three notebooks are compared:

| Notebook | Feature strategy | Feature count before PCA | Main representation |
|---|---:|---:|---|
| `fma_small_librosa_70.ipynb` | librosa, reduced (only mean and std) hand-crafted feature set | 70 | MFCC, chroma, RMS, spectral centroid, ZCR statistics |
| `fma_small_librosa_full.ipynb` | librosa, larger (additionally e.g. skew, curtosis) hand-crafted feature set | 518 | Broad librosa descriptor set with multiple statistics |
| `less_features_musicNN.ipynb` | pretrained model embeddings | 2,048 | PANNs/Cnn14 audio embeddings loaded through `panns_inference` |

Although the last notebook name mentions MusicNN, the implemented pretrained
embedding extractor uses `panns_inference.AudioTagging` with a Cnn14 checkpoint
(`Cnn14_mAP=0.431.pth`). The comparison below therefore treats it as a
pretrained audio-embedding experiment.

## 4. Feature Extraction

### Librosa Features

The librosa-only experiments calculate descriptive statistics directly from the
audio signal. The smaller feature set contains:

- Chroma STFT mean and standard deviation
- MFCC mean and standard deviation
- RMS energy mean and standard deviation
- Spectral centroid mean and standard deviation
- Zero crossing rate mean and standard deviation

The full librosa feature set expands this to 518 features, including additional
spectral, chroma, tonal, and statistical descriptors such as mean, standard
deviation, skewness, kurtosis, minimum, maximum, and median values.

These features are interpretable and relatively cheap to compute, but they
describe audio through fixed, hand-designed measurements.

### Pretrained Embeddings

The pretrained experiment loads audio at 32 kHz and passes it through a
pretrained Cnn14 model from PANNs. For each track, the model produces a
2,048-dimensional embedding.

These embeddings are not manually selected acoustic statistics. They are learned
representations from a neural network previously trained on large-scale audio
tagging data. As a result, they can encode higher-level information such as
instrumentation, timbre, texture, rhythm, and broad sound-event patterns that are
hard to capture with simple summary statistics.

## 5. Preprocessing and Dimensionality Reduction

For the librosa experiments:

- Features were standardized with `StandardScaler`.
- PCA was tested with 4 principal components.
- Models were also trained without PCA.

For the pretrained embedding experiment:

- Embeddings were L2-normalized.
- PCA was evaluated using explained variance.
- 106 components preserved 90% variance.
- 182 components preserved 95% variance.
- Several models were trained on the 95% PCA representation, and LightGBM was
  also tested directly on the normalized embeddings without PCA.

## 6. Models

The notebooks evaluate the following classifiers:

- Support Vector Machine (linear and RBF kernels)
- Random Forest
- LightGBM
- Logistic Regression

The same broad evaluation metrics are used throughout:

- Test accuracy
- Macro F1 score
- Precision and recall through classification reports
- Confusion matrices for selected models

## 7. Results

### Librosa, 70 Features

| Representation | Model | Train accuracy | Test accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| PCA, 4 components | SVM, linear | 0.342 | 0.339 | 0.31 |
| PCA, 4 components | Random Forest | 0.452 | 0.329 | 0.31 |
| No PCA | Random Forest | 0.598 | 0.438 | 0.42 |
| No PCA | SVM, RBF | 0.506 | 0.446 | 0.43 |
| No PCA | LightGBM | 0.738 | 0.444 | 0.44 |
| No PCA | Logistic Regression | 0.506 | 0.394 | 0.39 |

The best 70-feature librosa result is approximately **44.6% test accuracy** with
an RBF SVM. LightGBM achieves a very similar accuracy and slightly higher macro
F1, but with a larger train/test gap.

### Librosa, 518 Features

| Representation | Model | Train accuracy | Test accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| PCA, 4 components | SVM, linear | 0.340 | 0.317 | 0.27 |
| PCA, 4 components | Random Forest | 0.436 | 0.324 | 0.31 |
| No PCA | Random Forest | 0.653 | 0.460 | 0.44 |
| No PCA | SVM, RBF | 0.441 | 0.411 | 0.39 |
| No PCA | LightGBM | 0.844 | 0.472 | 0.47 |
| No PCA | Logistic Regression | 0.705 | 0.449 | 0.45 |

The larger librosa feature set improves the best result from about **44.6%** to
about **47.2%** test accuracy. The best model here is LightGBM, but the high
training accuracy indicates noticeable overfitting.

### Pretrained Embeddings

| Representation | Model | Train accuracy | Test accuracy | Macro F1 / other score |
|---|---:|---:|---:|---|
| Normalized 2,048-d embedding, no PCA | LightGBM | not reported | 0.610 | 0.610 |
| PCA, 182 components | LightGBM | 0.817 | 0.599 | 0.595 |
| PCA, 182 components | Logistic Regression | 0.672 | 0.611 | 0.610 |
| PCA, 182 components | Random Forest, grid search | 0.998 | 0.603 | test macro F1 not reported; CV macro F1 0.631 |

The best reported pretrained-embedding result is **61.1% test accuracy** with
Logistic Regression on PCA-reduced embeddings. LightGBM without PCA reaches
**61.0%** test accuracy and **0.610** macro F1, which is almost identical.

The Random Forest result has competitive test accuracy, but the training accuracy
of 0.998 shows severe overfitting.

## 8. Main Comparison

| Best setup | Test accuracy | Macro F1 | Difference vs best librosa |
|---|---:|---:|---:|
| Best librosa, 70 features | 0.446 | 0.43 | baseline reduced feature set |
| Best librosa, 518 features | 0.472 | 0.47 | baseline full feature set |
| Best pretrained embedding setup | 0.611 | 0.610 | +0.139 accuracy over full librosa |

The pretrained embeddings improve test accuracy by about **13.9 percentage
points** compared with the best full librosa-only model:

```text
0.611 - 0.472 = 0.139
```

Relative to the smaller 70-feature librosa setup, the improvement is about
**16.5 percentage points**:

```text
0.611 - 0.446 = 0.165
```

## 9. Interpretation

The experiments show that hand-crafted librosa features are useful, but they
hit a performance ceiling on this task. Even with 518 calculated descriptors,
the best test accuracy remains below 50%.

The pretrained embeddings perform better because they come from a model that has
already learned general audio representations from a much larger audio-tagging
task. Instead of describing each track only through fixed low-level statistics,
the embedding captures richer patterns across time and frequency.

Key observations:

- Increasing librosa features from 70 to 518 improves accuracy, but only
  moderately.
- PCA with only 4 components loses too much information for the librosa models.
- The best librosa models depend on nonlinear classifiers such as LightGBM or
  RBF SVM.
- Pretrained embeddings work well even with a simple Logistic Regression model.
- The embedding models generalize better, while Random Forest models still
  overfit when they become too flexible.

## 10. Genre-Level Behavior

Across the librosa classification reports, the strongest classes are usually:

- Rock
- Hip-Hop
- Electronic

The weakest class is often:

- Pop

This is expected because Pop overlaps acoustically with several other genres.
Other recurring confusions include:

- Rock vs Pop
- Folk vs International
- Electronic vs Experimental
- Instrumental vs Folk or International

These confusions suggest that genre labels are not always separable using only
low-level acoustic descriptors.

## 11. Conclusions

This project demonstrates a full machine-learning workflow for music genre
classification:

1. Metadata and audio loading
2. Feature extraction
3. Scaling and normalization
4. PCA analysis
5. Model training
6. Evaluation with accuracy, macro F1, and confusion matrices

The main conclusion is that **pretrained audio embeddings substantially
outperform manually calculated librosa features** on FMA Small. The best
librosa-only model reaches about **47.2%** test accuracy, while the best
pretrained-embedding model reaches about **61.1%**.

Librosa features remain valuable because they are interpretable, lightweight,
and useful as a baseline. However, for predictive performance, transfer learning
from pretrained audio models provides a much stronger representation.

## References

1. Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). FMA: A Dataset For Music Analysis.
2. FMA dataset repository: https://github.com/mdeff/fma
3. Librosa documentation: https://librosa.org
4. Scikit-learn documentation: https://scikit-learn.org
5. LightGBM documentation: https://lightgbm.readthedocs.io
6. PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
