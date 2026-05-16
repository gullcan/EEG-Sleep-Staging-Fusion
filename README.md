# EEG-Based Automatic Sleep Stage Classification with Manual Features and Fusion Models

This project focuses on automatic sleep stage classification from single-channel EEG signals using the Sleep-EDF dataset. The work is inspired by the paper **“Automated sleep staging from single-channel electroencephalogram using hybrid neural network with manual features and attention”** by Wan et al. (iScience, 2025).

The goal of this project is not only to run an existing GitHub implementation, but also to build a more reliable and academically defensible sleep staging pipeline. The project includes subject-independent data splitting, memory-efficient lazy loading, handcrafted EEG feature extraction, raw EEG deep learning models, feature fusion models, attention-based models, and detailed class-wise evaluation.

---

## Project Motivation

Automatic sleep staging is an important biomedical signal processing problem. Manual sleep scoring is time-consuming and requires expert knowledge. EEG-based automatic classification can support sleep analysis by predicting sleep stages such as:

- Wake (W)
- N1
- N2
- N3
- REM

A major challenge in this task is that some stages, especially **N1**, are difficult to classify because they represent transitional sleep states and share characteristics with both wakefulness and deeper sleep stages.

---

## Main Contributions

The original repository was extended and reorganized with the following contributions:

1. Implemented a memory-efficient lazy loading data pipeline.
2. Replaced random sequence-level splitting with subject-independent train/validation/test splitting.
3. Added subject overlap checks to reduce data leakage risk.
4. Added label distribution analysis for train, validation, and test sets.
5. Implemented a 35-dimensional handcrafted EEG feature extraction pipeline.
6. Generated manual feature cache files for the full Sleep-EDF dataset.
7. Developed a ManualFeature-BiLSTM baseline model.
8. Developed a raw EEG + manual feature fusion model.
9. Developed a fusion model with temporal attention.
10. Added detailed evaluation metrics:
   - Accuracy
   - Macro F1-score
   - Cohen’s Kappa
   - Classification report
   - Confusion matrix
   - N1-specific error analysis
11. Generated thesis-ready result tables and figures.

---

## Dataset

This project uses the Sleep-EDF dataset.

Each EEG recording is preprocessed into 30-second epochs. Since the sampling rate is 100 Hz, each epoch contains:

```text
100 Hz × 30 seconds = 3000 samples