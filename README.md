# Kaggle Digit Recognizer (MNIST) - CNN Ensemble

## Overview
This project solves Kaggle "Digit Recognizer" (MNIST) classification task using a CNN model.

To improve generalization performance, the model uses:

- Data augmentation
- Stratified 5-Fold cross-validation
- 2-seed ensemble (average of predicted probabilities)

## Result
Public leaderboard score: **0.99603**  
Rank: **61 / 907**

## Tech Stack
- Python
- NumPy
- pandas
- scikit-learn (StratifiedKFold)
- TensorFlow / Keras

## Approach

### Model
- Conv2D + BatchNormalization blocks
- MaxPooling + Dropout
- Dense (256) + Dropout
- Softmax output (10 classes)

### Training strategy
- Data augmentation: rotation / zoom / shift
- EarlyStopping (monitor = val_accuracy)
- ReduceLROnPlateau (monitor = val_loss)
- 5-Fold CV × 2 seeds
- Ensemble: mean of predicted probabilities

## How to run

### Option A: Kaggle Notebook
1. Create a Kaggle Notebook for Digit Recognizer.
2. Add the competition dataset.
3. Run `train_kaggle.py`.
4. `submission.csv` will be generated.

### Option B: Local (if you have the dataset)

1. Put `train.csv` and `test.csv` under `./input/`.

2. Install dependencies:

```bash
pip install -r requirements.txt

Run

python train_local.py
Output

submission.csv (ImageId, Label)

Repository Structure
digit-recognizer-ensemble/
├─ README.md
├─ requirements.txt
├─ train_kaggle.py
├─ train_local.py
└─ assets/
   └─ training_curve.png (optional)
