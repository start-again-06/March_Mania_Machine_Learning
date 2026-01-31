# March Machine Learning Mania 2025
Predicting NCAA Tournament Outcomes with AI

A machine learning project that predicts win probabilities for NCAA Men’s and Women’s tournament matchups using historical data and a CatBoost binary classification model. Built for Kaggle’s March Machine Learning Mania 2025 competition and focused on practical sports analytics.

# Dataset & Inputs
Uses official Kaggle NCAA datasets for both men’s and women’s tournaments.
Includes tournament results, regular season results, and team metadata.
Training data is augmented by flipping win/loss pairs to improve generalization.

# Project Pipeline

Data Loading & Preprocessing
Load men’s and women’s tournament and regular season results.
Generate win/loss records and construct matchup-level training samples.

# Feature Engineering
Features: Season, WTeamID, LTeamID.
Target: Match outcome (Win = 1, Loss = 0).

# Model Training
Model: CatBoostClassifier.
Evaluation Metric: Brier Score Loss.
Train/Test Split: 82% / 18%.
Iterations: 20,000.
Learning Rate: 0.3.
Depth: 10.

# Prediction & Submission
Predict win probabilities for all 2025 tournament matchups.
Generate Kaggle-ready submission.csv in ID–Pred format.
Visualize prediction confidence using probability histograms.

# Visualization & Insights
Predictions near 0 or 1 indicate high-confidence matchups.
Predictions near 0.5 indicate uncertain or evenly matched games.

# Dependencies
Python
NumPy
Pandas
Matplotlib
Seaborn
CatBoost
scikit-learn

# Execution
Run the notebook end-to-end in a Kaggle environment.
The pipeline automatically handles data loading, training, prediction, visualization, and submission file generation.

# Applications
Kaggle competition submissions.
Sports analytics and outcome prediction.
Applied machine learning on structured tabular data.

# License
Intended for educational, research, and competition use.
