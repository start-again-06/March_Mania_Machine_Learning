# March Machine Learning Mania 2025
Predicting NCAA Tournament Outcomes with AI

# System Overview
This system predicts win probabilities for NCAA Men’s and Women’s tournament matchups using historical game data and a supervised machine learning pipeline. The architecture is designed for reproducibility, scalability, and direct deployment within the Kaggle competition environment.

# High-Level Architecture

Data Layer
- Source: Kaggle NCAA datasets (Men’s and Women’s)
- Inputs:
  - Tournament results
  - Regular season results
  - Team metadata
- Storage: CSV-based datasets loaded directly from Kaggle input directory

Data Processing Layer
- Data ingestion and validation
- Cleaning and normalization of historical results
- Generation of win/loss records
- Dataset augmentation by flipping win/loss pairs
- Matchup-level sample construction

# Feature Engineering Layer
- Core Features:
  - Season
  - Winning Team ID (WTeamID)
  - Losing Team ID (LTeamID)
- Target Variable:
  - Binary classification label (Win = 1, Loss = 0)
- Feature format optimized for tree-based models

# Modeling Layer
- Model Type: Supervised Binary Classification
- Algorithm: CatBoostClassifier
- Advantages:
  - Handles categorical-like integer identifiers
  - Robust to feature scaling
  - Strong performance on tabular data
- Training Configuration:
  - Iterations: 20,000
  - Learning Rate: 0.3
  - Depth: 10
  - Train/Test Split: 82% / 18%

# Evaluation Layer
- Metric: Brier Score Loss
- Focus: Calibration and probability accuracy
- Validation on held-out test set

# Inference Layer
- Input: All possible 2025 tournament matchups
- Output: Win probability for each matchup
- Format:
  - ID: Season_Team1_Team2
  - Pred: Probability of Team1 winning

# Submission & Output Layer
- Generate Kaggle-ready submission.csv
- Ensure compatibility with Kaggle evaluation pipeline
- Store outputs locally within notebook runtime

# Visualization & Monitoring Layer
- Probability distribution analysis using histograms
- Confidence interpretation:
  - Values near 0 or 1 indicate high-confidence predictions
  - Values near 0.5 indicate uncertain matchups

# Dependencies
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- CatBoost
- scikit-learn

# Execution Flow
1. Load datasets from Kaggle input directory
2. Preprocess and augment historical game data
3. Engineer matchup-level features
4. Train CatBoost classification model
5. Evaluate using Brier Score Loss
6. Generate predictions for 2025 matchups
7. Create submission.csv
8. Visualize prediction confidence

# Scalability & Extensibility
- Additional features (seeds, margins, ELO ratings) can be integrated
- Regular season data can be expanded for improved learning
- Model can be swapped with LightGBM, XGBoost, or neural networks
- Hyperparameter tuning supported via GridSearchCV or Optuna

# Applications
- Kaggle March Machine Learning Mania competition
- Sports analytics and outcome probability modeling
- Demonstration of end-to-end ML system design on tabular data

# License
Intended for educational, research, and competition use. Further validation is recommended before production deployment.
