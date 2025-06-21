# March_Mania_Machine_Learning
# 🏀 March Machine Learning Mania 2025: Predicting NCAA Tournament Outcomes with AI

Welcome to the court where **machine learning meets madness**!  
This project uses CatBoost, historical NCAA tournament data, and a sprinkle of data science magic ✨ to **predict match win probabilities** for the 2025 Men's and Women's tournaments.

Whether you're here to **compete in a Kaggle competition**, **learn about sports analytics**, or just explore ML on real-world data, this repo’s got you covered!

---

## 📁 Project Structure

This repo includes:
- 📊 Data ingestion & preprocessing  
- 🤖 Model training using CatBoost  
- 🔮 Matchup win probability predictions  
- 📤 Submission file generator  
- 📈 Visualization of model confidence

---

## 📦 Dataset & Files

Expected directory:  
`/kaggle/input/march-machine-learning-mania-2025`

### Files used:
- `MNCAATourneyCompactResults.csv`  
- `MRegularSeasonCompactResults.csv`  
- `MTeams.csv`  
- `WNCAATourneyCompactResults.csv`  
- `WRegularSeasonCompactResults.csv`  
- `WTeams.csv`  
- `SampleSubmissionStage2.csv`

---

## ⚙️ Setup Instructions

### 1. 🧰 Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn catboost scikit-learn
Already using Kaggle notebooks? You're all set — these are pre-installed!

2. 🚀 Run the Code
Load and run the script as-is in a Kaggle notebook. The flow is:

Import dependencies

Load data (men's & women's tournaments)

Generate win/loss records

Train a binary classification model

Predict matchups for 2025 season

Save submission as submission.csv

Visualize prediction confidence

🤖 Model Details
Feature	Value
Model	CatBoostClassifier
Key Inputs	Season, WTeamID, LTeamID
Target	Win (1) / Loss (0)
Evaluation	Brier Score Loss
Training Split	82% train / 18% test
Iterations	20,000
Learning Rate	0.3
Depth	10

📤 Sample Output
submission.csv (Kaggle-ready):

python-repl
Copy
Edit
ID,Pred
2025_1101_1102,0.743
2025_1240_1422,0.391
...
Metric:

yaml
Copy
Edit
Brier Score: 0.1892
📈 Insights & Visualization
The script includes a histogram to show how confident the model is across matchups:

python
Copy
Edit
sns.histplot(submission_df['Pred'], bins=50, kde=True)
✅ Clustering around 0 or 1 → strong confidence
⚠️ Centered around 0.5 → uncertainty in matchup

🧪 Why It Works
✅ Combines tournament data for better generalization
✅ Doubles training data by flipping win/loss pairs
✅ Uses CatBoost’s power to handle categorical-like integer IDs
✅ Simple features — surprisingly effective!

💡 Ideas for Improvement
Want to take this project further? Try:

Adding seed info, margin of victory, or ELO ratings

Including regular season games for more training data

Using more advanced models (e.g., LightGBM, XGBoost, neural nets)

Hyperparameter tuning with Optuna or GridSearchCV

🤝 Contributing
Love college basketball or machine learning?
Fork this repo, open a PR, or drop ideas in Issues. Let's build a smarter bracket together! 🧠📈
