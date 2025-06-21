import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
input_folder = r"/kaggle/input/march-machine-learning-mania-2025"
mens_files = ["MNCAATourneyCompactResults.csv", "MRegularSeasonCompactResults.csv", "MTeams.csv"]
womens_files = ["WNCAATourneyCompactResults.csv", "WRegularSeasonCompactResults.csv", "WTeams.csv"]

dataframes = {}

def load_csv_files(file_list):
    for file in file_list:
        file_path = os.path.join(input_folder, file)
        key = os.path.splitext(file)[0]
        try:
            dataframes[key] = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file}: {e}")
def preprocess_results(df):
    df = df[['Season', 'WTeamID', 'LTeamID']].copy()
    df.loc[:, 'Result'] = 1  # Win label
    df_inv = df.copy()
    df_inv[['WTeamID', 'LTeamID']] = df[['LTeamID', 'WTeamID']].values
    df_inv.loc[:, 'Result'] = 0  # Loss label
    return pd.concat([df, df_inv], ignore_index=True)

men_results = preprocess_results(dataframes['MNCAATourneyCompactResults'])
women_results = preprocess_results(dataframes['WNCAATourneyCompactResults'])

all_results = pd.concat([men_results, women_results], ignore_index=True)
X = all_results[['Season', 'WTeamID', 'LTeamID']]
y = all_results['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.18, random_state=42)

model = CatBoostClassifier(iterations= 20000, depth= 10, learning_rate= 0.3, loss_function='Logloss', verbose= 3000)
model.fit(X_train, y_train, eval_set=(X_test, y_test))
preds = model.predict_proba(X_test)[:, 1]
print(f"Brier Score: {brier_score_loss(y_test, preds)}")
all_teams = np.concatenate((dataframes['MTeams']['TeamID'].values, dataframes['WTeams']['TeamID'].values))
pairings = list(combinations(all_teams, 2))
required_ids_df = pd.read_csv(os.path.join(input_folder, "SampleSubmissionStage2.csv"))
required_ids = set(required_ids_df['ID'])

def create_submission(pairings, season=2025, max_rows=131407):
    submission = []
    for (team1, team2) in pairings:
        matchup_id = f"{season}_{min(team1, team2)}_{max(team1, team2)}"
        if matchup_id in required_ids:
            input_data = pd.DataFrame({'Season': [season], 'WTeamID': [min(team1, team2)], 'LTeamID': [max(team1, team2)]})
            pred = model.predict_proba(input_data)[0, 1] if len(input_data) > 0 else 0.5
            submission.append([matchup_id, pred])
    submission_df = pd.DataFrame(submission, columns=["ID", "Pred"])
    print(f"Submission file has {submission_df.shape[0]} rows.")
    return submission_df

submission_df = create_submission(pairings)
submission_df.to_csv("submission.csv", index=False)
print("Submission file created successfully.")
plt.figure(figsize=(10, 6))
sns.histplot(submission_df['Pred'], bins=50, kde=True)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.show()
