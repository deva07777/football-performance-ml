
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# Load and preprocess data
df = pd.read_csv("players_fifa23.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Position'] = label_encoder.fit_transform(df['BestPosition'])

# Feature & target split for regression
X_reg = df[['Age', 'Overall', 'Potential', 'Position', 'IntReputation', 'WeakFoot', 'SkillMoves']]
y_reg = df['ValueEUR']

# Feature & target split for classification
X_clf = df[['Age', 'Overall', 'Potential', 'Position', 'IntReputation', 'WeakFoot', 'SkillMoves']]
y_clf = (df['Overall'] >= 85).astype(int)  # Binary classification: top-tier player or not

# Standard scaling
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)
X_clf_scaled = scaler.transform(X_clf)

# Split data
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42)

# Random Forest Regressor
reg = RandomForestRegressor()
reg_param = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
best_reg_rf = RandomizedSearchCV(reg, reg_param, n_iter=3, cv=3, n_jobs=-1)
best_reg_rf.fit(Xr_train, yr_train)

# Random Forest Classifier
clf = RandomForestClassifier()
clf_param = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
best_clf_rf = RandomizedSearchCV(clf, clf_param, n_iter=3, cv=3, n_jobs=-1)
best_clf_rf.fit(Xc_train, yc_train)

# Save models and preprocessors
joblib.dump(best_reg_rf, 'best_regressor.pkl')
joblib.dump(best_clf_rf, 'best_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
