import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Football Performance Prediction", layout="wide")
st.title("⚽ Football Player Performance Prediction Dashboard")

# Load the dataset
df = pd.read_csv("C:\Users\user\AppData\Local\Programs\Python\Python312\ML_Project\data\players_fifa23.csv")
st.subheader("Raw Data Preview")
st.dataframe(df.head())

# 1. Create custom Performance Score
attributes = ['Overall', 'Potential', 'ShootingTotal', 'PassingTotal', 'DribblingTotal', 'DefendingTotal', 'PaceTotal']
df = df.dropna(subset=attributes)
df['PerformanceScore'] = df[attributes].mean(axis=1)

# 2. Create binary classification label
df['WillPerformWell'] = (df['PerformanceScore'] >= 80).astype(int)

# 3. Preprocess
label_encoder = LabelEncoder()
df['Position'] = label_encoder.fit_transform(df['BestPosition'])

# Features and targets
features = ['Age', 'Overall', 'Potential', 'ShootingTotal', 'PassingTotal', 'DribblingTotal', 'DefendingTotal', 'PaceTotal', 'Position']
X = df[features]
y_reg = df['PerformanceScore']
y_clf = df['WillPerformWell']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_scaled, y_reg, y_clf, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning for Regression Model (RandomForestRegressor)
reg_param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

reg_rf = RandomForestRegressor(random_state=42)
rand_search_reg = RandomizedSearchCV(
    estimator=reg_rf,
    param_distributions=reg_param_dist,
    n_iter=10, random_state=42,
    cv=3, n_jobs=-1, scoring='neg_mean_squared_error'
)
rand_search_reg.fit(X_train, y_reg_train)
best_reg_rf = rand_search_reg.best_estimator_
preds_rf = best_reg_rf.predict(X_test)

# 5. Hyperparameter Tuning for Classification Model (RandomForestClassifier)
clf_param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

clf_rf = RandomForestClassifier(random_state=42)
rand_search_clf = RandomizedSearchCV(
    estimator=clf_rf,
    param_distributions=clf_param_dist,
    n_iter=10, random_state=42,
    cv=3, n_jobs=-1, scoring='accuracy'
)
rand_search_clf.fit(X_train, y_clf_train)
best_clf_rf = rand_search_clf.best_estimator_
preds_clf_rf = best_clf_rf.predict(X_test)

# 6. Regression Model Evaluation
st.subheader("📈 Regression Results")
st.write("#### Random Forest Regressor (Hyperparameter Tuned)")
st.write(f"MAE: {mean_absolute_error(y_reg_test, preds_rf):.2f}")
st.write(f"R2 Score: {r2_score(y_reg_test, preds_rf):.2f}")

# 7. Classification Model Evaluation
st.subheader("🤖 Classification Results")
st.write("#### Random Forest Classifier (Hyperparameter Tuned)")
st.write(f"Accuracy: {accuracy_score(y_clf_test, preds_clf_rf):.2f}")
st.text(classification_report(y_clf_test, preds_clf_rf))

# 8. Predict using player from the dataset
st.header("🎯 Predict from Existing Player")
selected_player = st.selectbox("Select a player", sorted(df['Name'].unique()))
player_row = df[df['Name'] == selected_player][features].iloc[0]
player_input = scaler.transform([player_row])
predicted_score_existing = best_reg_rf.predict(player_input)[0]
will_perform_existing = best_clf_rf.predict(player_input)[0]

st.write(f"**{selected_player}'s Predicted Performance Score:** {predicted_score_existing:.2f}")
st.write(f"**Will Perform Well?** {'Yes ✅' if will_perform_existing == 1 else 'No ❌'}")

estimated_goals = int((predicted_score_existing / 100) * 25)
st.write(f"**Estimated Goals in Season:** {estimated_goals}")

st.subheader("🔢 Confusion Matrix")
cm = confusion_matrix(y_clf_test, preds_clf_rf)
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# 9. Predict using manual input
st.header("🧠 Predict New Player Manually")
age = st.slider("Age", 16, 45, 25)
overall = st.slider("Overall", 40, 99, 70)
potential = st.slider("Potential", 40, 99, 75)
shooting = st.slider("ShootingTotal", 10, 100, 60)
passing = st.slider("PassingTotal", 10, 100, 60)
dribbling = st.slider("DribblingTotal", 10, 100, 60)
defending = st.slider("DefendingTotal", 10, 100, 60)
pace = st.slider("PaceTotal", 10, 100, 70)
position_label = st.selectbox("Position", label_encoder.classes_)
position_encoded = label_encoder.transform([position_label])[0]

input_data = np.array([[age, overall, potential, shooting, passing, dribbling, defending, pace, position_encoded]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Manually"):
    predicted_score = best_reg_rf.predict(input_scaled)[0]
    will_perform = best_clf_rf.predict(input_scaled)[0]
    goals = int((predicted_score / 100) * 25)

    st.write(f"### 🧠 Predicted Performance Score: {predicted_score:.2f}")
    st.write(f"### 🎯 Will Perform Well? {'Yes ✅' if will_perform == 1 else 'No ❌'}")
    st.write(f"### ⚽ Estimated Goals in Season: {goals}")

# 10. Wonderkids
st.header("🌟 Wonderkids")
wonderkids = df[(df['Age'] <= 21) & (df['Potential'] > df['Overall'] + 5)]
fig1 = px.scatter(wonderkids, x='Overall', y='Potential', color='Club', hover_data=['Name', 'Age'])
st.plotly_chart(fig1)
st.dataframe(wonderkids[['Name', 'Age', 'Overall', 'Potential', 'Club']].sort_values(by='Potential', ascending=False).head(10))

# 11. Underrated Gems
st.header("🔦 Underrated Gems")
underrated = df[(df['Overall'] >= 75) & (df['ValueEUR'] < df['ValueEUR'].median())]
fig2 = px.bar(underrated.sort_values(by='Overall', ascending=False).head(10), x='Name', y='Overall', color='Club')
st.plotly_chart(fig2)
st.dataframe(underrated[['Name', 'Overall', 'Potential', 'ValueEUR', 'Club']].sort_values(by='Overall', ascending=False).head(10))
