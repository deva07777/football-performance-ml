Football Player Performance Prediction
A machine learning pipeline that predicts football player performance using historical FIFA career mode data.  
Built with Python, Scikit-learn, TensorFlow, and data visualization tools.

 🚀 Project Overview
This project aims to predict football player performance metrics based on historical match and career data from FIFA (2015–2023).  
It uses regression and classification models to analyze features like goals, assists, ratings, and player attributes.
🧠 Features

- Data preprocessing and feature engineering from FIFA datasets  
- Multiple model evaluation (Random Forest, XGBoost, Linear Regression, ANN)  
- Achieved ~85% accuracy on player performance classification  
- Visualized model metrics and feature importance using Matplotlib/Seaborn  
- Exported trained models as `.pkl` for reuse  

🧩 Tech Stack
Languages & Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow  

 📊 Dataset
The dataset consists of FIFA player performance data (2015–2023), including metrics such as:
- Overall rating  
- Goals, assists, appearances  
- Age, position, and nationality  

  How to Run
```bash
git clone https://github.com/devadhatthanL/football-performance-ml.git
cd football-performance-ml
pip install -r requirements.txt
python src/train_models.py
