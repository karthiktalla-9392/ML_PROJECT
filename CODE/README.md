# Crop Recommendation Web App

Frontend inspired by the provided reference image, adapted to your **Crop Recommendation Using Multinomial Logistic Regression** ML project. The notebook uses seven features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, with `label` as the crop target.

## Requirements
- Python 3.10+
- Your dataset CSV, typically `Crop_recommendation.csv`

## Windows setup
1. Extract this ZIP.
2. Open the extracted `Crop_Recommendation_Web_App` folder in VS Code.
3. Open Terminal in that folder.
4. Create a virtual environment (optional but recommended):
   ```cmd
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install packages:
   ```cmd
   python -m pip install -r requirements.txt
   ```
6. Start the website:
   ```cmd
   python app.py
   ```
7. Open this URL in Chrome: `http://127.0.0.1:5000`

## Train the ML model
- On the Dashboard, use **Upload & train model** to select `Crop_recommendation.csv`.
- The CSV must contain columns exactly named: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, `label`.
- The app trains the same StandardScaler + LogisticRegression approach used in your notebook and displays measured evaluation scores.
- You can alternatively place `Crop_recommendation.csv` beside `app.py` before starting; the app will try to train automatically.

## Pages
- Dashboard: dataset upload/training, overview and workflow
- Crop prediction: seven numeric inputs and top five predicted crop probabilities
- Model reports: test accuracy, precision, recall and F1-score

## Important
The dataset itself was not included in the uploaded notebook. Upload your actual CSV through the dashboard. The prediction is a model estimate and should be checked against local agricultural guidance.
\n## Login page\nDemo username: `admin`\nDemo password: `crop123`\n\nDashboard, prediction, reports and training require login. This is demo authentication, not production-ready security.\n
## Complete screens
Login, Dashboard, Crop Prediction, Dataset Insights, Prediction History, and Model Reports.
