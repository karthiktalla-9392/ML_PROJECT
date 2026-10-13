# 🌱 AgriSense: Crop Recommendation Using Multinomial Logistic Regression

A machine learning-based web application that recommends suitable crops based on soil nutrients and environmental conditions. Built using Python, Flask, HTML, CSS, and Scikit-learn.

## 📌 Project Overview

AgriSense uses Multinomial Logistic Regression to classify suitable crop categories based on seven input parameters. The web application provides an interactive interface for crop prediction, dataset exploration, prediction history, and model performance evaluation.

## 🎯 Objectives

- Predict suitable crop categories using machine learning.
- Analyze soil nutrients and environmental conditions.
- Integrate a trained ML model with a Flask web application.
- Provide an interactive dashboard and crop prediction interface.
- Evaluate model performance using classification metrics.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend and machine learning |
| Flask | Web application framework |
| Pandas | Dataset processing |
| NumPy | Numerical computations |
| Scikit-learn | Model training and evaluation |
| HTML5 | Frontend structure |
| CSS3 | Styling and layout |

## 🌾 Input Features

The model uses the following seven features:

1. Nitrogen (N)
2. Phosphorus (P)
3. Potassium (K)
4. Temperature
5. Humidity
6. Soil pH
7. Rainfall

**Target variable:** `label` — the crop category.

## ✨ Features

- **Login Page:** User authentication for accessing the application.
- **Dashboard:** Dataset overview and model training.
- **Crop Prediction:** Predicts a crop from seven input parameters.
- **Dataset Insights:** Displays dataset information and sample records.
- **Prediction History:** Shows recent predictions during the current server session.
- **Model Reports:** Displays accuracy, precision, recall, and F1-score.

## 🧠 Machine Learning Workflow

1. Load the crop recommendation dataset.
2. Separate input features and target labels.
3. Split the data into training and testing sets (80:20).
4. Standardize numerical features using `StandardScaler`.
5. Train the Multinomial Logistic Regression classifier.
6. Generate test predictions and evaluate performance.
7. Use the trained model to recommend crops through the web interface.

## 📂 Project Structure

```text
Crop_Recommendation_Web_App/
├── app.py
├── requirements.txt
├── README.md
├── Crop_recommendation.csv
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── index.html
│   ├── predict.html
│   ├── insights.html
│   ├── history.html
│   └── reports.html
└── static/
    └── style.css
```

*The dataset may also be uploaded through the dashboard. Verify that the filenames match your repository.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Crop_Recommendation_Web_App
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```cmd
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

### 6. Open the website

Navigate to:

```text
http://127.0.0.1:5000
```

## 🔐 Demo Login

If the default demo credentials are enabled in the application:

- **Username:** `admin`
- **Password:** `crop123`

These are demonstration credentials. Configure secure authentication before deploying the application publicly.

## 📊 Model Evaluation

The application evaluates its predictions using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Reliability of predicted classes.
- **Recall:** Proportion of actual class instances identified correctly.
- **F1-score:** Harmonic mean of precision and recall.

Actual metric values depend on the dataset and model training results.

## ⚠️ Limitations

- Recommendations depend on the training dataset.
- Predictions do not guarantee crop yield or agricultural success.
- Additional local soil, climate, and farming information may improve recommendations.
- Prediction history is temporary and resets when the server restarts.

## 🚀 Future Enhancements

- Persistent prediction history using a database.
- Integration with live weather data.
- Comparison with additional machine learning algorithms.
- Improved user authentication and account management.
- Interactive charts and downloadable reports.

## ✅ Conclusion

AgriSense demonstrates the application of machine learning in agricultural decision support. By combining Multinomial Logistic Regression with a Flask-based web interface, the system provides crop classification, dataset exploration, and model evaluation features.

---

**Project:** Crop Recommendation Using Multinomial Logistic Regression  
**Application:** AgriSense  
**Algorithm:** Multinomial Logistic Regression  
**Backend:** Python, Flask  
**Frontend:** HTML5, CSS3  
**ML Library:** Scikit-learn
