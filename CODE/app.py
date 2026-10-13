from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from functools import wraps
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "demo-secret-change-before-deployment")
USERNAME = os.environ.get("APP_USERNAME", "admin")
PASSWORD = os.environ.get("APP_PASSWORD", "crop123")
FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
MODEL = {'model': None, 'scaler': None, 'metrics': {}, 'classes': [], 'rows': 0, 'source': None, 'preview': []}
PREDICTION_HISTORY = []

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)
    return wrapped

def train_from_csv(path):
    df = pd.read_csv(path)
    required = FEATURES + ['label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV needs columns: " + ", ".join(required) + ". Missing: " + ", ".join(missing))
    df = df[required].dropna()
    if len(df) < 20 or df['label'].nunique() < 2:
        raise ValueError("Dataset needs at least 20 valid rows and 2 crop classes.")
    X, y = df[FEATURES], df['label'].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    MODEL.update({'model': model, 'scaler': scaler,
        'metrics': {'accuracy': round(accuracy_score(y_test, pred)*100, 2),
                    'precision': round(precision_score(y_test, pred, average='weighted', zero_division=0)*100, 2),
                    'recall': round(recall_score(y_test, pred, average='weighted', zero_division=0)*100, 2),
                    'f1': round(f1_score(y_test, pred, average='weighted', zero_division=0)*100, 2)},
        'classes': list(model.classes_), 'rows': len(df), 'source': os.path.basename(path), 'preview': df.head(12).to_dict(orient='records')})

@app.route('/login', methods=['GET','POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        if request.form.get('username','').strip() == USERNAME and request.form.get('password','') == PASSWORD:
            session.clear()
            session['logged_in'] = True
            session['username'] = USERNAME
            next_url = request.args.get('next','')
            if next_url.startswith('/') and not next_url.startswith('//'):
                return redirect(next_url)
            return redirect(url_for('dashboard'))
        flash('Incorrect username or password. Please try again.', 'error')
    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    return render_template('index.html', model_ready=MODEL['model'] is not None, metrics=MODEL['metrics'], rows=MODEL['rows'], classes=len(MODEL['classes']), source=MODEL['source'])

@app.route('/train', methods=['POST'])
@login_required
def train():
    upload = request.files.get('dataset')
    if not upload or not upload.filename:
        flash('Choose your Crop_recommendation.csv file first.', 'error')
        return redirect(url_for('dashboard'))
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', 'Crop_recommendation.csv')
    upload.save(path)
    try:
        train_from_csv(path)
        flash(f"Model trained successfully using {MODEL['rows']} valid records.", 'success')
    except Exception as e:
        flash(str(e), 'error')
    return redirect(url_for('dashboard'))

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    result, top = None, []
    values = dict(N=90, P=42, K=43, temperature=20.87, humidity=82, ph=6.5, rainfall=202.93)
    if request.method == 'POST':
        try:
            if MODEL['model'] is None:
                raise ValueError('Upload the CSV dataset and train the model on Dashboard first.')
            values = {k: float(request.form[k]) for k in FEATURES}
            if any(not np.isfinite(v) for v in values.values()):
                raise ValueError('Please enter valid numeric values.')
            row = pd.DataFrame([[values[k] for k in FEATURES]], columns=FEATURES)
            scaled = MODEL['scaler'].transform(row)
            result = str(MODEL['model'].predict(scaled)[0])
            probs = MODEL['model'].predict_proba(scaled)[0]
            inds = np.argsort(probs)[::-1][:5]
            top = [{'crop': str(MODEL['model'].classes_[i]), 'prob': round(float(probs[i])*100, 2)} for i in inds]
            PREDICTION_HISTORY.append({'crop': result, 'values': dict(values), 'probability': top[0]['prob'] if top else 0})
        except Exception as e:
            flash(str(e), 'error')
    return render_template('predict.html', values=values, result=result, top=top, model_ready=MODEL['model'] is not None)

@app.route('/insights')
@login_required
def insights():
    return render_template('insights.html', model_ready=MODEL['model'] is not None, rows=MODEL['rows'], classes=len(MODEL['classes']), source=MODEL['source'], preview=MODEL['preview'], metrics=MODEL['metrics'])

@app.route('/history')
@login_required
def history():
    return render_template('history.html', history=list(reversed(PREDICTION_HISTORY[-25:])))

@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html', model_ready=MODEL['model'] is not None, metrics=MODEL['metrics'], rows=MODEL['rows'], classes=len(MODEL['classes']), source=MODEL['source'])

if __name__ == '__main__':
    if os.path.exists('Crop_recommendation.csv'):
        try:
            train_from_csv('Crop_recommendation.csv')
            print('Model trained from Crop_recommendation.csv')
        except Exception as e:
            print('Dataset found but could not train:', e)
    print('Open http://127.0.0.1:5000')
    app.run(debug=True)
