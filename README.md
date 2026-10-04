# Heart Disease Prediction System

Academic machine-learning project, modified from the instructor's **Titanic Passenger Survival Prediction** example.
Predicts whether **heart disease is present** from 4 inputs, using a Support Vector Classifier (SVC).

> Educational project only. This is a model prediction, **not** a medical diagnosis.

## Live App
`PASTE YOUR STREAMLIT URL HERE AFTER DEPLOYMENT`

## Dataset
UCI Heart Disease, Cleveland Clinic database (303 patients, no missing values in the chosen columns).

| Input | Meaning | Values |
|---|---|---|
| `sex` | Sex | 0 = Female, 1 = Male |
| `cp` | Chest pain type | 1 Typical angina, 2 Atypical angina, 3 Non-anginal, 4 Asymptomatic |
| `thalach` | Maximum heart rate achieved | number (bpm) |
| `exang` | Exercise-induced chest pain | 0 = No, 1 = Yes |

**Output:** `HeartDisease` — 0 = No Heart Disease, 1 = Heart Disease (original target 1–4 merged into 1).

## Model
`StandardScaler` + `SVC(gamma='auto')` in one pipeline. 80/20 train/test split. **Test accuracy: 77.05%** (47 of 61 test patients).
No LabelEncoder is used because all data is already numeric.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `Heart_Disease_Prediction.ipynb` | Training notebook (same steps as the Titanic notebook) |
| `heart-data.csv` | Dataset: 4 inputs + output |
| `heart_svc_model.pkl` | Trained model, loaded by `app.py` |
| `requirements.txt` | Libraries Streamlit Cloud installs |
| `runtime.txt` | Python version (3.11) |
| `.gitignore` | Files git should skip |

## Run locally
```
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
