# Smart Logistics MLflow Project 🚚📦

This project presents a complete end-to-end MLOps pipeline using **MLflow** to manage the machine learning lifecycle for a real-world logistics dataset from **Delhivery**. The main goal is to build, tune, serve, and monitor a model that can **predict the actual delivery time** of shipments.

---

## 🎯 Objective

Build a predictive model for **actual delivery time** using features such as:

- Actual and OSRM-predicted distances
- Estimated vs real delivery durations
- Cutoff and segment factors

The project includes:

- Data preparation and feature selection  
- Experiment tracking with MLflow  
- Hyperparameter tuning with Hyperopt  
- Model versioning and deployment  
- REST API-based inference

---

## 📁 Project Structure

```
mlops-delhivery-project/
├── data/                   # Raw dataset (Delhivery.csv)
├── notebooks/              # Jupyter notebooks for EDA and training
│   └── model_training.ipynb
├── src/                    # Python scripts for final training & deployment
│   └── final_model.py
├── mlruns/                 # MLflow tracking folder (auto-generated)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/mlops-delhivery-project.git
cd mlops-delhivery-project
python -m venv venv
source venv/bin/activate      # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Run the notebook or script
```bash
jupyter notebook notebooks/model_training.ipynb
# OR
python src/final_model.py
```

### 2. Launch MLflow UI
```bash
mlflow ui
```
Open: [http://localhost:5000](http://localhost:5000)

### 3. Serve the best model as REST API
```bash
mlflow models serve -m "models:/Delhivery_RF_Model/1" --port 5010 --no-conda
```

### 4. Test the API
```bash
curl -X POST http://localhost:5010/invocations \
  -H "Content-Type: application/json" \
  -d '{
        "dataframe_split": {
          "columns": ["actual_distance_to_destination", "osrm_distance", "osrm_time", "cutoff_factor", "factor", "segment_factor", "segment_osrm_distance"],
          "data": [[143.7, 140.2, 122.3, 2, 1.01, 0.95, 135.2]]
        }
      }'
```

---

## 📊 Model Summary

| Model              | MAE   | R² Score | Deployment |
|-------------------|-------|----------|------------|
| RandomForest (tuned) | **1.81** | **0.9996** | ✅ via `mlflow.models.serve` |

---

## 🧪 MLflow Features Used

- `mlflow.start_run()` for experiment tracking  
- `mlflow.log_param()`, `mlflow.log_metric()`  
- `mlflow.sklearn.log_model()` for model artifacts  
- `Model Registry` with versioning  
- `mlflow models serve` for deployment  

---

## 🧠 Future Work

- Add real-time drift monitoring with cron jobs or streaming data  
- Integrate with a front-end dashboard  
- Try XGBoost and LightGBM for performance comparison  
- Add multi-metric optimization with Optuna

---

## ✨ Acknowledgements

Thanks to **Bahçeşehir University**, our instructor **Gökşin Bakır**, and the creators of **MLflow** for making this hands-on MLOps project possible.
