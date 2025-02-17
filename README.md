# Fraud Detection Project

Welcome to the Fraud Detection Project repository. This end-to-end machine learning pipeline is designed to build a robust, production-ready fraud detection system. Our pipeline includes exploratory data analysis, advanced feature engineering, data preprocessing, model training, evaluation, and deployment—all structured according to the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template.

---

## Project Overview

This project leverages insights from detailed exploratory data analysis (EDA) to design a feature engineering pipeline that transforms raw transactional data into meaningful features. Key highlights include:

- **Time-Based Transformations:** Extraction of date features (day, week, period of day) from transaction timestamps.
- **Missing Value Strategies:** Tailored imputation and outlier techniques.
- **Text Processing:** Engineering multiple features from product descriptions.
- **Custom Mapping:** Using pre-defined mappings for product categories (stored in `src/utils/constants.py`) and discretizing continuous variables based on EDA insights.
- **Modular Pipeline:** A clear separation of responsibilities enabling reproducible, scalable, and maintainable code.

---

## Repository Structure

```plaintext
cookiecutter-data-science/
├── README.md                 # This file
├── data/
│   ├── raw/                  # Original data files (e.g., dados.csv)
│   ├── interim/              # Intermediate data files
│   ├── processed/            # Final processed data ready for modeling
│   └── external/             # Additional data sources (if any)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb  # Data exploration & EDA
│   ├── 02_feature_engineering.ipynb        # Data cleaning & feature engineering
│   ├── 03_model_training.ipynb             # Model training routines
│   ├── 04_model_evaluation.ipynb           # Model evaluation and reporting
│   ├── 05_cutoff_optimization.ipynb        # Classification threshold optimization
│   └── 06_model_deployment.ipynb           # Model deployment (e.g., FastAPI)
│
├── src/
│   ├── data/
│   │   ├── load_data.py      # Data ingestion routines
│   │   ├── make_dataset.py   # Core feature engineering & cleaning
│   │   ├── preprocess.py     # Further preprocessing (scaling, imputation)
│   │   └── split_data.py     # Train/test splitting logic
│   ├── models/
│   │   ├── train_model.py    # Model training code (e.g., XGBoost)
│   │   ├── evaluate_model.py # Model evaluation metrics & reporting
│   │   ├── optimize_cutoff.py# Classification threshold optimization
│   │   └── deploy_model.py   # Model saving/loading for deployment
│   ├── deployment/
│   │   ├── fastapi_app.py    # RESTful API for model serving
│   │   └── model_loader.py   # Model loading utilities for production
│   └── utils/
│       ├── constants.py      # Pre-defined constants (e.g., j_mapping)
│       └── config.py         # Additional configuration utilities (if needed)
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
├── Dockerfile                # Containerization configuration
└── .gitignore                # Git ignore rules
```

---

## Getting Started

Follow these steps to run the pipeline:

1. **Data Ingestion:**
   - **File:** `src/data/load_data.py`
   - **Action:** Load raw data (e.g., `data/raw/dados.csv`).

2. **Feature Engineering:**
   - **Notebook:** `notebooks/02_feature_engineering.ipynb`
   - **Action:**  
     - Import the `FraudFeatureEngineer` from `src/data/make_dataset.py`.  
     - Apply cleaning and feature engineering (including date transformations, text processing, and custom mappings stored in `src/utils/constants.py`).  
     - Save the engineered dataset to `data/processed/processed_data.csv`.

3. **Preprocessing:**
   - **File:** `src/data/preprocess.py`
   - **Action:** Perform further preprocessing (e.g., median imputation and scaling) on the engineered dataset.

4. **Data Splitting:**
   - **File:** `src/data/split_data.py`
   - **Action:** Split the preprocessed data into training and testing sets for model training.

5. **Model Training & Evaluation:**
   - **Notebooks:** `notebooks/03_model_training.ipynb` and `notebooks/04_model_evaluation.ipynb`
   - **Action:** Train your model, tune hyperparameters, and evaluate its performance.

6. **Deployment:**
   - **Files:** `src/deployment/fastapi_app.py` and `src/models/deploy_model.py`
   - **Action:** Deploy the trained model using FastAPI for real-time predictions.

---

## Configuration & Constants

All pre-defined constants, including the mapping for variable "j", are stored in `src/utils/constants.py`. This approach ensures that any updates to your configuration are centralized, maintainable, and separated from the business logic.

**Example (`src/utils/constants.py`):**

```python
# src/utils/constants.py

PREDEFINED_J_MAPPING = {
    'cat1': 0,
    'cat2': 1,
    'cat3': 2,
    'cat4': 3,
    'cat5': 4,
    'cat6': 5,
    'cat7': 6,
    'cat8': 7,
    'cat9': 8
}
```

---

## Advanced Usage

- **Modular Design:** Each script is responsible for a distinct stage of the ML pipeline (data ingestion, feature engineering, preprocessing, splitting, training, evaluation, deployment).
- **Reproducibility:** Every step—from raw data ingestion to production-ready deployment—is fully version-controlled and reproducible.
- **Scalability:** The repository is designed to integrate seamlessly with containerization (Docker) and API-based model serving (FastAPI).

---

