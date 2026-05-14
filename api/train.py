import json
import pickle
import base64
import tempfile
import os
import numpy as np
import pandas as pd
import requests
from http.server import BaseHTTPRequestHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/statlog/german/german.data"
)

COLUMN_NAMES = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker", "target"
]

FEATURE_DESCRIPTIONS = {
    "checking_account":    "Estado cuenta corriente",
    "duration":            "Duración del crédito (meses)",
    "credit_history":      "Historial crediticio",
    "purpose":             "Propósito del crédito",
    "credit_amount":       "Monto del crédito (DM)",
    "savings_account":     "Cuenta de ahorro/bonos",
    "employment":          "Tiempo de empleo actual",
    "installment_rate":    "Tasa de cuota (% ingreso)",
    "personal_status":     "Estado civil / sexo",
    "other_debtors":       "Otros deudores / garantes",
    "residence_since":     "Años en residencia actual",
    "property":            "Propiedad más valiosa",
    "age":                 "Edad (años)",
    "other_installments":  "Otros planes de cuotas",
    "housing":             "Tipo de vivienda",
    "existing_credits":    "Créditos existentes en este banco",
    "job":                 "Tipo de trabajo",
    "dependents":          "N° personas a cargo",
    "telephone":           "Tiene teléfono registrado",
    "foreign_worker":      "Es trabajador extranjero"
}


def load_and_prepare_data():
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(
        StringIO(resp.text),
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )
    # target: 1=good → 0 (approved), 2=bad → 1 (rejected)
    df["target"] = df["target"].map({1: 0, 2: 1})

    cat_cols = [c for c in df.columns if df[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y, encoders, df


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


def compute_metrics(clf, X_test, y_test, X):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = clf.feature_importances_
    feat_imp = sorted(
        zip(X.columns.tolist(), importances.tolist()),
        key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "accuracy": round(float(acc), 4),
        "auc_roc": round(float(auc), 4),
        "confusion_matrix": cm,
        "classification_report": {
            "approved": report.get("0", {}),
            "rejected": report.get("1", {})
        },
        "top_features": [
            {"feature": FEATURE_DESCRIPTIONS.get(f, f), "importance": round(v, 4)}
            for f, v in feat_imp
        ],
        "train_samples": int(len(X_test) * 4),
        "test_samples": int(len(X_test))
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            X, y, encoders, df = load_and_prepare_data()
            clf, X_train, X_test, y_train, y_test = train_model(X, y)
            metrics = compute_metrics(clf, X_test, y_test, X)

            # Store probs for threshold slider
            probs = clf.predict_proba(X_test).tolist()
            y_test_list = y_test.tolist()

            model_bundle = {
                "model": clf,
                "encoders": encoders,
                "feature_names": X.columns.tolist(),
                "feature_descriptions": FEATURE_DESCRIPTIONS
            }
            model_bytes = pickle.dumps(model_bundle)
            model_b64 = base64.b64encode(model_bytes).decode("utf-8")

            payload = {
                "success": True,
                "metrics": metrics,
                "model_b64": model_b64,
                "probs": probs,
                "y_test": y_test_list
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False,
                "error": str(e)
            }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
