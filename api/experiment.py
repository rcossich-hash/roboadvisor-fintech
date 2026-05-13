import json
import pickle
import base64
import requests
import numpy as np
import pandas as pd
from io import StringIO
from http.server import BaseHTTPRequestHandler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, classification_report,
    roc_curve
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
    "checking_account":   "Estado cuenta corriente",
    "duration":           "Duración del crédito (meses)",
    "credit_history":     "Historial crediticio",
    "purpose":            "Propósito del crédito",
    "credit_amount":      "Monto del crédito (DM)",
    "savings_account":    "Cuenta de ahorro/bonos",
    "employment":         "Tiempo de empleo actual",
    "installment_rate":   "Tasa de cuota (% ingreso)",
    "personal_status":    "Estado civil / sexo",
    "other_debtors":      "Otros deudores / garantes",
    "residence_since":    "Años en residencia actual",
    "property":           "Propiedad más valiosa",
    "age":                "Edad (años)",
    "other_installments": "Otros planes de cuotas",
    "housing":            "Tipo de vivienda",
    "existing_credits":   "Créditos existentes en este banco",
    "job":                "Tipo de trabajo",
    "dependents":         "N° personas a cargo",
    "telephone":          "Tiene teléfono registrado",
    "foreign_worker":     "Es trabajador extranjero"
}


def load_data():
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep=r"\s+", header=None, names=COLUMN_NAMES)
    df["target"] = df["target"].map({1: 0, 2: 1})
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y, encoders


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(length))

            n_estimators  = int(params.get("n_estimators", 100))
            max_depth      = int(params.get("max_depth", 8)) if params.get("max_depth") != "none" else None
            test_size      = float(params.get("test_size", 0.2))
            class_weight   = params.get("class_weight", "balanced")
            if class_weight == "none":
                class_weight = None

            X, y, encoders = load_data()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # ── Main RF with chosen params ─────────────────────────────────
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=10,
                random_state=42,
                class_weight=class_weight
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]

            acc  = float(accuracy_score(y_test, y_pred))
            auc  = float(roc_auc_score(y_test, y_prob))
            cm   = confusion_matrix(y_test, y_pred).tolist()
            rep  = classification_report(y_test, y_pred, output_dict=True)

            # ROC curve points (downsample to 40 pts for JSON size)
            fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
            idx = np.linspace(0, len(fpr_arr) - 1, min(40, len(fpr_arr))).astype(int)
            roc_points = [{"fpr": round(float(fpr_arr[i]), 4),
                           "tpr": round(float(tpr_arr[i]), 4)} for i in idx]

            # Feature importances
            fi = sorted(zip(X.columns.tolist(), rf.feature_importances_.tolist()),
                        key=lambda x: x[1], reverse=True)
            top_features = [
                {"feature": FEATURE_DESCRIPTIONS.get(f, f), "importance": round(v, 4)}
                for f, v in fi[:10]
            ]

            # ── Model comparison (fixed params, fast) ─────────────────────
            comparison_models = {
                "Random Forest":       rf,
                "Gradient Boosting":   GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
                "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
            }
            comparison = []
            for name, model in comparison_models.items():
                if name != "Random Forest":
                    model.fit(X_train, y_train)
                yp  = model.predict(X_test)
                ypr = model.predict_proba(X_test)[:, 1]
                comparison.append({
                    "name":     name,
                    "accuracy": round(float(accuracy_score(y_test, yp)) * 100, 1),
                    "auc":      round(float(roc_auc_score(y_test, ypr)) * 100, 1),
                    "is_main":  name == "Random Forest"
                })

            # ── Serialize model bundle ─────────────────────────────────────
            bundle = {
                "model": rf, "encoders": encoders,
                "feature_names": X.columns.tolist(),
                "feature_descriptions": FEATURE_DESCRIPTIONS
            }
            model_b64 = base64.b64encode(pickle.dumps(bundle)).decode("utf-8")

            payload = {
                "success": True,
                "params_used": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth if max_depth else "sin límite",
                    "test_size": test_size,
                    "class_weight": class_weight if class_weight else "sin ajuste"
                },
                "metrics": {
                    "accuracy": round(acc, 4),
                    "auc_roc":  round(auc, 4),
                    "confusion_matrix": cm,
                    "precision_approved": round(float(rep.get("0", {}).get("precision", 0)), 4),
                    "recall_approved":    round(float(rep.get("0", {}).get("recall", 0)), 4),
                    "precision_rejected": round(float(rep.get("1", {}).get("precision", 0)), 4),
                    "recall_rejected":    round(float(rep.get("1", {}).get("recall", 0)), 4),
                    "f1_approved":        round(float(rep.get("0", {}).get("f1-score", 0)), 4),
                    "f1_rejected":        round(float(rep.get("1", {}).get("f1-score", 0)), 4),
                    "train_samples": len(X_train),
                    "test_samples":  len(X_test),
                },
                "roc_curve":   roc_points,
                "top_features": top_features,
                "comparison":   comparison,
                "model_b64":    model_b64
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
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
