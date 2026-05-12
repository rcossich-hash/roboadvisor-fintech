import json
import pickle
import base64
import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler
from sklearn.preprocessing import LabelEncoder

# Categorical value maps matching German Credit encoding
CATEGORICAL_MAPS = {
    "checking_account": {
        "A11": "< 0 DM (negativo)",
        "A12": "0–200 DM",
        "A13": "> 200 DM / con salario asignado",
        "A14": "Sin cuenta corriente"
    },
    "credit_history": {
        "A30": "Sin créditos / todos pagados",
        "A31": "Todos pagados en este banco",
        "A32": "Créditos existentes al día",
        "A33": "Retrasos en el pasado",
        "A34": "Cuenta crítica / créditos en otros bancos"
    },
    "purpose": {
        "A40": "Auto (nuevo)",
        "A41": "Auto (usado)",
        "A42": "Muebles / equipos",
        "A43": "Radio / televisión",
        "A44": "Electrodomésticos",
        "A45": "Reparaciones",
        "A46": "Educación",
        "A48": "Reentrenamiento",
        "A49": "Negocios",
        "A410": "Otros"
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100–500 DM",
        "A63": "500–1000 DM",
        "A64": "> 1000 DM",
        "A65": "Desconocido / sin ahorro"
    },
    "employment": {
        "A71": "Desempleado",
        "A72": "< 1 año",
        "A73": "1–4 años",
        "A74": "4–7 años",
        "A75": "> 7 años"
    },
    "personal_status": {
        "A91": "Hombre divorciado / separado",
        "A92": "Mujer divorciada / separada / casada",
        "A93": "Hombre soltero",
        "A94": "Hombre casado / viudo",
        "A95": "Mujer soltera"
    },
    "other_debtors": {
        "A101": "Ninguno",
        "A102": "Co-solicitante",
        "A103": "Garante"
    },
    "property": {
        "A121": "Bien raíz",
        "A122": "Seguro de vida / ahorro",
        "A123": "Auto u otros bienes",
        "A124": "Desconocido / sin propiedad"
    },
    "other_installments": {
        "A141": "Banco",
        "A142": "Tiendas",
        "A143": "Ninguno"
    },
    "housing": {
        "A151": "Arriendo",
        "A152": "Propiedad propia",
        "A153": "Gratuita"
    },
    "job": {
        "A171": "Desempleado / no calificado no residente",
        "A172": "No calificado residente",
        "A173": "Empleado calificado / funcionario",
        "A174": "Alta dirección / independiente"
    },
    "telephone": {
        "A191": "No",
        "A192": "Sí (a nombre propio)"
    },
    "foreign_worker": {
        "A201": "Sí",
        "A202": "No"
    }
}


def encode_input(data, encoders, feature_names):
    row = {}
    for feat in feature_names:
        val = data.get(feat)
        if feat in encoders:
            le = encoders[feat]
            try:
                encoded = le.transform([str(val)])[0]
            except ValueError:
                encoded = 0
            row[feat] = int(encoded)
        else:
            try:
                row[feat] = float(val)
            except (TypeError, ValueError):
                row[feat] = 0.0
    return pd.DataFrame([row])[feature_names]


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            model_b64 = body.get("model_b64", "")
            applicant = body.get("applicant", {})

            if not model_b64:
                raise ValueError("model_b64 requerido")

            model_bundle = pickle.loads(base64.b64decode(model_b64))
            clf = model_bundle["model"]
            encoders = model_bundle["encoders"]
            feature_names = model_bundle["feature_names"]
            feature_descriptions = model_bundle.get("feature_descriptions", {})

            X_input = encode_input(applicant, encoders, feature_names)
            pred = clf.predict(X_input)[0]
            prob = clf.predict_proba(X_input)[0]

            decision = "APROBADO" if pred == 0 else "RECHAZADO"
            confidence = round(float(prob[pred]) * 100, 1)
            prob_approved = round(float(prob[0]) * 100, 1)
            prob_rejected = round(float(prob[1]) * 100, 1)

            # top contributing features via mean decrease impurity proxy
            importances = clf.feature_importances_
            top_feats = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            )[:5]

            result = {
                "success": True,
                "decision": decision,
                "confidence": confidence,
                "prob_approved": prob_approved,
                "prob_rejected": prob_rejected,
                "top_factors": [
                    {"feature": feature_descriptions.get(f, f), "importance": round(v, 4)}
                    for f, v in top_feats
                ]
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

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
