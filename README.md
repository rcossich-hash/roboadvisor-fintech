# CréditoML — Roboadvisor Crediticio

Trabajo Final · Tecnología Financiera: Fintech · UC Chile

## ¿Qué hace?

Roboadvisor que evalúa solicitudes de crédito usando Machine Learning (Random Forest) entrenado sobre el dataset público **Statlog German Credit** (UCI Repository, Hofmann 1994).

- **Pestaña "Entrenar modelo"**: descarga el dataset desde UCI, entrena el modelo y muestra métricas (Accuracy, AUC-ROC, Matriz de Confusión, Feature Importance)
- **Pestaña "Evaluar solicitud"**: ingresa datos de un solicitante y obtiene la decisión automatizada (APROBADO / RECHAZADO) con probabilidades

## Stack

- **Frontend**: HTML/CSS/JS estático
- **Backend**: Python (Vercel Serverless Functions)
- **ML**: scikit-learn — RandomForestClassifier
- **Dataset**: [UCI Statlog German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) — CC BY 4.0

## Deploy en Vercel

1. Clona o sube este repositorio a GitHub
2. Conecta el repositorio en [vercel.com](https://vercel.com)
3. Vercel detecta automáticamente la configuración (`vercel.json`)
4. Deploy → listo

No requiere variables de entorno ni base de datos.

## Estructura

```
roboadvisor/
├── api/
│   ├── train.py       # Descarga dataset, entrena RF, devuelve métricas + modelo
│   └── predict.py     # Recibe datos del solicitante, devuelve predicción
├── public/
│   └── index.html     # Frontend completo
├── requirements.txt   # Dependencias Python
└── vercel.json        # Configuración de rutas
```

## Dataset

**Statlog (German Credit Data)** — H. Hofmann, Universität Hamburg (1994)  
Fuente: UCI Machine Learning Repository · Licencia: CC BY 4.0  
1.000 registros · 20 variables · Variable objetivo: buen riesgo (700) / mal riesgo (300)

## Modelo

**Random Forest Classifier** — scikit-learn  
- 100 árboles de decisión (`n_estimators=100`)  
- Profundidad máxima 8 (`max_depth=8`)  
- Split mínimo 10 muestras (`min_samples_split=10`)  
- Pesos balanceados para compensar desbalance de clases (`class_weight="balanced"`)  
- División 80/20 train-test con estratificación  

Métricas esperadas: Accuracy ~75–80% · AUC-ROC ~78–83%
