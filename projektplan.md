# 🧠 Home Credit Default Risk – Projektplan (Portfolio-Version)

## 📌 Ziel
Ein strukturiertes, modernes Machine-Learning-Projekt zur Vorhersage von Kreditausfällen (Home Credit Default Risk), basierend auf dem bekannten Kaggle-Wettbewerb.  
Ziel ist eine **nachvollziehbare Modellpipeline**, **leistungsstarke Vorhersagemodelle** und der Einsatz von **branchenrelevanten Standards** wie Stacking, SHAP, Docker und Reproduzierbarkeit.

---

## 🚀 Aktueller Stand (Ausgangspunkt)

- ✅ Notebook-basiertes Rumprobieren mit ~0.78 ROC AUC
- ✅ Eigenes Feature Engineering (keine fertigen Kernels)
- ✅ LightGBM, Grid/Random Search, AUC-Drop-Feature-Auswahl
- ❌ Keine Strukturierung, Wiederverwendbarkeit oder klare Pipeline
- ❌ Kein README, kein Deployment, keine Visualisierung, kein Stack

---

## 🔧 Phase 1 – Strukturierung & Aufräumen (1–2 Tage)

### 1.1 Projektstruktur (2h)
- Erstelle standardisierte Ordnerstruktur:

project/
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── requirements.txt
├── README.md


### 1.2 Feature Engineering modularisieren (4–5h)
- `src/feature_engineering.py`
- Aufteilung nach Kategorien (z. B. Verhältnisfeatures, Flags, Summen)
- Saubere `make_features()`-Pipeline

### 1.3 Erste README.md (2–3h)
- Projektziel
- Wettbewerbsbeschreibung
- Aktueller Stand (Modell, Score)
- Überblick über den Code
- Geplante Schritte (aus diesem Plan)

---

## 📈 Phase 2 – Modell & Evaluation verbessern (3–5 Tage)

### 2.1 Evaluation erweitern (1 Tag)
- `src/evaluation.py`
- ROC AUC, ROC-Kurve, Confusion Matrix
- Threshold-Optimierung (Precision@k, F1)

### 2.2 Feature Selection systematisieren (1–2 Tage)
- AUC-Drop automatisieren
- SHAP Summary Plot & Feature Filtering
- Korrelationen prüfen (ggf. Feature-Drop)

### 2.3 Modelltraining robuster machen (1–2 Tage)
- `src/model_training.py`
- Cross-Validation (z. B. Stratified K-Fold)
- Vergleich: LightGBM, CatBoost, XGBoost

---

## 🌟 Phase 3 – Präsentation & Clean Notebook (2–3 Tage)

### 3.1 Finale Notebook-Version (1 Tag)
- `final_model.ipynb` mit Markdown-Kommentaren
- Kein Herumprobieren mehr, klarer Ablauf:
  - Daten laden → Features → Training → SHAP → Eval

### 3.2 SHAP Analyse (0.5–1 Tag)
- `src/shap_analysis.py`
- SHAP Summary Plot, Waterfall für einzelne Fälle
- Insights (z. B. „Top 5 Einflussfaktoren“)

### 3.3 README finalisieren (0.5 Tag)
- Projektbeschreibung
- Architekturdiagramm (optional)
- Finaler Score
- 1 Screenshot (z. B. SHAP-Plot)
- Learnings & Herausforderungen

---

## 🔝 Phase 4 – Advanced Features & Modeling (2–3 Tage)

### 4.1 Stacking Classifier (1 Tag)
- `sklearn.ensemble.StackingClassifier`
- Kombiniere LightGBM + CatBoost + XGBoost
- Logistic Regression als Meta-Layer

### 4.2 Meta-Features / Aggregationen (0.5–1 Tag)
- z. B. Credit-Summen pro Antragsteller (bureau.csv)
- Zahlungsverspätungen zählen
- Ziel: Mehr Kontext pro Kunde

### 4.3 Target Encoding (0.5 Tag)
- Smoothed target encoding für Kategoricals
- Achte auf Fold-Sicherheit (kein Leakage!)

---

## 🛠️ Phase 5 – Deployment & Reproduzierbarkeit (2–3 Tage)

### 5.1 CLI & Config-System (0.5 Tag)
- `train.py` → Kommandozeilen-Trainingsskript mit `argparse`
- `config.yaml` → Modellparameter, Pfade, CV-Typ

### 5.2 Dockerisierung (1 Tag)
- `Dockerfile` → Image mit Python, Abhängigkeiten, Skripten
- `docker run homecredit` startet `train.py`
- Optional: `Makefile` (train, eval, clean)

### 5.3 GitHub Actions (optional, 0.5 Tag)
- Linter + Testlauf bei jedem Push
- Zeigt Professionalität & DevOps-Grundverständnis

---

## ⏱️ Zeitplan-Vorschlag (realistisch: 7–14 Tage bei 1–3 h/Tag)

| Tag | Aufgabe |
|-----|--------|
| 1   | Struktur + Feature Engineering aufteilen |
| 2   | README + Evaluation |
| 3   | CV, Modellvergleich |
| 4   | SHAP + Feature Selection |
| 5   | Final Notebook bauen |
| 6   | Meta-Features + Target Encoding |
| 7   | Stacking einbauen |
| 8   | CLI + Config |
| 9   | Docker |
| 10  | README finalisieren, GitHub Actions (optional) |

---

## 📚 Optional – Weitere Ideen
- Streamlit Web-App für Prediction Demo
- Vergleich mit AutoML (z. B. H2O AutoML, AutoSklearn)
- Feature Store (z. B. `feast`, nur zur Übung)
- Save + Load Pipelines mit `joblib` oder `MLflow`

---

## ✅ Zielstatus: GitHub-Portfolio-Projekt

Am Ende hast du:
- Eine strukturierte, saubere ML-Pipeline
- Stacking-Modell mit SHAP-Analyse
- Dockerized CLI-Skript
- Beeindruckendes README
- Einsatz modernster Praktiken (CV, Encoding, Aggregationen, Reproducibility)
