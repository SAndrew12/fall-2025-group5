# Violent Extremist Organization Classification in Afghanistan Using Machine Learning

**Version:** 1  
**Year:** 2025  
**Semester:** Fall  

---

## Objective
This project develops supervised machine learning models to classify unattributed or ambiguously attributed violent events in Afghanistan among three organizations: **Taliban**, **al-Qaeda (AQ)**, and **ISIS-K** using ACLED data for **2015–2021** (through the U.S. withdrawal).

### Goals
- Build a reproducible pipeline for data ingestion, cleaning, feature engineering, modeling, and evaluation.  
- Compare text-only, spatiotemporal-only, and multimodal (text + structured) models.  
- Prioritize imbalanced-aware metrics (macro-F1, per-class PR-AUC) and model calibration for decision usefulness.  
- Deliver a transparent classifier with interpretable outputs (feature importances/SHAP) and robust temporal validation.  

### Planned Phases
1. **Preparation & EDA**: schema audit; class distribution; leakage checks; split design (train:2015–2019, test:2020–2021).  
2. **Feature Engineering**: textual (ACLED `notes`), spatiotemporal, operational signatures, and conflict-phase markers.  
3. **Training**: baselines (logistic regression, linear SVM), tree ensembles (RF/XGBoost), and text encoders (TF-IDF + linear, BERT).  
4. **Validation**: grouped/temporal CV; calibration; ablations; province/period generalization checks.  
5. **Classification & Reporting**: apply to unattributed/ambiguous records; quantify uncertainty; README and docs.  

---

## 🗂 Dataset

**Primary Dataset:** ACLED (Armed Conflict Location & Event Data), Afghanistan 2015–2021  
- Fields: `event_date`, `event_type`, `sub_event_type`, `actor1`, `assoc_actor_1`, `actor2`, `assoc_actor_2`, `province`, `district`, `latitude`, `longitude`, `fatalities`, `notes`, `source`, `timestamp`.  
- Classes: **Taliban**, **AQ**, **ISIS-K** (derived from curated rules).  
- Prediction target: Organization label for unattributed/ambiguous events.  

### Preprocessing & Label Policy
- Conservative label curation from `actor*`/`assoc_actor*`.  
- Exclude mixed/coalition cases from training.  
- Deduplicate near-identical events.  
- Remove post-event metadata that may leak attribution.  

### Engineered Features
- **Textual**: TF-IDF (uni/bi-grams), keyphrase flags, BERT embeddings.  
- **Spatiotemporal**: province/district, H3 cells, distance to prior class centroids, month/season, Ramadan, conflict milestones.  
- **Operational**: `event_type`, weapons/IED proxies, target cues, lethality buckets, repeat-offense rates.  

---

## Rationale
Accurate attribution enables trend analysis, risk mapping, and policy evaluation.  
This classifier aims to:  
- Provide **consistent** attribution with calibrated uncertainty.  
- Illuminate **distinct operational signatures** of Taliban, AQ, and ISIS-K.  
- Support analysts with **interpretable** evidence.  

### Research Gaps
- Lack of reproducible baselines for multi-class militant attribution.  
- Limited multimodal fusion of text + spatiotemporal signals.  
- Scarce evaluations of temporal and geographic generalization.  

---

## Approach

### Methodology
- Reproducible pipeline (Make/Poetry/conda + DVC optional).  
- Careful **train/validation/test** splits:  
  - Train/Val: 2015–2019 (temporal CV with province grouping).  
  - Test: 2020–2021 (out-of-time).  

### Models
- **Baselines**: Majority, stratified, simple rules.  
- **Linear**: TF-IDF + Logistic Regression / Linear SVM.  
- **Tree-based**: Random Forest, XGBoost.  
- **Neural**: BERT-based text encoders.  
- **Fusion**: Early (concat embeddings + structured) and late (probability averaging/stacking).  

### Evaluation
- Metrics: macro-F1, per-class F1, PR-AUC, calibration (ECE/Brier), bootstrap CIs.  
- Robustness: province hold-outs, conflict-phase slices, missing text sensitivity.  
- Explainability: SHAP, attention/feature ablations, exemplar retrieval.  

### Deliverables
- GitHub README with methods/results.  
- Scripts: `prepare.py`, `train.py`, `evaluate.py`, `inference.py`.  
- Model cards + ethical use statement.  

---

## ⏳ Timeline
- **Week 1**: Acquire ACLED subset, compliance, EDA, label policy.  
- **Week 2**: Cleaning, deduping, leakage checks, first features.  
- **Week 3**: Baselines & linear models with imbalance handling.  
- **Week 4**: Tree ensembles, calibration.  
- **Week 5**: BERT models, fusion, hyperparameter sweeps.  
- **Week 6**: Robustness tests, SHAP, error analysis.  
- **Week 7**: Apply to unattributed events, draft figures/tables.  
- **Week 8**: Final write-up, model card, packaging & release.  

---

## Expected Number of Students
**1 student** with skills in:  
- **ML/NLP**: Scikit-learn, PyTorch/TF, text preprocessing, imbalance handling.  
- **Geospatial/temporal**: feature engineering, leakage avoidance, validation.  
- **Software**: packaging, experiment tracking, visualization.  

---

## Research Contributions
- **Benchmark**: First open baselines for Taliban vs AQ vs ISIS-K attribution (2015–2021).  
- **Fusion**: Evidence on combining text + spatiotemporal features.  
- **Validation**: Transparent temporal & geographic testing.  
- **Interpretability**: Insights into province/period/tactic signatures.  

---

## Possible Issues

### Data & Labeling
- Class imbalance (AQ underrepresented).  
- Ambiguity in `notes` and aliases.  
- Deduplication challenges.  

### Modeling
- Temporal drift (2019–2021).  
- Geographic leakage between train/test.  

### Ethics & Safety
- Outputs are **analytical aids**, not operational tools.  
- Respect ACLED license and data restrictions.  

---

## Project Info
- **Proposed by**: Dr. Amir Jafari  
- **Email**: ajafari@gwu.edu  
- **Instructor**: Amir Jafari  
- **Instructor Email**: ajafari@gwu.edu  
- **GitHub Repo**: [Capstone Repository](https://github.com/amir-jafari/Capstone)  
