Violent Extremist Organization Classification in Afghanistan Using Machine Learning
Overview

This project develops supervised machine learning models to classify unattributed or ambiguously attributed violent events in Afghanistan among three organizations: Taliban, al-Qaeda (AQ), and ISIS-K using ACLED data for 2015–2021 (through the U.S. withdrawal).

The work builds a transparent, reproducible classification pipeline with rigorous evaluation and interpretability, aimed at providing analytical aids for conflict researchers and policy analysts.

Objectives

Build a reproducible pipeline for data ingestion, cleaning, feature engineering, modeling, and evaluation.

Compare text-only, spatiotemporal-only, and multimodal (text + structured) models.

Prioritize imbalanced-aware metrics (macro-F1, per-class PR-AUC) and model calibration for decision usefulness.

Deliver interpretable outputs with uncertainty estimates for real-world analyst use.

Dataset
Primary Dataset: ACLED (Armed Conflict Location & Event Data), Afghanistan 2015–2021

Fields used:

Event metadata: event_date, event_type, sub_event_type, fatalities

Actor info: actor1, actor2, assoc_actor*

Location: admin1/province, latitude, longitude

Narrative: notes (text descriptions of events)

Target Classes: Taliban, AQ, ISIS-K

Preprocessing & Labeling

Labels derived from actor* and assoc_actor* fields using conservative curation rules.

Deduplication of near-identical reports.

Exclusion of mixed/coalition cases from training; analyzed separately.

Engineered Features

Textual: TF-IDF features, keyphrase detection, contextual embeddings (BERT).

Spatiotemporal: province/district, geohash/H3 cells, event recency, Ramadan & seasonal markers, distance to prior hotspots.

Operational: event type, weapon proxies, lethality buckets, repeating patterns in provinces/time windows.

Rationale

Field data is often noisy and incomplete, leaving many ACLED events unattributed. A principled machine learning classifier can:

Provide consistent, transparent attribution hypotheses for unattributed events.

Reveal distinct operational signatures between Taliban, AQ, and ISIS-K.

Support analysis with interpretable evidence, not just black-box scores.

Research Gaps Addressed:

Lack of reproducible baselines for militant group attribution.

Limited multimodal fusion (text + spatial + operational) in conflict data.

Scarce temporal and geographic generalization evaluations.

Approach
Methodology

Train/validation split: 2015–2019

Test split: 2020–2021 (out-of-time generalization)

Grouped splits to prevent temporal & geographic leakage.

Models

Baselines: Majority class, last-seen cell class.

Linear: Logistic Regression, Linear SVM on TF-IDF.

Tree-based: Random Forest, XGBoost on structured + text-reduced features.

Neural (text): Fine-tuned BERT on notes.

Fusion: early fusion (feature concat), late fusion (stacked models).

Evaluation

Metrics: macro-F1, per-class F1, PR-AUC, calibration (ECE/Brier score).

Robustness: province hold-outs, temporal drift checks.

Interpretability: SHAP (tree/linear), attention visualizations (BERT).

Timeline

Week 1: Data acquisition, label policy, exploratory analysis.

Week 2: Cleaning, deduplication, leakage checks, initial features.

Week 3: Baseline + linear models.

Week 4: Tree-based ensembles; calibration.

Week 5: BERT text models; fusion experiments.

Week 6: Robustness tests, interpretability analysis, error breakdowns.

Week 7: Apply to unattributed events; draft results.

Week 8: Final documentation, ethical use statement, repo packaging.

Expected Contributions

Benchmark Dataset/Code: First open baselines for Taliban vs AQ vs ISIS-K attribution on ACLED.

Multimodal Fusion Evidence: Evaluation of text + spatial + operational feature integration.

Robust Validation Protocols: Transparent, temporally aware, and geography-aware testing.

Interpretability: Insights into tactics, geography, and temporal signatures of groups.

Possible Issues & Mitigation

Class Imbalance (AQ << Taliban, ISIS-K fluctuates): handle with weighted losses, PR-AUC focus, calibrated thresholds.

Label Ambiguity: strict curation; exclusion of contested records.

Temporal Drift: enforce time-based splits, conduct drift analysis.

Geographic Leakage: group-aware validation using H3/province.

Ethical Considerations: include model card; clarify that results are analytical aids only.

Project Info

Instructor / Advisor: Dr. Amir Jafari (ajafari@gwu.edu
)

Semester: Fall 2025

Repository: Capstone GitHub Repo
