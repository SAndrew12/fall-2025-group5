# Violent Extremist Organization Classification in Afghanistan, Using Machine Learning


**Year:** 2025  
**Semester:** Fall  

---

## Problem Statement
This project develops supervised machine learning models to classify unattributed or ambiguously attributed violent events in Afghanistan among three organizations: **Taliban**, **al-Qaeda (AQ)**, and **ISIS-K** using ACLED data for **2017–2021** (through the U.S. withdrawal).

### Project Goals
- Inform qualitative analysis with ML enabled feature importance testing and explainable AI.
- Test the feasibility of classifying attacks based on threat tactics/description (no spatial features). 
- Deploy UI to assist in easy classification attacks


### Planned Phases
1. **Preparation & EDA**: schema audit; class distribution; leakage checks; split design.  
2. **Feature Engineering**: textual (ACLED `notes`), spatiotemporal, operational signatures, and conflict-phase markers.  
3. **Training**: baselines (logistic regression, linear SVM), tree ensembles (RF/XGBoost), and text encoders (TF-IDF + linear, BERT).  
4. **Validation**: grouped/temporal CV; calibration; ablations; province/period generalization checks.  
5. **Classification & Reporting**: apply to unattributed/ambiguous records; quantify uncertainty; README and docs.  

---

## Welcome Video



https://github.com/user-attachments/assets/1261be98-40d2-4214-86a8-1cc2079c6587





##  Repository Structure

```
fall-2025-group5/
│
├── 📝 README.md
├── 📄 requirements.txt
├── 📄 project_structure.txt
│
├── 📁 demo/
│   ├── 🐍 demo.py
│   ├── 🐍 text_data_processing.py
│   └── 📊 unattributed_attacks_processed.csv
│
├── 📁 presenation/
│   └── presentation
│
├── 📁 reports/
│   └── 📄 final_paper_davanzo.docx
│
├── 📁 research_paper/
│   └── 📄 final_report_davanzo.docx
│
└── 📁 src/                                     # Main source code directory
    ├── 🐍 main.py                              # Main pipeline orchestrator with CLI
    ├── 🐍 main_experiments.py                  # Experimental pipeline runner
    ├── 🐍 data_loader.py                       # ACLED data loading and preprocessing
    ├── 🐍 feature_eng.py                       # Feature engineering (text, spatiotemporal, operational)
    ├── 🐍 train_test_split.py                  # Train/validation/test split logic
    ├── 🐍 under_over.py                        # Imbalance handling (undersampling/oversampling)
    ├── 🐍 models.py                            # Classical ML models (RF, XGBoost, MLP)
    ├── 🐍 non_classical.py                     # BERT-based classifier
    ├── 🐍 feature_fusion.py                    # BERT + manual features fusion model
    ├── 🐍 feature_fusion_experiments.py        # Feature fusion experimentation
    ├── 🐍 mislabed_data_exp.py                 # Mislabeled data experiments
    ├── 🐍 unattrib.py                          # Unattributed attack analysis
    ├── 🐍 vis.py                               # Visualization utilities
    ├── 🐍 xai_explanations.py                  # Explainable AI (SHAP, LIME)
    ├── 🐍 save_trained_models.py               # Model persistence utilities
    ├── 🐍 experiment_runner.py                 # Experiment execution script
    │
    ├── 📁 data/
    │   └── 📊 ACLED Data_2025-09-11.csv        # Primary ACLED dataset
    │
    ├── 📁 experiment_results/                  # Experimental training results
    │   ├── Results from extesntive Feature Fusion testing...
    │
    ├── 📁 results/                             # Model training results
    │   ├── Results by model...
    │
    ├── 📁 visualizations/                      # Generated plots and figures
    │   ├── visualiations...
    │
    └── 📁 xai_explanations/                    # Explainable AI outputs
        ├── XAI outputs...

```

---

##  Quick Start

### Installation

#### Prerequisites
- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package installer)
- **Git** (for cloning repository)
- **CUDA** (optional, for GPU acceleration with PyTorch)

#### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Capstone
```

2. **Set up Python virtual environment** (recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes depending on your internet connection, as it downloads PyTorch, transformers, and other large packages.

#### Troubleshooting Installation

**If you encounter PyTorch installation issues:**
```bash
# For CPU-only installation (no GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (if you have NVIDIA GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install remaining requirements
pip install -r requirements.txt
```

**If you encounter memory issues during installation:**
```bash
# Install packages one at a time
pip install numpy pandas scikit-learn
pip install torch transformers sentence-transformers
pip install matplotlib seaborn shap lime imbalanced-learn
```

**Verify installation:**
```bash
# Test imports
python -c "import torch; import transformers; import sklearn; print('All core packages imported successfully!')"

# Check if GPU is available (optional)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

4. **Prepare data**
Ensure `data/acled_afghanistan_2015_2021.csv` is in the data directory.
```bash
# Create data directory if it doesn't exist
mkdir -p data

# Place your ACLED data file in the data/ directory
# Expected path: data/acled_afghanistan_2015_2021.csv
```

### Running the Pipeline

The pipeline uses **command-line arguments** to select which models to run. This provides flexibility without editing code.

#### Basic Usage

```bash
# View all available options
python main.py --help

# Run only classical models (Random Forest, XGBoost, MLP)
python main.py --classical

# Run BERT model
python main.py --bert

# Run feature fusion model (BERT + manual features)
python main.py --fusion

# Run all models
python main.py --all
```

#### With Explainable AI (XAI)

```bash
# Classical models with comprehensive XAI analysis
python main.py --classical --xai

# Classical models with quick XAI (faster, fewer samples)
python main.py --classical --xai --xai-mode quick

# All models with XAI
python main.py --all --xai

# BERT and fusion with quick XAI
python main.py --bert --fusion --xai --xai-mode quick
```

#### Command-Line Arguments Reference

**Model Selection** (at least one required):
- `--classical` : Run classical ML models (Random Forest, XGBoost, MLP)
- `--bert` : Run BERT-based text classifier
- `--fusion` : Run feature fusion model (BERT + manual features)
- `--all` : Run all three model types

**XAI Configuration** (optional):
- `--xai` : Enable explainable AI analysis (SHAP, LIME)
- `--no-xai` : Explicitly disable XAI
- `--xai-mode {quick,comprehensive}` : XAI depth (default: comprehensive)

#### Common Workflows

```bash
# Quick demo - classical models only
python main.py --classical

# Full research pipeline - all models with comprehensive XAI
python main.py --all --xai

# Compare BERT vs Feature Fusion
python main.py --bert --fusion --xai

# Development/testing - quick XAI for faster iteration
python main.py --classical --xai --xai-mode quick

# Production - all models, no XAI for speed
python main.py --all --no-xai
```

### Expected Outputs

After running, you'll find:
- **Model Results**: `*_results.csv` files with performance metrics
- **Visualizations**: `./visualizations/` directory with confusion matrices, ROC curves, PR curves
- **Saved Models**: `./saved_models/` directory with trained model artifacts
- **XAI Explanations** (if `--xai` enabled): `./xai_explanations/` directory with SHAP plots and LIME reports

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
 

---

##  Methodology



### Class Imbalance Handling
Given the severe class imbalance (Taliban:ISIS-K ratio ~49:1), we employ:
- **Undersampling**: Reduce majority class in training
- **Focal Loss**: Down-weight well-classified examples (BERT models)
- **Batch Balancing**: Equal class representation per training batch
- **Threshold Optimization**: Find optimal decision threshold on validation set

### Models

#### 1. Classical ML (`--classical`)
- **Random Forest**: Ensemble of decision trees with feature importance
- **XGBoost**: Gradient boosting with regularization
- **MLP**: Multi-layer perceptron with dropout

**Features**: 300-dim sentence embeddings + engineered features (sub_event_type, fatalities, civilian_targeting, etc.)

#### 2. BERT (`--bert`)
- **Architecture**: BERT-base-uncased with classification head
- **Input**: Masked event descriptions (semantic masking of group/location names)
- **Training**: Focal loss, batch balancing, gradient accumulation
- **Configuration**: 320 max tokens, 12 epochs, early stopping

#### 3. Feature Fusion (`--fusion`)
- **Architecture**: BERT embeddings + 23 manually engineered features
- **Fusion**: Concatenate BERT [CLS] token (768-dim) with scaled manual features
- **Manual Features**: civilian_targeting, fatalities, violence_against_women, sub_event_type (one-hot)
- **Performance**: Best minority class recall (70.6% on ISIS-K)

### Semantic Masking
To prevent data leakage, we mask:
- **Group names**: [ORGANIZATION], [MILITANT_GROUP]
- **Location names**: [LOCATION], [PROVINCE], [DISTRICT]

This forces models to learn from tactics/patterns rather than memorizing entity names.

---

##  Evaluation

### Metrics
- **Primary**: Macro-F1 (balanced performance across classes)
- **Per-Class**: Precision, Recall, F1 (especially ISIS-K minority class)
- **Calibration**: Expected Calibration Error (ECE), Brier score
- **Threshold**: ROC-AUC, PR-AUC

### Explainable AI (XAI)
When `--xai` flag is enabled:
- **SHAP**: Feature importance and contribution analysis
- **LIME**: Local interpretable model explanations for individual predictions
- **Modes**:
  - `quick`: Faster analysis with fewer samples (15 features, 50 samples)
  - `comprehensive`: Deep analysis with more samples (20 features, 100 samples)

### Robustness Tests
- Temporal generalization (2020–2021 test set)
- Error analysis on minority class
- Confusion matrix analysis

---

##  Key Results

### Model Performance Summary
| Model | Test Accuracy | Macro-F1 | ISIS-K Recall | Taliban Recall |
|-------|--------------|----------|---------------|----------------|
| Random Forest | ~85% | ~0.75 | ~45% | ~95% |
| XGBoost | ~87% | ~0.78 | ~50% | ~96% |
| BERT | ~88% | ~0.80 | ~60% | ~94% |
| **Feature Fusion** | **~89%** | **~0.82** | **~71%** | **~95%** |

*Note: Feature Fusion achieves the best balance, particularly on minority class (ISIS-K) detection.*

### Key Findings
1. **Feature Fusion Superior**: Combining BERT's contextual understanding with engineered features yields best results
2. **Minority Class Challenge**: ISIS-K recall remains challenging due to 49:1 class imbalance
3. **Focal Loss Effective**: Batch balancing + focal loss significantly improves minority class detection
4. **XAI Insights**: SHAP analysis reveals `sub_event_type` and `civilian_targeting` as critical features

---

##  Technical Details

### Dependencies

The project requires Python 3.8+ and the following packages (all specified in `requirements.txt`):

#### Core Data Science
- **numpy** (>=1.24.0): Numerical computing
- **pandas** (>=2.0.0): Data manipulation and analysis
- **scikit-learn** (>=1.3.0): Classical ML algorithms and evaluation metrics

#### Deep Learning
- **torch** (>=2.0.0): PyTorch deep learning framework
- **transformers** (>=4.30.0): Hugging Face transformers (BERT)
- **sentence-transformers** (>=2.2.0): Sentence embeddings for classical models

#### Imbalanced Learning
- **imbalanced-learn** (>=0.11.0): Undersampling and oversampling techniques

#### Explainable AI
- **shap** (>=0.42.0): SHapley Additive exPlanations
- **lime** (>=0.2.0): Local Interpretable Model-agnostic Explanations

#### Visualization
- **matplotlib** (>=3.7.0): Plotting and visualization
- **seaborn** (>=0.12.0): Statistical data visualization

#### Utilities
- **tqdm** (>=4.65.0): Progress bars
- **joblib** (>=1.3.0): Model serialization and parallel processing

**Installation**: See the [Installation](#installation) section for detailed setup instructions.

**Full list**: See `requirements.txt` for complete package list with version constraints.

### Configuration
Model hyperparameters are defined in `main.py`:
- **BERT_CONFIG**: BERT-specific settings (learning rate, epochs, focal loss params)
- **FUSION_CONFIG**: Feature fusion settings (inherits BERT config + fusion-specific)
- **FUSION_MANUAL_FEATURES**: List of manual features to include in fusion

### Anti-Overfitting Strategies
1. **Nested Cross-Validation**: Separate hyperparameter tuning from evaluation
2. **Early Stopping**: Monitor validation loss (patience=3 epochs)
3. **Dropout**: 0.3 dropout rate in BERT classification head
4. **Gradient Accumulation**: Effective batch size of 32 (16 × 2 steps)
5. **Semantic Masking**: Prevent memorization of group/location names

---

##  Rationale
Accurate attribution enables trend analysis, risk mapping, and policy evaluation.  
This classifier aims to:  
- Provide **consistent** attribution with calibrated uncertainty.  
- Illuminate **distinct operational signatures** of Taliban, AQ, and ISIS-K.  
- Support analysts with **interpretable** evidence.  

### Research Gaps Addressed
- Lack of reproducible baselines for multi-class militant attribution.  
- Limited multimodal fusion of text + spatiotemporal signals.  
- Scarce evaluations of temporal and geographic generalization.  

---

##  Possible Issues

### Data & Labeling
- Class imbalance (ISIS-K severely underrepresented at ~2% of events)
- Ambiguity in `notes` and aliases
- Deduplication challenges

### Modeling
- Temporal drift (2019–2021 conflict dynamics changed)
- Geographic leakage potential between train/test
- Overfitting risk on minority class

### Ethics & Safety
- Outputs are **analytical aids**, not operational tools
- Respect ACLED license and data restrictions
- Models may perpetuate biases in original labeling

---

## 🔍 Research Contributions
- **Benchmark**: First open baselines for Taliban vs AQ vs ISIS-K attribution (2015–2021)
- **Fusion Architecture**: Novel combination of BERT + engineered features for conflict classification
- **Validation**: Transparent temporal & geographic testing methodology
- **Interpretability**: Comprehensive XAI analysis revealing tactical signatures
- **Imbalance Handling**: Effective strategies for extreme class imbalance (49:1 ratio)

---

##  Citation

If you use this work, please cite:
```
@misc{afghanistan_veo_classification_2025,
  title={Violent Extremist Organization Classification in Afghanistan Using Machine Learning},
  author={[Your Name]},
  year={2025},
  institution={George Washington University},
  howpublished={\url{https://github.com/amir-jafari/Capstone}}
}
```

---

## 👥 Project Info
- **Author**: Stephen Davanzo
- **Instructor**: Amir Jafari  
- **Instructor Email**: ajafari@gwu.edu  


---

##  License

This project uses ACLED data subject to their terms of use. Academic and non-commercial use only. Please review ACLED's data access policy before use.

---

## Acknowledgments

- **ACLED**: For providing comprehensive conflict event data
- **Hugging Face**: For transformers library and pre-trained BERT models
- **GWU Data Science Program**: For guidance and support
- **Dr. Amir Jafari**: For project supervision and domain expertise
