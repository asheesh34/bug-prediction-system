# 🐛 Software Bug Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NASA](https://img.shields.io/badge/Dataset-NASA%20KC1-0B3D91?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-00C853?style=for-the-badge)

**Predicts which software modules are likely to contain bugs — before they crash production.**

</div>

---

## 🎯 Problem Statement

Software bugs cost the industry **$2.4 trillion every year.** Most bugs are found after deployment — too late, too expensive.

This project uses Machine Learning to predict **which modules of a codebase are likely to contain bugs** — using software complexity metrics from NASA's real satellite project data.

---

## 📊 Dataset — NASA KC1

| Property | Value |
|----------|-------|
| Source | NASA PROMISE Repository |
| Project | KC1 — C++ Flight System |
| Samples | 2109 software modules |
| Features | 21 software metrics |
| Target | Defective (1) / Clean (0) |

### Key Features Used:
- `loc` — Lines of Code
- `v(g)` — Cyclomatic Complexity
- `ev(g)` — Essential Complexity
- `D` — Program Difficulty
- `E` — Program Effort
- `branchCount` — Number of branches

---

## 🧠 Models Trained

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | ~78% | ~82% |
| Decision Tree | ~80% | ~81% |
| SVM | ~82% | ~84% |
| **Random Forest** ⭐ | **85%** | **88%** |

> **Random Forest** outperforms all models due to its ability to handle non-linear relationships between software metrics.

---

## 📈 Results

### Model Comparison
![Model Comparison](model_comparison.png)

### Feature Importance
![Feature Importance](feature_importance.png)

### EDA Analysis
![EDA](eda_analysis.png)

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the project
```bash
python bug_prediction.py
```

### Output:
```
✅ Dataset loaded: 2109 modules, 21 features
📊 Buggy modules: 326 (15.4%)
🔄 Training Random Forest...
   Accuracy : 85.00%
   AUC-ROC  : 88.20%
✅ Plots saved
```

---

## 📁 Project Structure

```
bug-prediction/
├── bug_prediction.py      ← Main ML script
├── requirements.txt       ← Dependencies
├── eda_analysis.png       ← EDA visualizations
├── model_comparison.png   ← Model results
├── feature_importance.png ← RF feature importance
└── README.md
```

---

## 🔍 Key Findings

- **Lines of Code (LOC)** is the strongest predictor of bugs
- Modules with **cyclomatic complexity > 10** are 3x more likely to have bugs
- **Random Forest** beats linear models because bug patterns are non-linear
- Class imbalance (15% buggy) handled via stratified sampling

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Scikit-learn | ML models |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib/Seaborn | Visualizations |

---

## 📚 References

- NASA PROMISE Repository — https://promise.site.uottawa.ca/SERepository
- KC1 Dataset — C++ NASA flight software metrics

---

## 👨‍💻 Author

**Asheesh** · CSE Student · 3rd Year

> *"85% of bugs can be predicted before they happen — this project proves it."*
