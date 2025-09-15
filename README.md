# Credit Card Fraud Detection 🚨💳

## 📌 Project Overview
This project focuses on detecting fraudulent transactions using the **Kaggle Credit Card Fraud Dataset**.  
The dataset is highly imbalanced (frauds are very rare), so special techniques like **SMOTE oversampling** and robust modeling are used.

---

## 📂 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Rows: 284,807  
- Features: 30 (including anonymized PCA components)  
- Target: `Class` (0 = legitimate, 1 = fraud)

---

## ⚙️ Steps
1. **Data Cleaning**  
   - Removed duplicates  
   - Handled outliers with IQR capping  

2. **Feature Engineering**  
   - Log transformation of skewed `Amount`  
   - Extracted cyclical features (`Hour`, `Day`, `Month`) from `Time`  
   - Applied cyclical encoding (sin/cos)  

3. **Preprocessing**  
   - Scaled `Amount` with `RobustScaler`  
   - Train-test split (80/20)  
   - Handled imbalance using **SMOTE**  

4. **Modeling**  
   Tuned and trained:  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - Support Vector Machine (SVM)  

5. **Evaluation**  
   - Classification Report (Precision, Recall, F1)  
   - ROC-AUC Score  
   - Confusion Matrix  

---

## 📊 Results

| Model                | Accuracy | Precision (Fraud=1) | Recall (Fraud=1) | F1-Score (Fraud=1) | ROC-AUC |
|-----------------------|----------|----------------------|------------------|---------------------|---------|
| Logistic Regression   | ...      | ...                  | ...              | ...                 | ...     |
| Random Forest         | ...      | ...                  | ...              | ...                 | ...     |
| XGBoost               | ...      | ...                  | ...              | ...                 | ...     |
| SVM                   | ...      | ...                  | ...              | ...                 | ...     |

*(Fill in your results after running the code)*

---

## 📈 Visuals
- Confusion Matrices for each model  
- ROC-AUC curves *(optional)*  

---

## 🛠️ Installation
Clone this repo and install dependencies:

