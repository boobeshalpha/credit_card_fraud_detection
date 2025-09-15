import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1️⃣ Load data
# -----------------------------
df = pd.read_csv("creditcard.csv")
print("Shape of dataset:", df.shape)
print(df.head())

# -----------------------------
# 2️⃣ Check missing values & duplicates
# -----------------------------
print(df.isnull().sum())
df = df.drop_duplicates()
print("Shape after dropping duplicates:", df.shape)

# -----------------------------
# 3️⃣ Handle outliers
# -----------------------------
for col in ['Amount', 'Time']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    Lower_Bound = Q1 - 1.5*IQR
    Upper_Bound = Q3 + 1.5*IQR
    df[col] = np.where(df[col] > Upper_Bound, Upper_Bound,
                       np.where(df[col] < Lower_Bound, Lower_Bound, df[col]))

# -----------------------------
# 4️⃣ Skewness transformation
# -----------------------------
df['Amount'] = np.log1p(df['Amount'])

# -----------------------------
# 5️⃣ Extract cyclical features from Time
# -----------------------------
df['Hour'] = (df['Time'] // 3600) % 24
df['Day'] = (df['Time'] // 86400) % 7
df['Month'] = (df['Time'] // 2592000) % 12

# Cyclical encoding
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Drop original Time and extracted columns
df = df.drop(['Time','Hour','Day','Month'], axis=1)

# -----------------------------
# 6️⃣ Scaling
# -----------------------------
scaler = RobustScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

# -----------------------------
# 7️⃣ Train-Test Split
# -----------------------------
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 8️⃣ Handle class imbalance with SMOTE
# -----------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:\n", pd.Series(y_train_smote).value_counts())

# -----------------------------
# 9️⃣ Hyperparameter Tuning
# -----------------------------
# Logistic Regression
log_params = {'C':[0.01,0.1,1,10],'penalty':['l2'],'solver':['lbfgs']}
log_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                        log_params, cv=3, scoring='roc_auc', n_jobs=-1)
log_grid.fit(X_train_smote, y_train_smote)
print("Best params Logistic Regression:", log_grid.best_params_)

# Random Forest
rf_params = {
    'n_estimators':[100,200],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2],
    'max_features':['sqrt','log2']
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                       rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_smote, y_train_smote)
print("Best params Random Forest:", rf_grid.best_params_)

# XGBoost
xgb_params = {
    'n_estimators':[100,200],
    'max_depth':[3,5],
    'learning_rate':[0.01,0.1],
    'subsample':[0.7,1],
    'colsample_bytree':[0.7,1]
}
xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    xgb_params, cv=3, scoring='roc_auc', n_jobs=-1
)
xgb_grid.fit(X_train_smote, y_train_smote)
print("Best params XGBoost:", xgb_grid.best_params_)

# SVM
svm_params = {
    'C':[0.1,1,10],
    'gamma':['scale',0.01,0.1],
    'kernel':['rbf','linear']
}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42),
                        svm_params, cv=3, scoring='roc_auc', n_jobs=-1)
svm_grid.fit(X_train_smote, y_train_smote)
print("Best params SVM:", svm_grid.best_params_)

# -----------------------------
# 10️⃣ Train final models with tuned params
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(**log_grid.best_params_, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(**rf_grid.best_params_, random_state=42),
    "XGBoost": XGBClassifier(**xgb_grid.best_params_, use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(**svm_grid.best_params_, probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {"Classification Report": report, "ROC AUC": roc_auc}
    print(f"\nClassification Report for {name}:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC for {name}: {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# -----------------------------
# 11️⃣ Summary table
# -----------------------------
summary = []
for model_name, metrics in results.items():
    report = metrics['Classification Report']
    summary.append({
        "Model": model_name,
        "Accuracy": report['accuracy'],
        "Precision (1)": report['1']['precision'],
        "Recall (1)": report['1']['recall'],
        "F1-score (1)": report['1']['f1-score'],
        "ROC-AUC": metrics['ROC AUC']
    })

summary_df = pd.DataFrame(summary)
print("\nModel Comparison:")
print(summary_df)
