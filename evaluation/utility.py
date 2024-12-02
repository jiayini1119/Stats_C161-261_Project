from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler

def evaluate(method, X_train, y_train, X_test, y_test, random_state=123):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if method == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif method == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif method == "catboost":
        model = CatBoostClassifier(verbose=0, random_state=random_state)
    else:
        raise ValueError("Unsupported method. Choose from: 'logistic_regression', 'random_forest', 'catboost'.")
    
    model.fit(X_train_scaled, y_train)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    
    print(f"Evaluation metrics for {method}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="b", label=f"{method} ROC Curve (AUC = {metrics['ROC-AUC']:.2f})")
    plt.plot([0, 1], [0, 1], color="r", linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{method} ROC Curve")
    plt.legend()
    plt.show()
    
    return metrics