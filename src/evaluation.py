from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

def evaluate(test_scaled):
    X_test = test_scaled.drop("Transported", axis=1)
    y_test = test_scaled["Transported"]

    model = joblib.load("artifacts/model.pkl")

    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec  = recall_score(y_test, preds, average="macro")

    print(f"Evaluation | Accuracy= {acc:.3f} | Precision= {prec:.3f} | Recall= {rec:.3f}")
    return acc, prec, rec

if __name__ == "__main__":
    evaluate(None, None)
