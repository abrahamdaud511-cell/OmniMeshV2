import pickle
from sklearn.metrics import accuracy_score

def evaluate_ml_classic(model_path="./models/ml_classic.pkl", test_csv="./data/raw/test.csv"):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    pipeline = data['pipeline']
    import pandas as pd
    df = pd.read_csv(test_csv)
    X = df['text']
    y_true = df['label']
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate_ml_classic()
