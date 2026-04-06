import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("Workflow_CI_Model_Mohamad_Saiful_Rizal")


def load_data(data_dir=DATA_DIR):
    X_train = joblib.load(os.path.join(data_dir, "X_train.joblib"))
    X_test = joblib.load(os.path.join(data_dir, "X_test.joblib"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.joblib"))
    y_test = joblib.load(os.path.join(data_dir, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


def train_model():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1 Score :", f1)


if __name__ == "__main__":
    train_model()
