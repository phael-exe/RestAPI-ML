import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_model.joblib')

def train():
    # Ensure directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    print("Done!")

if __name__ == "__main__":
    train()
