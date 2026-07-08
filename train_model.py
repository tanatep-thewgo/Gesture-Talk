import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# =========================
# โหลด dataset
# =========================
data = pd.read_csv("hand_sign_data.csv", header=None)

print("Total samples:", len(data))

# แยก feature / label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Classes:", y.unique())

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# สร้างโมเดล KNN
# =========================
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# =========================
# Evaluate
# =========================
y_pred = model.predict(X_test)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print("\nAccuracy:", accuracy)

# =========================
# Save model
# =========================
joblib.dump(model, "sign_model.pkl")

print("\nModel saved as sign_model.pkl")