import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from feature_schema import FEATURE_ORDER

df = pd.read_csv("dataset.csv")

X = df[FEATURE_ORDER]
y = df["severity"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print("Accuracy:",acc)

joblib.dump(model,"model.pkl")
print("Model saved as model.pkl")
