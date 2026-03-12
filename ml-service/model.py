import joblib
import numpy as np

from feature_schema import FEATURE_ORDER, SEVERITY_MAP

class AccidentModel:

    def __init__(self):
        self.model = joblib.load("model.pkl")

    def predict(self,features):

        X = np.array([[features[f] for f in FEATURE_ORDER]])

        probs = self.model.predict_proba(X)[0]

        cls = probs.argmax()

        return {
            "severity": SEVERITY_MAP[cls],
            "confidence": float(probs[cls])
        }
