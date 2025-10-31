import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

# Feature selection
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Ensemble model (Voting Classifier)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

lr = LogisticRegression(max_iter=500)
svm = SVC(probability=True)
ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svm', svm)], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
