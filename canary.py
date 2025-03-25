#Canary Release is a deployment strategy where a new version of an AI
#model is released gradually to a subset of users before a full rollout
#this reduces risk and allows monitoring of performance before 
#widespread adoption
#1. select a small percentage of traffic/users (5-10%) to test the new model
#2. monitor performance using key metrics (accuracy, latency,..etc.)
#3. gradually increase the percentage of traffic if the model performs well
#rollback immediately if issues arise. 

#Setting up a Canary Release System:
#1. deploy two AI model versions: old.pkl and new.Pkl
#2. use a flask API to serve predictions. 
#3. randomly assign 5% of requests to the new model initially
#4. log predictions and compare preformance
#5. gradually shift more traffic to the new model if it succeeds.abs

#Steps:
#1. train and save two model versions
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Train old model
old_model = RandomForestClassifier(n_estimators=10)
old_model.fit(X_train, y_train)
joblib.dump(old_model, 'old_model.joblib')


# Train new model (improved version)
new_model = RandomForestClassifier(n_estimators=50)  # More trees = better accuracy
new_model.fit(X_train, y_train)
joblib.dump(new_model, 'new_model.joblib')

print("Old and new models saved.")