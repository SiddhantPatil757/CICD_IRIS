from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train():
    data = load_iris()
    X,y = data.data,data.target

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    joblib.dump(model,"model.pkl")
    return model
if __name__ == "__main__":
    train()


# echo "# CICD_IRIS" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M main
# git remote add origin https://github.com/SiddhantPatil757/CICD_IRIS.git
# git push -u origin main