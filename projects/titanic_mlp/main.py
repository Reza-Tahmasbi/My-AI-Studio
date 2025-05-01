from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from mlp import Mlp
import pandas as pd
import numpy as np

if "__main__" == __name__:
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/train.csv")
    y = df_train["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
    X = pd.get_dummies(df_train[features])
    X_test = pd.get_dummies(df_test[features])
    input_size = X.shape[1]
    hidden_size = 350
    output_size = y_onehot.shape[1]
    mlp = Mlp(input_size, hidden_size, output_size)
    mlp.train(X.values, y_onehot, epochs=1000, learning_rate=0.01)
    preds = mlp.predict(X.values)
    y_labels = np.argmax(y_onehot, axis=1)
    acc = accuracy_score(y_labels, preds)
    print(f"Training Accuracy: {acc:.4f}")