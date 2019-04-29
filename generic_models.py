import numpy as np
import argparse
from sklearn.model_selection import train_test_split

# imports for models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# "preprocess" using train test split

def preprocess(dataset):
    X = dataset[:-1]
    y = dataset[-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def fit_model(model):
    model = model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    model_params = model.get_params()

    return model, model_score, model_params

def main():
    X_train, X_test, y_train, y_test = preprocess(dataset)

    models = [LogisticRegression(), BernoulliNB(), MLPClassifier(), LinearSVC(), DecisionTreeClassifier()]

    results = np.array([])

    for model in models:
        _, score, _ = fit_model(model)
        results.append(score)
    
    print(results)

    # best_model = np.argmax(results)

# if __name__ == "__main__":
#     print("hello")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str)
#     args = parser.parse_args()

#     main()
    