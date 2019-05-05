import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from arts_ed_preprocess import get_rows

# imports for models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# "preprocess" using train test split

def preprocess(dataset):
    X = dataset[:,:-1]
    y = dataset[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def fit_model(model, X_train, X_test, y_train, y_test):
    model = model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    model_params = model.get_params()

    return model, model_score, model_params

def main():
    dataset = np.array(get_rows())
    print(dataset)
    X_train, X_test, y_train, y_test = preprocess(dataset)

    models = [LogisticRegression(), BernoulliNB(), MLPClassifier(), LinearSVC(), DecisionTreeClassifier()]

    results = []

    for model in models:
        _, score, _ = fit_model(model, X_train, X_test, y_train, y_test)
        results.append(score)
    
    print(results)

    # best_model = np.argmax(results)

if __name__ == "__main__":

    main()