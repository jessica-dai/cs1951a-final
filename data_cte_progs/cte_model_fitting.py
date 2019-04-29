import numpy as np
import collections
from sklearn.model_selection import train_test_split

# imports for models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# import data
from cte_process import add_cross_class, region, u_type

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

def priors(dataset):
    y = dataset[:,-1]
    ctr = collections.Counter()
    for clss in y:
        ctr[clss] +=1
    return ctr

def main():

    results = []
    
    models = [LogisticRegression(max_iter=400, solver='lbfgs', multi_class='multinomial'), \
        BernoulliNB(), MLPClassifier(), LinearSVC(), DecisionTreeClassifier()]

    ans_sets = [add_cross_class(), region(), u_type()]

    for ans_set in ans_sets:

        print("priors:")
        print(priors(ans_set))

        X_train, X_test, y_train, y_test = preprocess(ans_set)

        set_results = []
        for model in models:
            _, score, _ = fit_model(model, X_train, X_test, y_train, y_test)
            set_results.append(score)

        print("model results:")
        print(set_results)

        results.append(set_results)
    # print(results)

    # best_model = np.argmax(results)

if __name__ == "__main__":
    print("hello")
    main()
    