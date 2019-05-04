import numpy as np
import argparse
import sqlite3
from sklearn.model_selection import train_test_split

# imports for models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

database = "/Users/rebeccazuo/Desktop/DataScienceFinal/edtech/new_data.db"

numberOfSamples = 1832
#method that performs kmeans clustering algorithm


connection = sqlite3.connect("new_data.db")
cursor = connection.cursor()
cursor.execute("SELECT edTech.dist_size, edTech.urb, edTech.region, edTech.totalComputers, edTech.training, edTech.integration FROM edTech;")
results = cursor.fetchall()
combinedA =[]


#TODO JOIN ON SOME ATTRIBUTES, ALSO TRY USING DECISION TREE AND OTHER CLASSIFIERS

for r in results:
    dist_size = r[0]
    urban1 = r[1]
    region1 = r[2]
    totalComputers = r[3]
    training = r[4]
    integration = r[5]
	#a combined column for everything
    if (urban1 == 1 and region1 == 1):
	    combined = 1
    if(urban1 == 1 and region1 == 2):
	    combined = 2
    if(urban1 == 1 and region1 == 3):
	    combined = 3
    if(urban1 == 1 and region1 == 4):
	    combined = 4
    if(urban1 == 2 and region1 == 1):
	    combined = 5
    if(urban1 == 2 and region1 == 2):
	    combined = 6
    if(urban1 == 2 and region1 == 3):
	    combined = 7
    if(urban1 == 2 and region1 == 4):
	    combined = 8
    if(urban1 == 3 and region1 == 1):
	    combined = 9
    if(urban1 == 3 and region1 == 2):
	    combined = 10
    if(urban1 == 3 and region1 == 3):
	    combined = 11
    if(urban1 == 3 and region1 == 4):
	    combined = 12
    if(urban1 == 4 and region1 == 1):
	    combined = 13
    if(urban1 == 4 and region1 == 2):
	    combined = 14
    if(urban1 == 4 and region1 == 3):
	    combined = 15
    if(urban1 == 4 and region1 == 4):
	    combined = 16
    temp = [totalComputers, training,integration,combined]
    combinedA.append(temp)
combinedA = np.array(combinedA)


# "preprocess" using train test split

def preprocess(dataset):
    X = dataset[:,:3]
    #y = dataset[:,-1]
    y = dataset[:,3]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def fit_model(model, X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = preprocess(combinedA)
    model = model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    model_params = model.get_params()
    return model, model_score, model_params

def main():

    X_train, X_test, y_train, y_test = preprocess(combinedA)

    models = [LogisticRegression(), BernoulliNB(), MLPClassifier(), LinearSVC(), DecisionTreeClassifier()]
    #models = [LogisticRegression(), LinearSVC()]
    results = np.array([])

    for model in models:
        _, score, _ = fit_model(model, X_train,X_test,y_train,y_test)
        print(model)
        print(score)

    #print(results)
    #logistic and svc work the best
    # best_model = np.argmax(results)
    #Logistic: 0.1572052401746725
    #Bernoulli: 0.1572052401746725
    #MLPClassifier: 0.15283842794759825
    #LinearSVC: 0.1572052401746725
    #DecisionTree:0.11353711790393013


# if __name__ == "__main__":
#     print("hello")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str)
#     args = parser.parse_args()

main()
