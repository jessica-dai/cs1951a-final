import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import sklearn
import scipy

#sklearn modules for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import linregress

import matplotlib.image as mpimg

database = "/Users/rebeccazuo/Desktop/cs1951a-final/all_data.db"

numberOfSamples = 1832
#method that performs kmeans clustering algorithm


connection = sqlite3.connect("all_data.db")
cursor = connection.cursor()
cursor.execute("SELECT edTech.dist_size, edTech.urb, edTech.region, edTech.totalComputers, edTech.training, edTech.integration FROM edTech;")
results = cursor.fetchall()
sizeA = []
district = []
region = []
training = []
integration = []
computers = []
dependentVariables = []
combinedA =[]
averageComputer = np.zeros(16)
averageTraining = np.zeros(16)
count1,count2,count3,count4,count5 = 0,0,0,0,0
count6,count7,count8,count9,count10 = 0,0,0,0,0
count11,count12,count13,count14,count15,count16  = 0,0,0,0,0,0
#TODO JOIN ON SOME ATTRIBUTES, ALSO TRY USING DECISION TREE AND OTHER CLASSIFIERS
for r in results:
    size = r[0]
    urban1 = r[1]
    region1 = r[2]
    computers.append(r[3])
    training.append(r[4])
    integration.append(r[5])
	#a combined column for everything
    if (urban1 == 1 and region1 == 1):
	    combined = 1
	    averageComputer[0] += r[3]
	    averageTraining[0] += r[4]
	    count1 +=1
    if(urban1 == 1 and region1 == 2):
	    combined = 2
	    averageComputer[1] += r[3]
	    averageTraining[1] += r[4]
	    count2 +=1
    if(urban1 == 1 and region1 == 3):
	    combined = 3
	    averageComputer[2] += r[3]
	    averageTraining[2] += r[4]
	    count3 +=1
    if(urban1 == 1 and region1 == 4):
	    combined = 4
	    averageComputer[3] += r[3]
	    averageTraining[3] += r[4]
	    count4 +=1
    if(urban1 == 2 and region1 == 1):
	    combined = 5
	    averageComputer[4] += r[3]
	    averageTraining[4] += r[4]
	    count5 +=1
    if(urban1 == 2 and region1 == 2):
	    combined = 6
	    averageComputer[5] += r[3]
	    averageTraining[5] += r[4]
	    count6 +=1
    if(urban1 == 2 and region1 == 3):
	    combined = 7
	    averageComputer[6] += r[3]
	    averageTraining[6] += r[4]
	    count7 +=1
    if(urban1 == 2 and region1 == 4):
	    combined = 8
	    averageComputer[7] += r[3]
	    averageTraining[7] += r[4]
	    count8 +=1
    if(urban1 == 3 and region1 == 1):
	    combined = 9
	    averageComputer[8] += r[3]
	    averageTraining[8] += r[4]
	    count9 +=1
    if(urban1 == 3 and region1 == 2):
	    combined = 10
	    averageComputer[9] += r[3]
	    averageTraining[9] += r[4]
	    count10 +=1
    if(urban1 == 3 and region1 == 3):
	    combined = 11
	    averageComputer[10] += r[3]
	    averageTraining[10] += r[4]
	    count11 +=1
    if(urban1 == 3 and region1 == 4):
	    combined = 12
	    averageComputer[11] += r[3]
	    averageTraining[11] += r[4]
	    count12 += 1
    if(urban1 == 4 and region1 == 1):
	    combined = 13
	    averageComputer[12] += r[3]
	    averageTraining[12] += r[4]
	    count13+=1
    if(urban1 == 4 and region1 == 2):
	    combined = 14
	    averageComputer[13] += r[3]
	    averageTraining[13] += r[4]
	    count14 += 1
    if(urban1 == 4 and region1 == 3):
	    combined = 15
	    averageComputer[14] += r[3]
	    averageTraining[14] += r[4]
	    count15 += 1
    if(urban1 == 4 and region1 == 4):
	    combined = 16
	    averageComputer[15] += r[3]
	    averageTraining[15] += r[4]
	    count16 += 1
    combinedA.append(combined)
    sizeA.append(size)
    averageComputer = np.array(averageComputer)
    counter = np.array([count1,count2,count3,count4,count5,count6,count7,count8,count9,count10,count11,count12,count13,count14,count15,count16])
res= np.zeros(len(averageComputer))
res1 = np.zeros(len(averageComputer))
for i in range(0,len(averageComputer)):
    res[i] = float(averageComputer[i])/float(counter[i])
    res1[i] = float(averageTraining[i])/float(counter[i])
objects = np.arange(1,17)


# make a bar graph
plt.bar(objects, res, align='center', alpha=0.5)
plt.xticks(objects)
plt.ylabel('Average Number of Computers')
plt.xlabel('Region 1-16')
plt.title('Average Computers per Region')
plt.show()


y = computers
x = combinedA

plt.scatter(x, y, c='BLUE', alpha=0.5)
plt.show()


#first perform clustering on size, district, region on training and integration => 4 different labels
#independent variable
dependentVariables = np.vstack((training,integration,computers)).reshape((2970,3))
print(np.shape(dependentVariables))

X = np.array(list(dependentVariables))
y = np.array(list(combinedA))

# Run a linear regression
def linearRegression(data,dependent):
    x = np.array(data)
    y = np.array(dependent)
    fit = np.polyfit(x, y,1)
    fit_fn = np.poly1d(fit)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    print(slope,intercept,r_value, p_value)

    plt.plot(x,y, 'yo', x, fit_fn(x), '--k')


# Doesn't seem to have a linear correlation
#linearRegression(size,computers)
#0.18698082606798605 3.132266376876023 0.07351926672251217 0.026076530423758643

#linearRegression(district,computers)
# -0.15786110611314022 4.027203983680139 -0.06019585266619635 0.06860367939016074

#linearRegression(region,computers)
#-0.20649747118057157 4.167950061511824 -0.0805293223620027 0.014773195696497538


#split the data
X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size = 0.1 ,
		random_state = 0,
		shuffle = True
	)

#need to also determine the number of clusters to use
def cluster(data, num_clusters = 16):
    k_means = KMeans(n_clusters=num_clusters, random_state=0)
    # TODO: Use k_means to cluster the documents and return the clusters and centers
    k_means.fit(data)
    return k_means.labels_, k_means.cluster_centers_

def plot_clusters(features, clusters, centers):
	"""
	Uses matplotlib to plot the clusters of documents

	Args:
		document_topics: a dictionary that maps document IDs to topics.
		clusters: the predicted cluster for each document.
		centers: the coordinates of the center for each cluster.
	"""
	topics = np.array([x for x in features])
	topics = np.reshape(topics, (2970,3))
	training = []
	integration = []
	computers = []

	for row in topics[:1500]:
		training.append(row[0])
		integration.append(row[1])
		computers.append(row[2])

	ax = plt.figure().add_subplot(111, projection='3d')
    #why is this not working?
	ax.scatter(list(training), list(integration), list(computers), c= 'blue', alpha=0.3) # Plot the documents
	#ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],c='black', alpha=1) # Plot the centers
	ax.set_xlabel("training")
	ax.set_ylabel("integration")
	ax.set_zlabel("number of computers")
	plt.title("Training vs. Integration vs. Number of Computers")
	# ax.set_xlim3d(0,4)
	# ax.set_ylim3d(0,4)
	# ax.set_zlim3d(0,40)
	plt.tight_layout()
	plt.show()

LABEL_COLOR_MAP = {1 : 'yellow',
                   2 : 'orange',
                   3 : 'Blue',
                   4 : 'Red',
                   5 : 'yellow',
                   6 : 'orange',
                   7 : 'Blue',
                   8 : 'Red',
                   9 : 'yellow',
                   10 : 'orange',
                   11 : 'Blue',
                   12 : 'Red',
                   13 : 'yellow',
                   14 : 'orange',
                   15 : 'Blue',
                   16 : 'Red'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in y]
def plot_actual(features,labels):

    topics = np.array([x for x in features])

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(topics[:, 0], topics[:, 1], topics[:, 2], c='Blue', alpha=0.3) # Plot the documents
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],c='black', alpha=1) # Plot the centers
    ax.set_xlabel("training")
    ax.set_ylabel("integration")
    ax.set_zlabel("number of computers")
    plt.tight_layout()
    plt.show()

#created clusters based on three dependent variables size, district and region
clusters, centers = cluster(X)
plot_clusters(X, clusters, centers)

#plot_actual(X, y)
#print(centers)

# Results : these demonstrate approximately where the four different labels for integration should be placed, see
# if there is any pattern with these results. Does not seem to demonstrate a relationship with all three.
# [[1.46464646 3.57575758 2.86868687]
#[3.36466165 2.73308271 1.42105263]
#[1.79807692 1.39423077 2.71153846]
#[3.6557377  2.70491803 3.61885246]]


#size on integration and training
# [[2.70810811 1.36756757 2.92432432 2.25945946 2.50810811]
#  [1.5326087  3.30434783 2.94021739 3.07065217 3.11956522]
#  [3.36312849 3.18994413 1.8547486  2.6424581  1.96648045]
#  [3.56521739 3.44293478 3.48913043 3.3423913  3.45380435]]

# dist on integration and training
# [[2.70810811 1.36756757 2.92432432 2.25945946 2.50810811]
#  [1.5326087  3.30434783 2.94021739 3.07065217 3.11956522]
#  [3.36312849 3.18994413 1.8547486  2.6424581  1.96648045]
#  [3.56521739 3.44293478 3.48913043 3.3423913  3.45380435]]

# region on integration and training
# [[2.70810811 1.36756757 2.92432432 2.25945946 2.50810811]
#  [1.5326087  3.30434783 2.94021739 3.07065217 3.11956522]
#  [3.36312849 3.18994413 1.8547486  2.6424581  1.96648045]
#  [3.56521739 3.44293478 3.48913043 3.3423913  3.45380435]]


cursor.close()
connection.close()
