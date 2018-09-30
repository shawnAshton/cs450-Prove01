from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
# print(iris.data, 'DONE\n')
scaler = StandardScaler()
scaler.fit(iris.data)
iris.data = scaler.transform(iris.data)
# print(iris.data, 'DONE\n')

# Show the target values (in numeric format) of each instance
# TARGET is a n dimensional array... but is is all one dimension
# print(iris.target)
# print(len(iris.target))

# Show the actual target names that correspond to each number
# print(iris.target_names)

# assign train and testing variables using the train_test_split function
data_train, data_test, target_train, target_test = train_test_split\
    (iris.data,iris.target,test_size=0.3,random_state=41)

# using the GaussianNB, we create a model that we are able to use to
# predict targets
classifier = KNeighborsClassifier(n_neighbors=4)
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# printing results
print('TARGETS PREDICTED\n', targets_predicted)
# comes from train_test_split function
print('TARGETS Actual\n', target_test)

# computing Match Percentage
match_percent = 0
for i in range(0,len(targets_predicted)):
    if targets_predicted[i] == target_test[i]:
        match_percent += 1
match_percent /= len(targets_predicted)
match_percent *= 100

# display match percent
print("accuracy is {:.2f}%".format(match_percent))

############################################################################################
#  Hard Coded stuff...
############################################################################################
# creating classes...
class HardCodedModel:
    def predict(self,data):
        predictions = []
        # loop through list and return zeros
        for i in range(0, len(data)):
            predictions.append(0)
        return predictions

class HardCodedClassifier:
    def fit(self,data_train,target_train):
        return HardCodedModel()

classifier_hardCoded = HardCodedClassifier()
model_HardCoded = classifier_hardCoded.fit(data_train, target_train)
targets_predicted_hardCoded = model_HardCoded.predict(data_test)


def mostFrequent(arr):
    # Sort the array
    n = len(arr)
    arr.sort()

    # find the max frequency using
    # linear traversal
    max_count = 1;
    res = arr[0];
    curr_count = 1

    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count += 1
        else:
            if (curr_count > max_count):
                max_count = curr_count
                res = arr[i - 1]
            curr_count = 1
    # If last element is most frequent
    if (curr_count > max_count):
        max_count = curr_count
        res = arr[n - 1]

    return res

############################################################################################
# knn Coded stuff...
############################################################################################
class knnCodedModel:
    def __init__(self, modelData, modelTarget, k):
        self.modelData = modelData
        self.modelTarget = modelTarget
        self.k = k
    # this uses a single data line and compares it to all of the model data
    def predictOne(self,oneDataTest):
        # get distance into array
        k = self.k
        distance = []
        for datum in self.modelData:
            distance.append(((datum[0] - oneDataTest[0]) ** 2) + ((datum[1] - oneDataTest[1]) ** 2) +
                            ((datum[2] - oneDataTest[2]) ** 2) + ((datum[3] - oneDataTest[3]) ** 2))
        # get dictionary made
        dictionary = dict(zip(distance, self.modelTarget))

        # find the KNN with the one data piece. this is the key that is found and shoved into lowest array
        lowest = []
        lowestValue = []
        for i in range(0, k):
            lowest.append(min(distance))
            lowestValue.append(dictionary[lowest[i]])
            distance.remove(min(distance))
        answer = mostFrequent(lowestValue)

        return answer

    # this takes in test data and tries to predict with it by looping through allTestData
    # I NEED A K
    def predictALL(self,allTestData):

        predictions = []
        # loop through list and return zeros
        for i in range(0, len(allTestData)):
            predictions.append(self.predictOne(allTestData[i]))
        return predictions

class knnCodedClassifier:
    def __init__(self,k):
        self.k = k
    def fit(self,data_train,target_train):

        return knnCodedModel(data_train, target_train, self.k)


classifier_knn = knnCodedClassifier(4)
model_knn = classifier_knn.fit(data_train, target_train)
targets_predicted_knn = model_knn.predictALL(data_test)


# printing results
# print('\n\nTARGETS PREDICTED Hard Coded\n', targets_predicted_hardCoded)
print('\n\nTARGETS PREDICTED KNN\n', targets_predicted_knn)
# comes from train_test_split function
print('TARGETS Actual\n', target_test)

# computing Match Percentage
match_percent = 0
for i in range(0,len(targets_predicted_knn)):
    if targets_predicted_knn[i] == target_test[i]:
        match_percent += 1
match_percent /= len(targets_predicted_knn)
match_percent *= 100

# display match percent
print("accuracy is {:.2f}%".format(match_percent))