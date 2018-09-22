from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
# print(iris.data)

# Show the target values (in numeric format) of each instance
# print(iris.target)

# Show the actual target names that correspond to each number
# print(iris.target_names)


# assign train and testing variables using the train_test_split function
data_train, data_test, target_train, target_test = train_test_split\
    (iris.data,iris.target,test_size=0.3,random_state=42)

# using the GaussianNB, we create a model that we are able to use to
# predict targets
classifier = GaussianNB()
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


# printing results
print('\n\nTARGETS PREDICTED Hard Coded\n', targets_predicted_hardCoded)
# comes from train_test_split function
print('TARGETS Actual\n', target_test)

# computing Match Percentage
match_percent = 0
for i in range(0,len(targets_predicted_hardCoded)):
    if targets_predicted_hardCoded[i] == target_test[i]:
        match_percent += 1
match_percent /= len(targets_predicted_hardCoded)
match_percent *= 100

# display match percent
print("accuracy is {:.2f}%".format(match_percent))