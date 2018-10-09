from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas
# from sklearn.cross_validation import KFold, cross_val_score
# from sklearn.model_selection import cross_val_predict

def loadCar():
    headers = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    car_data = pandas.read_csv(r"C:\Users\Shawn\Desktop\School fall 2018\car.data.csv",
                         names=headers, index_col=False)
    # changes the data from words to numbers
    car_data = car_data.apply(LabelEncoder().fit_transform)
    # gets the data and targets by splicing the dataFrame
    car_data_data = car_data.iloc[:, 1:]
    car_data_targets = car_data.iloc[:, 0]

    car_data_data = car_data_data.values
    car_data_targets = car_data_targets.values
    # print(car_data)
    # print(car_data_data)
    # print(car_data_targets)
    return car_data_data, car_data_targets


data, targets = loadCar()


# Applying K Fold Cross Validation and initializing classifier


# k_fold = KFold(len(y), n_folds=10, shuffle=True, random_state=7)
# classifier = KNeighborsClassifier(n_neighbors = optimal_k, metric = 'minkowski', p = optimal_p)               X
def load_autism():
    headersAutism = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "age", "gender",
               "ethnicity", "jundice", "autism", "country_of_res", "used_app_before", "result",
               "age_desc", "relation", "Class/ASD"]
    autism_data = pandas.read_csv(r"C:\Users\Shawn\Desktop\School fall 2018\Autism-Adult-Data.csv",
                         names=headersAutism, index_col=False, na_values="?")
    autism_data.dropna(how="any", inplace=True)

    autism_data = autism_data.apply(LabelEncoder().fit_transform)
    # gets the data and targets by splicing the dataFrame
    autism_data_data = autism_data.iloc[:, 1:]
    autism_data_targets = autism_data.iloc[:, 0]
    return autism_data_data, autism_data_targets


adata, atargets = load_autism()

def load_mpg():
    headersMPG = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration",
                  "model_year", "origin", "car_name"]
    mpg_data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    mpg_data = pandas.read_csv(mpg_data_link, names=headersMPG, delim_whitespace=True, na_values="?")
    mpg_data.dropna(how="any", inplace=True)
    print(mpg_data)




load_mpg()

























iris = datasets.load_iris()
# Show the data (the attributes of each instance)
scaler = StandardScaler()
scaler.fit(iris.data)
iris.data = scaler.transform(iris.data)


# assign train and testing variables using the train_test_split function
data_train, data_test, target_train, target_test = train_test_split\
    (iris.data,iris.target,test_size=0.3, random_state=39)

# predict targets
classifier = KNeighborsClassifier(n_neighbors=5)
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

# printing results
# print('TARGETS PREDICTED\n', targets_predicted)
# comes from train_test_split function
# print('TARGETS Actual\n', target_test)

# computing Match Percentage
match_percent = 0
for i in range(0,len(targets_predicted)):
    if targets_predicted[i] == target_test[i]:
        match_percent += 1
match_percent /= len(targets_predicted)
match_percent *= 100

# display match percent
# print("accuracy is {:.2f}%".format(match_percent))

