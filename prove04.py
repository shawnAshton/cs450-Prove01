from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math


def find_one_entropy(column, target):
    # print('col', type(column))
    # print('target', type(target))
    column = column.values

    # the following splits the array into categories for a specific attribute input
    category_split = np.unique(column, return_counts=True)
    # print((category_split))
    target_split = np.unique(target, return_counts=True)
    # the instances of any answer withing an attribute are stored into categories
    categories = category_split[0]  # number of "bins" for an attribute
    target_keys = target_split[0]   # number of "bins" for a target
    # print('\n', type(categories))
    # print('this->',categories, '\n', type(categories), '<-this')
#     from here I need to find the yes no ratio according to each category
#     create a dictionary to hold 0 in for each category
    category_dict = {}
    for i in categories:
        category_dict[i] = {}
        for j in target_keys:
            category_dict[i][j] = 0
    # print(category_dict)

#     loop through the column and targets and populate the counts for each attribute's possible answer...
#     aka the yeses and the no's down attributes
#     aka each "branches" possible answers.
#     print(len(column), ":", len(target))
    total_length = len(column)
    for col, atarget in zip(column, target):
        category_dict[col][atarget] += 1
    # print('CD', category_dict, "\n")


#     now I have all the "yes no's" i need to find the numbers going down each "branch"
#     put the counts into a 2 dimensional array
    attribute_target_counts = []
    i = 0
    # print(len(category_dict))
    for aCategory in category_dict:
        attribute_target_counts.append([])
        for aKey in category_dict:
            attribute_target_counts[i].append(category_dict[aCategory][aKey])
            # print(category_dict[aCategory][aKey])
        i += 1
        # print('\n')
#     verify that i put it into the 2d array...
#     print(attribute_target_counts)
    # for i in range(len(attribute_target_counts)):
    #     for j in range(len(attribute_target_counts[i])):
    #         print(attribute_target_counts[i][j])

#   find denominator (for entropy.. totals going down each branch)
#   of single attribute and make it the last item in the array.
    for i in range(len(attribute_target_counts)):
        mini_total = 0
        for j in range(len(attribute_target_counts[i])):
            mini_total += attribute_target_counts[i][j]
        attribute_target_counts[i].append(mini_total)
# ****at this point I can now use entropy and return the result***
#   find the inner entropy for each possible attribute's answer
    for i in range(len(attribute_target_counts)):
        entropy_inner = 0
        denominator = attribute_target_counts[i][-1]
        # print('DENOMINATOR', denominator)
        # print(attribute_target_counts[i])
        # print(type(attribute_target_counts[i]), '\n\n')
        for j in attribute_target_counts[i][:-1]:
             entropy_inner -= j / denominator * math.log2(j / denominator)
        attribute_target_counts[i].append(entropy_inner)

        # check the inner entropy. ITS GREAT! and now stored in last index of attribute_target_counts
        # the second to last index holds the count that went down that "branch"
        # print('NEW THINGIES', attribute_target_counts)

        # calculate and return the outer entropy from the
    entropy_for_this_column = 0
    for i in range(len(attribute_target_counts)):
        entropy_for_this_column += (attribute_target_counts[i][-2] / total_length) * attribute_target_counts[i][-1]
    # print('entropy_for_this_column', entropy_for_this_column)
    return entropy_for_this_column

class tree_Model:
    def __init__(self, modelData, modelTarget):
        self.modelData = modelData
        self.modelTarget = modelTarget

    def find_lowest_entropy(self):
        # print('THIS THING', type(self.modelData.iloc[:]))
        # find_one_entropy(self.modelData.iloc[:, 0], self.modelTarget)
        best_entropy = 1
        best_entropy_col = 0
        for i in range(len(self.modelData.columns)):
            col_entropy = find_one_entropy(self.modelData.iloc[:, i], self.modelTarget)
            if best_entropy > col_entropy:
                best_entropy = col_entropy
                best_entropy_col = i
        print('BEST ', best_entropy)
        print('BEST COL INDEX', best_entropy_col)
        return best_entropy_col

    def create_tree(self, rootColIndex):
        print(rootColIndex)



class tree_classifier:
    def fit(self,data_train,target_train):
        return tree_Model(data_train, target_train)



def readIrisDataSet():
    iris = datasets.load_iris()
    # Show the data (the attributes of each instance)
    # print(iris.data, 'DONE\n')
    scaler = StandardScaler()
    scaler.fit(iris.data)
    iris.data = scaler.transform(iris.data)
    # print(iris.data)
    return iris.data, iris.target

def get_autism_list():
    # The Autism dataset
    headers = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "age", "gender", "ethnicity", "jaundice",
               "autism", "country", "used", "result", "age_desc", "relation", "class"]
    filename = 'Autism-Adult-Data.csv'
    data = pd.read_csv(filename, names=headers, na_values="?")

    # axis tells us what to drop. the things in quotes finde the header of the column to drop
    data = data.drop(['used', 'age_desc', 'age', 'relation', 'ethnicity', 'country', 'result'], axis='columns')
    # instead of exploding the columns, just replace them with one or zero
    replace = {"gender": {"f": 0, "m": 1},
               "jaundice": {"no": 0, "yes": 1},
               "autism": {"no": 0, "yes": 1}}
    # replaces with any instance of genders options with numerical values.
    # inplace changes data
    data.replace(replace, inplace=True)
    data.dropna(how='any', inplace=True)
    # transform the things like ethnicity, class and country
    data = data.apply(LabelEncoder().fit_transform)
    # set the target as the column with header called class
    target = data['class']
    data = data.iloc[:, 0:-1]  # all rows, 0 to last column
    # print(target)

    # casts the types to floats
    data = data.astype(float, copy=True)
    # print(data)
    # print(target)

    return data, target


autism_data, autism_targets = get_autism_list()
# print('AD->', type(autism_data))
# print('AT=>', type(autism_targets))
model = tree_Model(autism_data, autism_targets)
model.create_tree(model.find_lowest_entropy())




