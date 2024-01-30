import pandas

from sklearn import tree
#have to import tree for decision tree?

dataset = pandas.read_csv("temperature_data.csv")

print(dataset)

dataset = pandas.get_dummies(dataset)

print (dataset)

dataset.sample(frac=1)
#randomly draw sample from your dataset, if you put 0.5 will randomly draw from dataset for a dataset 1/2 size. If you put 1, it will essentially just sort the data

print(dataset)
#good practice to print a lot while coding in order to debug, can copy out after you write the entire program as it can slow down computation if you leave all of them 

#want to divide up into y and x variable, in dataset, the column name: (actual), is the y variable here, have to make it into a matrix
target = dataset["actual"].values
#Now need to turn it into a vector? or a matrix?
data = dataset.drop(["actual","level_0"], axis = 1)

#or can write it divided up like this:
#data = dataset.drop("actual", axis = 1)
#data = dataset.drop("level_0", axis = 1)

feature_list = data.columns
data = data.values

print(feature_list)
print(target)
print(data)


machine = tree.DecisionTreeClasifier(criterion="gini", max_depth=10)
#can  use entropy or gini here
#depth, how deep the trees are?, how many times it can chop?
return_values = kfold_template.run_kfold(machine, data, target, 4, True)
#True is meaning continuous here
print(return_values)

machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
machine.fit(data, target)
feature_importances_raw = machine.feature_importances_
print(feature_importances_raw)
print(feature_list)

#to put in same function?
feature_zip = zip(feature_list, feature_importances_raw)
print(feature_zip)
#can't read printed zip when you run it, have to -->
feature_importances = [ (feature, round(importance, 4))	for feature, importance in feature_zip]
feature_importances = sorted(feature_importances, key = lambda x: x[1] )
print(feature_importances)
[ print('{:13}: {}'.format(*feature_importance)) for feature_importance in feature_importances]
#what the first part means: for first items, give 13 character spaces(Tom got number through trial and error), then colon, space, then no restrictions


