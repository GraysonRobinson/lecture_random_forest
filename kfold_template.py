from sklearn.model_selection import KFold
from sklearn import metrics

def run_kfold(machine, data, target, n, continuous): #continuous: if y variable is continuous or not
	kfold_object = KFold(n_splits=n)
	kfold_object.get_n_splits(data)


	all_return_values = []
	i=0
	for train_index, test_index in kfold_object.split(data):
		i=i+1
		data_train = data[train_index]
		target_train = target[train_index]
		data_test = data[test_index]
		target_test = target[test_index]

		machine.fit(data_train, target_train)
		prediction = machine = machine.predict(data_test)

		if (continuous==True):
			r2 = metrics.r2_score(target_test, prediction) #not the r2 of your OLS model (target on data, y on x), it is the r2 for the target_test and prediction (how well your prediction explains the prediction of your target test), used as a tool to test what you are doing
			all_return_values.append(r2)
		else:
			accuracy_score = metrics.accuracy_score(target_test, prediction)
			all_return_values.append(accuracy_score)
			confusion_matrix = metrics.confusion_matrix(target_test, prediction)
		return all_return_values
		

