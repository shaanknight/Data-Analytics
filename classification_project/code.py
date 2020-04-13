#importing modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


#set options for pandas
pd.set_option('display.max_rows', 5)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def cleandata(data):
	"""
	Remove null,nan values along with irrelevant rows and cols
	"""

	#remove irrelevant rows and columns
	drop_col = [0,4,5,7,8,9,10,11,15,16,17,18,19]
	data = data.drop(data.columns[drop_col],axis=1)
	data = data.iloc[1:,]
	print(data.columns)

	#replace blank strings and empty cells with NaN
	data = data.replace(r'\s+',np.nan, regex=True)

	#remove records where magnitude=NaN
	data = data.dropna(subset=['MAGNITUDE'],how='any')

	#add values where NaN present
	data['YEAR'] = data['YEAR'].fillna(0)
	data['MONTH'] = data['MONTH'].fillna(0)
	data['DATE'] = data['DATE'].fillna(0)
	data['DEPTH (km)'] = data['DEPTH (km)'].fillna(-1)
	data['LAT (N)'] = data['LAT (N)'].fillna(-1)
	data['LONG (E)'] = data['LONG (E)'].fillna(-1)

	#convert data to float for comparing
	data = data.apply(pd.to_numeric)
	return data


def classification_model(data,C):
	"""
	Returns results of model based on given data and comparision value
	"""

	#setting labels for the classifier
	X,Y = [],[]
	tot = 0
	lt = 0
	rt = 0
	for index,rows in data.iterrows():
		tot = tot+1
		tmp = []
		c = 0
		for i in rows:
			c = c+1
			if c != 4:
				tmp.append(i)
		X.append(tmp)
		if rows['MAGNITUDE']<C:
			lt += 1
			Y.append(0)
		else:
			rt += 1
			Y.append(1)

	print('total dataset size : ',tot,'\nlower magnitude class size :',lt,'\nhigher magnitude class size :',rt)
	X = pd.DataFrame(X)
	Y = pd.DataFrame(Y)

	#split into test and train
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

	lt = 0
	rt = 0
	print(y_test)
	for i in len(y_test):
		if y_test[i] == 0:
			lt += 1
		else:
			rt += 1

	print(lt,rt)
	#Apply decision trees
	clf = DecisionTreeClassifier()
	clf = clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	
	feat_importance = clf.tree_.compute_feature_importances(normalize=True)
	print("feat importance = " + str(feat_importance))

	print("Max depth of the tree :",clf.tree_.max_depth)

	#print metrics
	print("\n-------")
	print("RESULTS")
	print("-------")
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	print("Error:",1-metrics.accuracy_score(y_test, y_pred))
	print("Recall:",metrics.recall_score(y_test, y_pred))
	print("Precision Score:",metrics.precision_score(y_test, y_pred))
	print("F1 Score:",metrics.f1_score(y_test, y_pred))
	print("Confusion Matrix:")
	print(metrics.confusion_matrix(y_test,y_pred))


#read data
data=pd.read_csv("./data.csv",low_memory=False,header=[0],encoding = 'unicode_escape')

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<58 and ord(i)>45)

data['LAT (N)'] = data['LAT (N)'].apply(remove_non_ascii)
data['LONG (E)'] = data['LONG (E)'].apply(remove_non_ascii)

#clean the data
data=cleandata(data)
print(data)

#Part A
print("\n----------------------------")
print("Results for combined dataset")
print("----------------------------\n")
classification_model(data,3)
classification_model(data,4)
classification_model(data,5)


#Part B
data1=data.loc[data['YEAR'] < 2000] 
data1=data1.loc[data1['YEAR'] >= 1990] 

data2=data.loc[data['YEAR'] < 2010] 
data2=data2.loc[data2['YEAR'] >= 2000] 

data3=data.loc[data['YEAR'] < 2020] 
data3=data3.loc[data3['YEAR'] >= 2010] 


print("\n-----------------------------")
print("Results for dataset 1990-2000")
print("-----------------------------\n")
classification_model(data1,3)
classification_model(data1,4)
classification_model(data1,5)

print("\n-----------------------------")
print("Results for dataset 2000-2010")
print("-----------------------------\n")
classification_model(data2,3)
classification_model(data2,4)
classification_model(data2,5)

print("\n-----------------------------")
print("Results for dataset 2010-2020")
print("-----------------------------\n")
classification_model(data3,3)
classification_model(data3,4)
classification_model(data3,5)