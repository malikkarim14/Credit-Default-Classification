from time import process_time

a = process_time()

# Load dataset
import pandas as pd

filename = 'UCI_Credit_Card.csv'
dataset = pd.read_csv(filename)

# head
print(dataset.head())

# descriptions
print(dataset.describe())

# rename columns
dataset = dataset.rename(columns={'default.payment.next.month': 'def_pay', 
                        'PAY_0': 'PAY_1'})

#Moving unlabeled category to fourth category
fil = (dataset.EDUCATION == 5) | (dataset.EDUCATION == 6) | (dataset.EDUCATION == 0)
dataset.loc[fil, 'EDUCATION'] = 4
print(dataset.EDUCATION.value_counts())

#Moving unlabeled category to third category
dataset.loc[dataset.MARRIAGE == 0, 'MARRIAGE'] = 3
print(dataset.MARRIAGE.value_counts())

#Change value to zero
fil = (dataset.PAY_1 == -2) | (dataset.PAY_1 == -1) | (dataset.PAY_1 == 0)
dataset.loc[fil, 'PAY_1'] = 0
fil = (dataset.PAY_2 == -2) | (dataset.PAY_2 == -1) | (dataset.PAY_2 == 0)
dataset.loc[fil, 'PAY_2'] = 0
fil = (dataset.PAY_3 == -2) | (dataset.PAY_3 == -1) | (dataset.PAY_3 == 0)
dataset.loc[fil, 'PAY_3'] = 0
fil = (dataset.PAY_4 == -2) | (dataset.PAY_4 == -1) | (dataset.PAY_4 == 0)
dataset.loc[fil, 'PAY_4'] = 0
fil = (dataset.PAY_5 == -2) | (dataset.PAY_5 == -1) | (dataset.PAY_5 == 0)
dataset.loc[fil, 'PAY_5'] = 0
fil = (dataset.PAY_6 == -2) | (dataset.PAY_6 == -1) | (dataset.PAY_6 == 0)
dataset.loc[fil, 'PAY_6'] = 0

#Remove "ID" column
dat = dataset.drop(columns = "ID")
feat_names = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","def_pay"]

#Mendefinisikan x dan y
X = dat.iloc[:, :-1].values
Y = dat.iloc[:, -1].values

#Rescale Data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# Split-out validation dataset
import sklearn.model_selection as model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(rescaledX, Y, test_size=0.2, random_state=35)

#Finding feature importances with Random Forest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=35)
clf.fit(X_train, Y_train)

for feature in zip(feat_names, clf.feature_importances_):
    print(feature)
    
#Visualisasi Pengaruh Fitur Terhadap Label

import matplotlib.pyplot as plt
import numpy as np

yv = clf.feature_importances_
fig, ax = plt.subplots() 
width = 0.4 # the width of the bars 
ind = np.arange(len(yv)) # the x locations for the groups
ax.barh(ind, yv, width, color='green')
ax.set_yticks(ind+width/10)
ax.set_yticklabels(feat_names, minor=False)
plt.title('Feature importance in RandomForest Classifier')
plt.xlabel('Relative importance')
plt.ylabel('feature') 
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)

#Memilih fitur yang paling berpengaruh
sfm = SelectFromModel(clf, threshold=0.043478)
sfm.fit(X_train, Y_train)

for feature_list_index in sfm.get_support(indices=True):
    print(feat_names[feature_list_index])
    
#Mendefinisikan x baru
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Spot-Check Algorithms
models = []
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_important_train, Y_train, cv=kfold,
    scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = " %s: %f " % (name, cv_results.mean())
    print(msg)

# Compare Algorithms

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset

classifier = SVC(kernel='rbf')
classifier.fit(X_important_train, Y_train)
y_pred = classifier.predict(X_important_test)
print(accuracy_score(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))

b = process_time()

Total_time = b-a