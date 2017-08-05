import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

activity_label = {'1': 'WALKING',
                  '2': 'WALKING_UPSTAIRS',
                  '3': 'WALKING_DOWNSTAIRS',
                  '4': 'SITTING',
                  '5': 'STANDING',
                  '6': 'LAYING'}
                  
X = []
y = []
print("Accessing data...")
try:
	#do the same for test sets.
    with open("data/msg_token_test.txt", 'rU') as f:
        res = list(f)
        for line in res:
            line.strip("\n")
            features = line.split("\t")
            while features.__contains__(""):
                features.remove("")
            for i in range(len(features)):
                features[i] = float(features[i])
            X.append(features)
        
    with open("data/msg_label_test.txt", 'rU') as f:
        res = list(f)
        for line in res:
            y.append(int(line.strip("\n")[0]))
        f.close()
except:
    print("Error in reading the train set file.")
    exit()
print("Dataset opened.")

X = np.array(X) #change to matrix
y = np.array(y)

rf_clf2 = joblib.load("data/rf_clf.pkl")

count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X)):
    count2 += 1
    classinrow = X[i]
    classinrow = np.array(X[i]).reshape(1,-1)
    predicted = rf_clf2.predict(classinrow)#predict class for each row.. each i is a row.
    actual = y[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1

print()
print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)
print(metrics.classification_report(actualist, predlist, target_names = list(activity_label.values())))
cm = metrics.confusion_matrix(actualist, predlist)
print(cm)


