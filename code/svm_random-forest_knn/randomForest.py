import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

activity_label = {'1': 'info',
                  '2': 'Critical',
                  '3': 'error',
                  '4': 'notice',
                  '5': 'warning',
                  '6': 'alert',
				  '7': 'emergency'}

# Open data set
X = []
y = []
X_val = [] #validation set features
y_val = [] #validation set target

print("Opening dataset...")
try:
    with open("data/msg_token_train.txt", 'rU') as f:
        res = list(f)
        for line in res:#each line is one sample, or row (can be viewed as 1*561 vector)
            line.strip("\n")
            features = line.split("\t")
            while features.__contains__(""):
                features.remove("")
        #print(len(features)) is 561, applied for each line in the file --> 10000*561 feature matrix!!
            for i in range(len(features)):
                features[i] = float(features[i])
            X.append(features)
         
    #read the classes from file and put them in list.      
    with open("data/msg_label_train.txt", 'rU') as f:
        res = list(f)
        for line in res:
            y.append(int(line.strip("\n")[0]))
            
except:
    print("Error in reading the train set file.")
    exit()
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
            X_val.append(features)
        
    with open("data/msg_label_test.txt", 'rU') as f:
        res = list(f)
        for line in res:
            y_val.append(int(line.strip("\n")[0]))
        f.close()
except:
    print("Error in reading the train set file.")
    exit()
print("Dataset opened.")


X = np.array(X) #change to matrix
y = np.array(y) #change to matrix (sklearn models only accept matrices)

print("Separating data into 67% training set & 33% test set...")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=33)#random split.
print("Dataset separated.\n")

print("---------------Random Forest---------------")
n_estimators_list = range(1, 21) #try from one to 21 estimators.
result_random_forests = [] #to be used later for comparing rf with different estimators.
max_score_rf = float("-inf") #just in case we get NaN
best_param_rf = None

for trees in n_estimators_list:
    print("Testing %d trees" % trees)
    rf_clf = RandomForestClassifier(n_estimators=trees, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_validation.cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv=7)
    result_random_forests.append(scores.mean())
    if scores.mean() > max_score_rf:
        max_score_rf = scores.mean()
        best_param_rf = {"n_estimators": trees}

rf_clf_test_score = RandomForestClassifier(n_estimators=best_param_rf.get("n_estimators"), max_depth=None,
                    min_samples_split=2, random_state=0).fit(X_test, y_test).score(X_test, y_test)
#print("Test set accuracy: ", rf_clf_test_score)

rf_clf = RandomForestClassifier(n_estimators=best_param_rf.get("n_estimators"), max_depth=None, 
         min_samples_split=2, random_state=0).fit(X, y)

#save trained model for future use.
joblib.dump(rf_clf,'data/rf_clf.pkl', compress=9)

count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X_val)):
    count2 += 1
    classinrow = X_val[i]
    classinrow = np.array(X_val[i]).reshape(1,-1)#Need to do this so we can do predictions in sklearn
    #yval ma3roof, a is our prediction of yval. each xval is a set of features la one sample. 561. based on these feature it predicts activity. cool. check into validation sets..
    predicted = rf_clf.predict(classinrow)#predict class for each row.. each i is a row.
    actual = y_val[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1
print()
print("Number of trees in forest: ", len(n_estimators_list))
print("Results: ", result_random_forests)
print("Best accuracy: ", max_score_rf)
print("Best parameter: ", best_param_rf)
print("Test set accuracy: ", rf_clf_test_score)

print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)

#cmap can be changed to many colors, (colormaps.Oranges,OrRd, etc)
def plot_CM(cm, title="Normalized Confusion Matrix", cmap=plt.cm.Greens):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(activity_label))
	plt.xticks(tick_marks, activity_label.values(), rotation=90)
	plt.yticks(tick_marks, activity_label.values())
	plt.tight_layout()
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.show()
	
print(metrics.classification_report(actualist, predlist, target_names = list(activity_label.values())))
cm = metrics.confusion_matrix(actualist, predlist)
print(cm)

#with cool visuals, this shows a normalized matrix as a separate figure.
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_CM(cm_normalized)

