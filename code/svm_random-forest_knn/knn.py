import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, metrics
from sklearn.neighbors import KNeighborsClassifier

activity_label = {'1': 'info',
                  '2': 'Critical',
                  '3': 'error',
                  '4': 'notice',
                  '5': 'warning',
                  '6': 'alert',
				  '7': 'emergency'}

# ############################# Open data set ###############################
X = []
y = []
X_val = [] #validation set features
y_val = [] #validation set target

print("Opening dataset...")
try:
    with open("data/msg_token_train.txt", 'rU') as f:
        res = list(f)
        for line in res:
            line.strip("\n")
            features = line.split("\t")
            while features.__contains__(""):
                features.remove("")
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
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=33)#add random state here...
print("Dataset separated.\n")

print("---------------K Nearest Neighbors----------------")
n_neighbors_list = range(1, 2, 1)
result_n_neighbors = []
max_score_knn = float("-inf")
best_param_knn = None

for n_neighbors in n_neighbors_list:
    print("Testing %d nearest neighbors" % n_neighbors)
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_validation.cross_val_score(knn_clf, X_train, y_train, scoring="accuracy", cv=7)
    result_n_neighbors.append(scores.mean())
    if scores.mean() > max_score_knn:
        max_score_knn = scores.mean()
        best_param_knn = {"n_neighbors": n_neighbors}

knn_clf_test_score = KNeighborsClassifier(best_param_knn.get("n_neighbors")).fit(X_test, y_test).score(X_test, y_test)
#print("Test set accuracy: ", knn_clf_test_score)


knn_clf = KNeighborsClassifier(best_param_knn.get("n_neighbors")).fit(X, y)
count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X_val)):
    count2 += 1
    classinrow = X_val[i]
    classinrow = np.array(X_val[i]).reshape(1,-1)
    predicted = knn_clf.predict(classinrow)
    actual = y_val[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1

print("Number of neighbors: ", len(n_neighbors_list))
print("Results: ", result_n_neighbors)
print("Best accuracy: ", max_score_knn)
print("Best parameter: ", best_param_knn)
print("Test set accuracy: ", knn_clf_test_score)

print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)

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

#with cool visuals
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_CM(cm_normalized)

