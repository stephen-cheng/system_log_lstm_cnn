import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation, metrics

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

# Separate our training data into test and training.
print("Separating data into 67% training set & 33% test set...")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=33)#add random state here...
print("Dataset separated.\n")

print("-------------------------- SVM, Kernel = Linear --------------------------")
#C_linear = [0.1, 3, 10, 100...]
C_linear = [1]
result_linear = []

print("C value chosen from: ", C_linear)
print("Calculating accuracy with K-fold...")

for C in C_linear:
    svc_linear = svm.SVC(kernel='linear', C=C)
    scores = cross_validation.cross_val_score(svc_linear, X_train, y_train, scoring='accuracy', cv=7)
    result_linear.append(scores.mean())

print("result:", result_linear)
#Result with different C are equal, so here choose C=1 directly as the best parameter.
best_param_linear = {"C": 1}


linear_test_score = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X_train, y_train).score(X_test, y_test)
#rbf_test_score = svm.SVC(kernel='rbf', C=best_param_rbf.get("C"), gamma=best_param_rbf.get("gamma")).fit(X_test, y_test).score(X_test, y_test)
#poly_test_score = svm.SVC(kernel='poly', C=best_param_poly.get("C"), degree=best_param_poly.get("degree")).fit(X_test, y_test).score(X_test, y_test)
linear_test = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X_train, y_train)

count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X_val)):
    count2 += 1
    classinrow = X_val[i]
    classinrow = np.array(X_val[i]).reshape(1,-1)#Need to do this so we can do predictions in sklearn
    #yval ma3roof, a is our prediction of yval. each xval is a set of features la one sample. 561. based on these feature it predicts activity. cool. check into validation sets.. play around with metrics and what not tomorrow, make sure u fully understand this code 100% !!!
    predicted = linear_test.predict(classinrow)#predict class la kul row.. each i is a row.
    actual = y_val[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1

print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)
print("Linear Kernel test score: ", linear_test_score)

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

#with cool visuals
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_CM(cm_normalized)

'''
#C_linear = [0.1, 1, 10, 100]
C_linear = [3]

result_linear = []

print "C value chosen from: ", C_linear
print "Calculating accuracy with K-fold..."

for C in C_linear:
    svc_linear = svm.SVC(kernel='linear', C=C)
    scores = cross_validation.cross_val_score(svc_linear, X_train, y_train, scoring='accuracy', cv=6)
    result_linear.append(scores.mean())

print "result:", result_linear
#Result with different C are equal, so here choose C=1 directly as the best parameter.
best_param_linear = {"C": 3}


#linear_test_score = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X_test, y_test).score(X_test, y_test)
#rbf_test_score = svm.SVC(kernel='rbf', C=best_param_rbf.get("C"), gamma=best_param_rbf.get("gamma")).fit(X_test, y_test).score(X_test, y_test)
#poly_test_score = svm.SVC(kernel='poly', C=best_param_poly.get("C"), degree=best_param_poly.get("degree")).fit(X_test, y_test).score(X_test, y_test)
linear_test = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X, y)
count1 = 0
count2 = 0
for i in xrange(X_fin.__len__()):
    count2 += 1
    a = linear_test.predict(X_fin[i])
    b = y_fin[i]

    if a == [b]:
        count1 += 1

print "Total cases: ", count2
print "Correct Prediction: ", count1
print "Correct Rate: ", float(count1) / count2


#print "Linear Kernel test score: ", linear_test_score
#print "RBF Kernel test score: ", rbf_test_score
#print "Poly Kernel test score: ", poly_test_score
'''

