
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation, metrics


# # load data

# In[2]:

labels = {
    '0':'file', '1':'network', '2':'service', '3':'database', '4':'communication', '5':'memory', '6':'driver', 
    '7':'system', '8':'application', '9':'io', '10':'others', '11':'security', '12':'disk', '13':'processor'}

fault_label = {
    '0':'file', '1':'network', '2':'service', '3':'database', '5':'memory', 
    '10':'others', '11':'security', '12':'disk', '13':'processor'}

X = []
y = []

print("Opening dataset...")
try:
    with open("data_msg_type/x.txt", 'rU') as f:
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
    with open("data_msg_type/y.txt", 'rU') as f:
        res = list(f)
        for line in res:
            y.append(int(line.strip("\n")[0]))
            
except:
    print("Error in reading the train set file.")
    exit()

print("Dataset loaded.")


# # split data

# In[3]:

X = np.array(X) #change to matrix
y = np.array(y) #change to matrix (sklearn models only accept matrices)

# Separate our training data into test and training.
print("Separating data into 80% training set & 20% test set...")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=33)#add random state here...
print("Dataset separated.\n")


# # svm-linear train

# In[7]:

print("-------------------------- SVM, Kernel = Linear --------------------------")
#C_linear = [0.1, 3, 10, 100...]
C_linear = [1]
result_linear = []

print("C value chosen from: ", C_linear)
print("Calculating accuracy with K-fold...")

for C in C_linear:
    svc_linear = svm.SVC(kernel='linear', C=C)
    scores = cross_validation.cross_val_score(
        svc_linear, X_train, y_train, scoring='accuracy', cv=9)
    result_linear.append(scores.mean())

print("result:", result_linear)


# # svm-linear test and predict

# In[8]:

# Result with different C are equal, so here choose C=1 directly as the best parameter.
best_param_linear = {"C": 1}
linear_test = svm.SVC(kernel='linear', C=best_param_linear.get("C")).fit(X_train, y_train)
linear_test_score = linear_test.score(X_test, y_test)

count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X_test)):
    count2 += 1
    classinrow = X_test[i]
    classinrow = np.array(X_test[i]).reshape(1,-1)
    predicted = linear_test.predict(classinrow)#predict class.
    actual = y_test[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1

print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)
print("Linear Kernel test score: ", linear_test_score)


# # svm-linear plot

# In[9]:

# cmap can be changed to many colors, (colormaps.Oranges,OrRd, etc)
def plot_CM(cm, title="Normalized Confusion Matrix", cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(fault_label))
    plt.xticks(tick_marks, fault_label.values(), rotation=90)
    plt.yticks(tick_marks, fault_label.values())
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
print(metrics.classification_report(
    actualist, predlist, target_names = list(fault_label.values())))
cm = metrics.confusion_matrix(actualist, predlist)
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_CM(cm_normalized)


# # svm-rbf train

# In[10]:

print("-------------------------- SVM, Kernel = RBF --------------------------")
#C_rbf = [0.1, 3, 10, 100...]
C_rbf = [1]
result_rbf = []

print("C value chosen from: ", C_rbf)
print("Calculating accuracy with K-fold...")

for C in C_rbf:
    svc_rbf = svm.SVC(kernel='rbf', C=C)
    scores = cross_validation.cross_val_score(
        svc_rbf, X_train, y_train, scoring='accuracy', cv=9)
    result_rbf.append(scores.mean())

print("result:", result_rbf)


# # svm-rbf test and predict

# In[11]:

# Result with different C are equal, so here choose C=1 directly as the best parameter.
best_param_rbf = {"C": 1}
rbf_test = svm.SVC(kernel='rbf', C=best_param_rbf.get("C")).fit(X_train, y_train)
rbf_test_score = rbf_test.score(X_test, y_test)

count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X_test)):
    count2 += 1
    classinrow = X_test[i]
    classinrow = np.array(X_test[i]).reshape(1,-1)
    predicted = rbf_test.predict(classinrow)#predict class.
    actual = y_test[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1

print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)
print("RBF Kernel test score: ", rbf_test_score)


# # svm-rbf plot

# In[12]:

# cmap can be changed to many colors, (colormaps.Oranges,OrRd, etc)
def plot_CM(cm, title="Normalized Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(fault_label))
    plt.xticks(tick_marks, fault_label.values(), rotation=90)
    plt.yticks(tick_marks, fault_label.values())
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
print(metrics.classification_report(
    actualist, predlist, target_names = list(fault_label.values())))
cm = metrics.confusion_matrix(actualist, predlist)
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_CM(cm_normalized)


# # svm-poly train

# In[13]:

print("-------------------------- SVM, Kernel = Poly --------------------------")
#C_poly = [0.1, 3, 10, 100...]
C_poly = [1]
result_poly = []

print("C value chosen from: ", C_poly)
print("Calculating accuracy with K-fold...")

for C in C_poly:
    svc_poly = svm.SVC(kernel='poly', C=C)
    scores = cross_validation.cross_val_score(
        svc_poly, X_train, y_train, scoring='accuracy', cv=9)
    result_poly.append(scores.mean())

print("result:", result_poly)


# # svm-poly test and predict

# In[14]:

# Result with different C are equal, so here choose C=1 directly as the best parameter.
best_param_poly = {"C": 1}
poly_test = svm.SVC(kernel='poly', C=best_param_poly.get("C"), 
                    #degree=best_param_poly.get("degree")
                   ).fit(X_train, y_train)
poly_test_score = poly_test.score(X_test, y_test)

count1 = 0
count2 = 0
actualist = []
predlist = []

for i in range(len(X_test)):
    count2 += 1
    classinrow = X_test[i]
    classinrow = np.array(X_test[i]).reshape(1,-1)
    predicted = poly_test.predict(classinrow)#predict class.
    actual = y_test[i]
    actualist.append(actual)
    predlist.append(predicted[0])
    if predicted == actual:
        count1 += 1

print("Total cases: ", count2)
print("Correct Prediction: ", count1)
print("Correct prediction rate: ", float(count1) / count2)
print("Poly Kernel test score: ", poly_test_score)


# # svm-poly plot

# In[16]:

# cmap can be changed to many colors, (colormaps.Oranges,OrRd, etc)
def plot_CM(cm, title="Normalized Confusion Matrix", cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(fault_label))
    plt.xticks(tick_marks, fault_label.values(), rotation=90)
    plt.yticks(tick_marks, fault_label.values())
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
    
print(metrics.classification_report(
    actualist, predlist, target_names = list(fault_label.values())))
cm = metrics.confusion_matrix(actualist, predlist)
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
plt.figure()
plot_CM(cm_normalized)

