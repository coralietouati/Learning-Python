import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier




###### CLASSIFICATION MODELS
mnist = fetch_mldata('MNIST original')
## other way to get the mist dataset
# from sklearn.datasets import load_digits
# mnist = load_digits()
mnist.target
y, X = mnist.target, mnist.data
X.shape
# look at the data
some_digit = X[36000] # line 36000 with 784 columns/feqtures
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# shuffle train set
suffle_index = np.random.permutation(60000)
X_train, y_train = X[suffle_index], y[suffle_index]

# Binary category
y_train_5 = (y_train == 5) # return True/false array
y_test_5 = (y_test == 5)

# stochastics Gradient Descent
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit]) # give a list ??

# perf measures
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy') # accuracy is not good measure for minority classes
# if we do a classifier that always predict that it is not a 5, the accurancy would be 90%.
# Solution is to a confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) # also do the k-cross validation but return the predictions
confusion_matrix(y_train_5,y_train_pred)# non-5:[True Neg, False neg] 5:[False pos, True pos]
# Precision: TP/(TP + FP) --> When it claims it is a 5, it is 70% correct
precision_score(y_train_5, y_train_pred)
# Recall: TP/(TP + FN) --> it detects 80% of the 5s
recall_score(y_train_5, y_train_pred)
# F1 scores combine precision and recall as the harmonic mean. You need a high recall and precision to have a high f1
f1_score(y_train_5, y_train_pred)
# Set the desired threshold
y_scores = cross_val_predict(sgd_clf,X_train,y_train_5, cv=3, method='decision_function') # return decision scores instead
precisions, recalls, threshold = precision_recall_curve(y_train_5,y_scores)
plt.plot(threshold, precisions[:-1], 'b--', label = "Precision")
plt.plot(threshold, recalls[:-1], 'b--', label = "Recall")
plt.xlabel('Threshold')
plt.ylim([0,1])

def plot_pr_curve(recalls, precisions):
    plt.plot(recalls,precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

plot_pr_curve(recalls,precisions)

threshold = 70000
y_train_pred_90 = (y_scores > 70000)
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)
f1_score(y_train_5, y_train_pred_90)

# ROC curve
fpr, tpr, threshold = roc_curve(y_train_5,y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

plot_roc_curve(fpr, tpr)

roc_auc_score(y_train_5,y_scores) # should be close to 1, 0.5 is a random classificater
# Compare two models: stochastics Gradient Descent to RandomForest
forest_clf = RandomForestClassifier(random_state=42)
# does not have the decision function but predict proba that returns an array with the proba that the given instance belong to the given class
y_probas_forest = cross_val_predict(forest_clf,X_train, y_train_5, cv=3, method="predict_proba")
forest = pd.DataFrame(y_probas_forest, columns=['Neg','Pos'])
forest['Answer'] = y_train_5
forest.loc[forest.Pos>0.6,'predict_60'] = True
forest.loc[forest.Pos<=0.6,'predict_60'] = False
forest.loc[forest.Pos>0.5,'predict_50'] = True
forest.loc[forest.Pos<=0.5,'predict_50'] = False

# avec cross val predict
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)

# The ROC needs scores not proba, score = proba of positive class
y_scores_forest = y_probas_forest[:,1] # pos
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)
plot_roc_curve(fpr, tpr, label='SGD')
plot_roc_curve(fpr_forest,tpr_forest, label='RandomForest')
plt.legend(loc='lower right')
plt.show()

# PR analsysis
precision_forest_manual_50 = precision_score(y_train_5,forest.predict_50.values)  # use .values to go from pandas col to array
precision_forest = precision_score(y_train_5,y_train_pred_forest) # has a thereforehold of 0.5
precision_forest_manual_60 = precision_score(y_train_5,forest.predict_60.values) # better precision

precisions_forest, recalls_forest, threshold_forest = precision_recall_curve(y_train_5,y_scores_forest) # problem with output
#plot_pr_curve(recalls_forest,precision_forest)




# SDG OvA
sgd_clf.fit(X_train,y_train)# use the OvA
sgd_clf.predict([some_digits])
some_digit_scores = sdg_clf.decision_function([some_digit]) # gives 10 scores, one for each class
np.argmax(some_digit_scores) # the highest scores is the class 5

# SDG OvO
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train,y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

# Random Forest
forest_clf.fit(X_train,y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])

# evulate with cross_val
cross_val_score(sgd_clf,X_train,y_train, cv=3, scoring='accuracy') # 84% accuracy. A random one would have 10% (1/10 chances)
# Improve accuracy with scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64)) # comapre with cahap2
cross_val_score ( sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
 
