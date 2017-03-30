#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
print 'init visualization'
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from time import time

	### kNN ###
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 20)
clf.fit(features_train, labels_train)
print 'kNN score:', clf.score(features_test, labels_test)


	### random forest ###
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 1000)
t0 = time()
clf.fit(features_train, labels_train)
print 'random forest training time:', time() - t0
print 'random forest score:', clf.score(features_test, labels_test)

	### adaboost ###
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators = 100)
t0 = time()
clf.fit(features_train, labels_train)
print 'adaboost training time:', time() - t0
print 'adaboost score:', clf.score(features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    print 'failed to draw prettyPicture'
