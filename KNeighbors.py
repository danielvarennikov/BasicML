import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')

# mark the missing data as an outlier
df.replace('?', -99999, inplace=True)

# get rid of the id column
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()

# train the model
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print("The accuracy is: " + str(accuracy))

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print('benign' if prediction == 2 else 'malignant')
