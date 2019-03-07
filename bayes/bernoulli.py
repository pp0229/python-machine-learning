import numpy as np
from sklearn.naive_bayes import BernoulliNB

'''
The features in X are broken down as follows:
[Walks like a duck, Talks like a duck, Is small]

Walks like a duck: 0 = False, 1 = True
Talks like a duck: 0 = False, 1 = True
Is small: 0 = False, 1 = True
'''

# Some data is created to train with
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
# These are our target values (Classes: Duck or Not a duck)
y = np.array(['Duck', 'Not a Duck', 'Not a Duck'])

# This is the code we need for the Multinomial model
clf = BernoulliNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print(clf.predict([[1, 1, 1]]))
