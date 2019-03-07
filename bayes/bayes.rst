##########################
Naive Bayes Classification
##########################
------------
Introduction
------------
A recurring problem in machine learning is the need to classify input into some preexisting class. Consider the following example.

Say we want to classify some random piece of fruit we found lying around. In this example, we have three existing fruit categories: apple, blueberry, and coconut. Each of these fruits have three features we care about: size, weight, and color. This information is shown in the table below.

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/bayes/images/bayes/Table.png

We observe the piece of fruit we found and determine its features are moderate, heavy, and red. We can compare these features against the features of our known classes to guess what type of fruit it is. The unknown fruit shares 2 of 3 characteristics with the apple class so we guess that it’s an apple. The random fruit is heavy like a coconut but it shares more features with the apple class. We used the fact that the random fruit is moderately sized and red like an apple to make our guess.

-----------
What is it?
-----------
Naive Bayes is a classification technique that uses probabilities we already know to determine how to classify input. These probabilities are related to existing classes and what features they have. Like in the example above, we choose the class that most resembles our input as its classification. This technique is based around using Bayes’ Theorem. If you’re unfamiliar with what Bayes’ Theorem is, don’t worry! We will explain it in the next section.

--------------
Bayes’ Theorem
--------------
Bayes’ Theorem is a very useful theorem that shows up in probability theory and other disciplines. It looks like this:

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/bayes/images/bayes/Bayes.png

With Bayes’ Theorem we examine conditional probabilities (the probability of an event happening given another event has happened). Here P(A|B) is the probability that event A will happen given that event B has happened. We can determine this value using other information we know about events A and B. We need to know P(B|A) (the probability that event B will happen given that event A has happened), P(B) (the probability event B will happen), and P(A) (the probability event A will happen). The interesting thing is we can apply Bayes’ Theorem to machine learning problems!

-----------
Naive Bayes
-----------
Naive Bayes classification uses Bayes’ Theorem with some additional assumptions. The main thing we will assume is that features are independent. In the case of our fruit example above, being red does not affect the probability of being moderately sized. This is often not the case in real problems where features may have complex relationships. This is why “naive” is in the name. The result we care about: the probability of a set of features occurring given a certain class is now the same as the product of all the different probabilities of each feature occurring given that class. If this seems complicated, don’t worry! The code will handle the number crunching for us. Just remember that we are assuming that features are independent of each other.

We take some input and calculate the probability of it happening given that it belongs to one of our classes. We must do this for each of our classes. After we have all these probabilities, we just take the one that’s the largest as our prediction for what class the input belongs to.

----------
Algorithms
----------
Below are some common models used for Naive Bayes classification. We have separated them into two general cases based on what type of feature distributions they use: continuous and discrete. Continuous means real valued (you can have decimal answers) and discrete means a count (you only have whole number answers). Also provided are the relevant code snippets for each algorithm.

~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gaussian Model (Continuous)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gaussian models assumes features follow a normal distribution. As far as we need to know, a normal distribution is just a specific type of distribution. This is another big assumption because many features do not follow a normal distribution. While this is true, assuming a normal distribution makes our calculations a lot easier. We use Gaussian models when features are not counts and include decimal values.

The following code snippet is based around guessing a color from RGB percentages.

.. code-block:: python

   import numpy as np
   from sklearn.naive_bayes import GaussianNB

   '''
   The features in X are broken down as follows:
   [Red %, Green %, Blue %]
   '''

   # Some data is created to train with
   X = np.array([[.5, 0, .5], [1, 1, 0], [0, 0, 0]])
   # These are our target values (Classes: Purple, Yellow, or Black)
   y = np.array(['Purple', 'Yellow', 'Black'])

   # This is the code we need for the Multinomial model
   clf = GaussianNB()
   # We train the model on our data
   clf.fit(X, y)
   
   # Now we can make a prediction on what class new data belongs to
   print(clf.predict([[1, 0, 1]]))

Multinomial Model (Discrete)
Multinomial models are used when we are working with discrete counts. Specifically, we want to use them when we are counting how often a feature occurs. For example, we might want to count how often the word “count” appears on this page.

The following code snippet is based on our fruit example.

.. code-block:: python

   import numpy as np
   from sklearn.naive_bayes import MultinomialNB

   '''
   The features in X are broken down as follows:
   [Size, Weight, Color]

   Size: 0 = Small, 1 = Moderate, 2 = Large
   Weight: 0 = Light, 1 = Moderate, 2 = Heavy
   Color: 0 = Red, 1 = Blue, 2 = Brown
   '''

   # Some data is created to train with
   X = np.array([[1, 1, 0], [0, 0, 1], [2, 2, 2]])
   # These are our target values (Classes: Apple, Blueberry, or Coconut)
   y = np.array(['Apple', 'Blueberry', 'Coconut'])

   # This is the code we need for the Multinomial model
   clf = MultinomialNB()
   # We train the model on our data
   clf.fit(X, y)

   # Now we can make a prediction on what class new data belongs to
   print(clf.predict([[1, 2, 0]]))

~~~~~~~~~~~~~~~~~~~~~~~~~~
Bernoulli Model (Discrete)
~~~~~~~~~~~~~~~~~~~~~~~~~~
Bernoulli models are also used when we are working with discrete counts. Unlike the multinomial case, here we are counting whether or not a feature occurred. For example, we might want to check if the word “count” appears on this page. We can also use Bernoulli models when features only have 2 possible values.

The following code snippet is based around guessing if something is a duck or not.

.. code-block:: python

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

-------
Summary
-------
Naive Bayes classification lets us classify an input based on probabilities of existing classes and features. As shown in the code snippets above, you don’t need a lot of training data for Naive Bayes to be useful. Another bonus is speed which can come in handy for real-time predictions. We make a lot of assumptions to use Naive Bayes so results should be taken with a grain of salt. But if you don’t have much data and need fast results, Naive Bayes is a good choice for classification problems.
