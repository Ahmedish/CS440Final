# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.probs = {}
    self.extra = False

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter
    that gives the best accuracy on the held-out validationData.

    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """
    count = util.Counter()
    pd = util.Counter()
    for label in trainingLabels:
        count[label] += 1
        pd[label] += 1
    count = {}
    for feature in self.features:
        count[feature] = {}
        for label in self.legalLabels:
            count[feature][label] = {
                0: 0,
                1: 0
            }
    for i in range(len(trainingData)):
        datum = trainingData[i]
        label = trainingLabels[i]
        for (feature, val) in datum.items():
            count[feature][label][val] += 1
    pd.normalize()
    self.priorD = pd

    bestK = -1
    bestAcc = -1
    for k in kgrid:
        temp_probs = {}
        for (feature, labels) in count.items():
            temp_probs[feature] = {}
            for (label, vals) in labels.items():
                temp_probs[feature][label] = {}
                total = sum(count[feature][label].values())
                total += 2*k
                for (val, c) in vals.items():
                    #Normalizing the probability
                    temp_probs[feature][label][val] = (count[feature][label][val] + k) / total
        self.probs = temp_probs
        pred = self.classify(validationData)

        #count correct predictions
        acc = 0
        for i in range(len(pred)):
            if pred[i] == validationLabels[i]:
                acc += 1

        #checking k values for best accuracy and if found store it
        if acc > bestAcc:
            bestK = k
            bestAcc = acc

    #calculate the probabilities using best k to get most accurate results
    t_probs = {}
    for (feature, labels) in count.items():
        t_probs[feature] = {}
        for (label, vals) in labels.items():
            t_probs[feature][label] = {}
            total = sum(count[feature][label].values())
            total += 2*bestK
            for (val, c) in vals.items():
                t_probs[feature][label][val] = (count[feature][label][val] + bestK) / total

    self.probs = t_probs


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """
    lJoint = util.Counter()
    for label in self.legalLabels:
        lJoint[label] = math.log(self.priorD[label])
        for (feat, val) in datum.items():
            p = self.probs[feat][label][val];
            lJoint[label] += math.log(p)
    return lJoint

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)

    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    features_odds = []

    for feat in self.features:
        #Doing what is defined above in the comment P(feature=1 | label1)/P(feature=1 | label2)
        features_odds.append((self.probs[feat][label1][1] / self.probs[feat][label2][1], feat))

    #First we sort the featuresOdds list and then reverse it to get the last 100 of the list
    features_odds.sort()
    fOddslist = list(map(lambda x: x[1], features_odds[-100:]))
    return fOddslist








