# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation

import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """

    """What is Mira?
        y' = arg max score(f, y'')
        we are comparing y' to y, if y' = y we are 100% certain
        If not then we need to guess"""
    #Used to keep track of our accuracy
    guess = -1

    #Used to keep track of our weights in a list
    weights = {}

    #iterates through Cgrid
    for item in Cgrid:

        #initializes weights to zero for each run
        self.initializeWeightsToZero()

        #iterates through the training data
        start = 0
        iteration = self.max_iterations
        for num in range(iteration):
            for data in range(len(trainingData)):
                
                #Not a instant match
                if self.classify([trainingData[data]])[start] != trainingLabels[data]:
                    t = 1.0
                    datum = trainingData[data]
                    label = trainingLabels[data]
                    predicted_label = self.classify([datum])[0]

                    #from the descripction we have t = (w^y' - w^2) / (2* ||f||^2)
                    #only calculates top portion of the eqaution
                    for feat in datum.keys():
                        t += (self.weights[predicted_label][feat] - self.weights[label][feat]) * datum[feat]
                    #calculates the bottom half

                    temp = (2 * sum([x ** 2 for x in datum.values()]))
                    t = t / temp
                    t = min(item, t)
                    
                    hold = datum.copy()
                    for feat in datum.keys():
                        hold[feat] = hold[feat] * t

                    # now we have to update all the weights
                    for feat in hold.keys():
                        self.weights[label][feat] += hold[feat]
                        self.weights[predicted_label][feat] -= hold[feat]

                #Match found
                else:
                    continue

        p = self.classify(validationData)

        target = 0
        index = 0
        #counts how many times we get a right prediction
        while index < len(p):
            #
            if p[index] == validationLabels[index]:
                target = target + 1

            #next index
            index = index + 1

        if guess < target:
            weights = self.weights.copy()
            guess = target

    self.weights = weights


  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    features_odds = []
    f1 = self.features

    for f in f1:
        features_odds.append((self.weights[label1][f] - self.weights[label2][f], f))

    #sorts the features
    features_odds = sorted(featuresOdds, key = lambda x: x[0], reverse = True)[:100]

    newfeat = list(map(lambda x: x[1], features_odds))
    return newfeat
