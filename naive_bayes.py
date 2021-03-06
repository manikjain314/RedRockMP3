# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

from collections import defaultdict
import math
import numpy as np

def preprocessFrequencies(train_set, train_labels):
    POS = 1
    NEG = 0

    positive = defaultdict(int)
    negative = defaultdict(int)

    for i in range(len(train_set)):
        for word in train_set[i]:
            if train_labels[i] == POS:
                positive[word] += 1
            else:
                negative[word] += 1

    return positive, negative

def predictProbabilities(dev_set, positive, negative, num_positive_words, num_negative_words, smoothing_parameter, pos_prior):
    predictions = []

    for i in range(len(dev_set)):
        probability_positive = 0
        probability_negative = 0
        for word in dev_set[i]:
            probability_positive += math.log((max(0, positive[word]) + smoothing_parameter) / (num_positive_words + 3 * smoothing_parameter))
            probability_negative += math.log((max(0, negative[word]) + smoothing_parameter) / (num_negative_words + 3 * smoothing_parameter))

        #probability_positive += math.log(num_positive_words / (num_positive_words + num_negative_words))
        #probability_negative += math.log(num_negative_words / (num_positive_words + num_negative_words))
        probability_positive += math.log(pos_prior)
        probability_negative += math.log(1-pos_prior)

        # 1 = positive, 0 = negative
        predictions.append(int(probability_positive > probability_negative))

    return predictions


"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=0.05, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """

    positive, negative = preprocessFrequencies(train_set, train_labels)
    num_positive_words = sum(positive.values())
    num_negative_words = sum(negative.values())

    return predictProbabilities(dev_set, positive, negative, num_positive_words, num_negative_words, smoothing_parameter, pos_prior)


def preprocessWordFairFreqs(train_set, train_labels):
    POS = 1
    NEG = 0

    positive_pairs = defaultdict(int)
    negative_pairs = defaultdict(int)

    for i in range (len(train_set)):
        review = train_set[i]
        for j in range (len(review) - 1):
            if train_labels[i] == POS:
                positive_pairs[(review[j], review[j+1])] += 1
            else:
                negative_pairs[(review[j], review[j+1])] += 1
    
    return positive_pairs, negative_pairs


def predictPairedProbabilities(dev_set, positive_pairs, negative_pairs, num_positive_pairs, num_negative_pairs, smoothing_parameter, pos_prior):
    predictions = []

    for i in range(len(dev_set)):
        probability_positive = 0
        probability_negative = 0
        review = dev_set[i]
        for j in range (len(review)-1):
            probability_positive += math.log((max(0, positive_pairs[(review[j], review[j+1])]) + smoothing_parameter) / (num_positive_pairs + 3 * smoothing_parameter))
            probability_negative += math.log((max(0, negative_pairs[(review[j], review[j+1])]) + smoothing_parameter) / (num_negative_pairs + 3 * smoothing_parameter))

        #probability_positive += math.log(num_positive_words / (num_positive_words + num_negative_words))
        #probability_negative += math.log(num_negative_words / (num_positive_words + num_negative_words))
        probability_positive += math.log(pos_prior)
        probability_negative += math.log(1-pos_prior)

        # 1 = positive, 0 = negative
        predictions.append(int(probability_positive > probability_negative))

    return predictions
            


def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.1, bigram_smoothing_parameter=0.5, bigram_lambda=0.5,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model

    
    # Creates list of words associated with positive value and negative values for bigram model.
    bi_positive_pairs, bi_negative_pairs = preprocessWordFairFreqs(train_set, train_labels)
    
    bi_num_pos_pairs = sum(bi_positive_pairs.values())
    bi_num_neg_pairs = sum(bi_negative_pairs.values())
    
    # Creates list of words associated with positive value and negative values for unigram model.
    uni_positive, uni_negative = preprocessFrequencies(train_set, train_labels)
    uni_num_pos_words = sum(uni_positive.values())
    uni_num_neg_words = sum(uni_negative.values())
    
    
    # Predicts probabilities for bigram and unigram models.
    bi_prob = predictPairedProbabilities(dev_set, bi_positive_pairs, bi_negative_pairs, bi_num_pos_pairs, bi_num_neg_pairs, bigram_smoothing_parameter, pos_prior)
    uni_prob = predictProbabilities(dev_set, uni_positive, uni_negative, uni_num_pos_words, uni_num_neg_words, unigram_smoothing_parameter, pos_prior)

    bi_prob_arr = np.array(bi_prob)
    uni_prob_arr = np.array(uni_prob)
    #print (*bi_prob_arr)
    #return (bigram_lambda)*np.log(bi_prob_arr) + (1 - bigram_lambda)*np.log(uni_prob_arr)
    return bi_prob_arr
    #return uni_prob_arr
