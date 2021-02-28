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
            probability_positive += math.log((max(1, positive[word]) + smoothing_parameter) / (num_positive_words + 3 * smoothing_parameter))
            probability_negative += math.log((max(1, negative[word]) + smoothing_parameter) / (num_negative_words + 3 * smoothing_parameter))

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



def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=1.0, bigram_lambda=0.5,pos_prior=0.8):
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
    
    # Creates the word pairs from the train_set
    # Pulled from https://www.geeksforgeeks.org/python-all-possible-pairs-in-list/
    word_pairs = [(a, b) for idx, a in enumerate(train_set) for b in train_set[idx + 1:]]
    
    # Creates list of words associated with positive value and negative values for bigram model.
    bi_positive, bi_negative = preprocessFrequencies(word_pairs, train_labels)
    bi_num_pos_words = sum(positive.values())
    bi_num_neg_words = sum(negative.values())
    
    # Creates list of words associated with positive value and negative values for unigram model.
    uni_positive, uni_negative = preprocessFrequencies(train_set, train_labels)
    uni_num_pos_words = sum(positive.values())
    uni_num_neg_words = sum(negative.values())
    
    
    # Predicts probabilities for bigram and unigram models.
    bi_prob = predictProbabilities(word_pairs, bi_positive, bi_negative, bi_num_pos_words, bi_num_neg_words, bigram_smoothing_parameter, pos_prior)
    uni_prob = predictProbabilities(dev_set, uni_positive, uni_negative, uni_num_pos_words, uni_num_neg_words, unigram_smoothing_parameter, pos_prior)
    return (1-bigram_lambda)*bi_prob - (1-unigram_lambda)*uni_prob
