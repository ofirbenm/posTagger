"""
intro2nlp, assignment 3, 2021

In this assignment you will implement a Hidden Markov model
to predict the part of speech sequence for a given sentence.

"""

from math import log, isfinite
from collections import Counter

import sys, os, time, platform, nltk, random


# utility functions to read the corpus
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Ofir Ben Moshe', 'id': '315923151', 'email': 'ofirbenm@post.bgu.ac.il'}


def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence


def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


alpha = 0.01
START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}

# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {}  # transisions probabilities
B = {}  # emissions probabilities


def learn_params(tagged_sentences):
    for sent in tagged_sentences:
        prev_tag = START

        for index, (word, tag) in enumerate(sent):

            # allTagCounts
            allTagCounts[tag] += 1

            # compute perWordTagCounts
            if word not in perWordTagCounts.keys():
                perWordTagCounts[word] = Counter()
                perWordTagCounts[word][tag] += 1
            else:
                perWordTagCounts[word][tag] += 1

            # compute transitionCounts
            if prev_tag not in transitionCounts.keys():
                transitionCounts[prev_tag] = Counter()
                transitionCounts[prev_tag][tag] += 1
            else:
                transitionCounts[prev_tag][tag] += 1
            # End dummy
            if (index + 1) == len(sent):
                end_tag = END
                if tag not in transitionCounts.keys():
                    transitionCounts[tag] = Counter()
                    transitionCounts[tag][end_tag] += 1
                else:
                    transitionCounts[tag][end_tag] += 1

            # compute emissionCounts
            if tag not in emissionCounts.keys():
                emissionCounts[tag] = Counter()
            emissionCounts[tag][word] += 1

            prev_tag = tag

    # Smooth to transitionCounts and emissionCounts
    for tag_1 in transitionCounts.keys():
        transitionCounts[tag_1][UNK] = 0
        for tag_2 in transitionCounts[tag_1].keys():
            transitionCounts[tag_1][tag_2] += alpha

    for key in emissionCounts.keys():
        emissionCounts[key][UNK] = 0
        for v in emissionCounts[key].keys():
            emissionCounts[key][v] += alpha

    # normalize counts and store log probability distributions in A
    for previous in transitionCounts.keys():
        total = 0
        for comb in transitionCounts[previous].keys():
            total += transitionCounts[previous][comb]
        for tag in transitionCounts[previous].keys():
            if previous not in A.keys():
                A[previous] = {}
            A[previous][tag] = log(float(transitionCounts[previous][tag]) / total)

    # normalize counts and store log probability distributions in B
    for tag in emissionCounts.keys():
        total = 0
        for word in emissionCounts[tag].keys():
            total += emissionCounts[tag][word]
        for word in emissionCounts[tag].keys():
            if tag not in B.keys():
                B[tag] = {}
            B[tag][word] = log(float(emissionCounts[tag][word]) / total)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    tagged_sentence = []
    for word in sentence:
        if word in perWordTagCounts.keys():
            pred_tag = max(perWordTagCounts[word], key=perWordTagCounts[word].get)
            tagged_sentence.append((word, pred_tag))
        else:
            pred_tag = allTagCounts.most_common(1)[0][0]
            tagged_sentence.append((word, pred_tag))

    return tagged_sentence


# ===========================================
#       POS tagging with HMM
# ===========================================


def hmm_tag_sentence(sentence, A, B):
    # fill in the Viterbi chart
    end_item = viterbi(sentence, A, B)

    # then retrace your steps from the best way to end the sentence, following backpointers
    lst = retrace(end_item, len(sentence))

    tagged_sentence = []
    for index, (word) in enumerate(sentence):
        tup = (word, lst[index])
        tagged_sentence.append(tup)

    # finally return the list of tagged words
    return tagged_sentence


def viterbi(sentence, A, B):
    # make a dummy item with a START tag
    current = [(START, "", 0)]

    # for each word in the sentence:
    #    previous list = current list
    #    current list = []
    #    determine the possible tags for this word
    #
    #    for each tag of the possible tags:
    #         add the highest-scoring item with this tag to the current list
    for index, word in enumerate(sentence):
        prev_lst = current
        current = []
        tagset = []

        for tag in B.keys():
            if word in B[tag].keys():
                tagset.append(tag)
        if not tagset:
            tagset = B.keys()

        for poss_tag in tagset:
            best_item = predict_next_best(word, poss_tag, prev_lst, A, B)
            current.append(best_item)

        if (index + 1) == len(sentence):
            max_prob = -10000
            best_predecessor = ()
            for tup in current:
                prev_tag = tup[0]
                if END in A[prev_tag].keys():
                    prob = tup[2] + A[prev_tag][END]
                else:
                    prob = tup[2] + A[prev_tag][UNK]
                if prob > max_prob:
                    max_prob = prob
                    best_predecessor = tup

            # end the sequence with a dummy
            return (END, best_predecessor, max_prob)


def retrace(end_item, sentence_length):
    tags = []
    item = end_item[1]
    for iter in range(sentence_length):
        tags.append(item[0])
        item = item[1]
    tags.reverse()
    return tags


def predict_next_best(word, tag, predecessor_list, A, B):
    # determine the emission probability:
    #  the probability that this tag will emit this word
    if word in B[tag].keys():
        emission_prob = B[tag][word]
    else:
        emission_prob = B[tag][UNK]

    best_prob = -10000.0
    best_predecessor = ()
    for tup in predecessor_list:
        prev_tag = tup[0]
        if tag in A[prev_tag].keys():
            prob = emission_prob + A[prev_tag][tag] + tup[2]
        else:
            prob = emission_prob + A[prev_tag][UNK] + tup[2]
        if prob > best_prob:
            best_prob = prob
            best_predecessor = tup

    # return a new item (tag, best predecessor, best total log probability)
    return (tag, best_predecessor, best_prob)


def joint_prob(sentence, A, B):
    p = 0  # joint log prob. of words and tags
    prev_tag = START
    for index, (word, tag) in enumerate(sentence):
        if word not in B[tag].keys():
            if tag in A[prev_tag].keys():
                p += (A[prev_tag][tag] + B[tag][UNK])
            else:
                p += (A[prev_tag][UNK] + B[tag][UNK])
        else:
            if tag in A[prev_tag].keys():
                p += (A[prev_tag][tag] + B[tag][word])
            else:
                p += (A[prev_tag][UNK] + B[tag][word])
        prev_tag = tag
        if (index + 1) == len(sentence):
            if END in A[tag].keys():
                p += A[tag][END]
            else:
                p += A[tag][UNK]

    assert isfinite(p) and p < 0  # Should be negative. Think why!
    return p


# ===========================================================
#       Wrapper function (tagging with a specified model)
# ===========================================================

def tag_sentence(sentence, model):
    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])


def count_correct(gold_sentence, pred_sentence):

    assert len(gold_sentence) == len(pred_sentence)
    correct = 0
    correctOOV = 0
    OOV = 0
    for index, (word, gold_tag) in enumerate(gold_sentence):
        if gold_tag == pred_sentence[index][1]:
            correct += 1
        if word not in perWordTagCounts.keys():
            OOV += 1
            if gold_tag == pred_sentence[index][1]:
                correctOOV += 1

    return correct, correctOOV, OOV


