#!/usr/bin/python26
# -*- coding: utf-8 -*-

# eval.py
# PGA 11/9/12
# Evaluation module

# test files in
# /home/j/anick/patent-classifier/ontology/annotation/en
# doc_feats.eval
# phr_occ.lab (phrases labeled y, n, "" from a general ontology)
# phr_occ.eval.unlab (unlabeled phrases corresponding to doc_feats.eval, with associated sentences for each phrase)
# phr_occ.eval.lab (labeled phrases corresponding to doc_feats.eval in the format of phr_occ.lab)


# Assume we have 
# 1. a single file with a set of lines in doc_feats format
# 2. a corresponding manually labeled file for each phrase [gold dynamic labels: Gd]
# 4. a system labeled file for each phrase with thresholds [system labels: Sd@T]
# 3. an independent file of machine labled phrases (static ontology with thresholds) [gold static labels: Gs@T]

# We measure dynamic precision/recall @threshold as

# Precision@T = intersection(#Y(Sd@T), #Y(Gd))/#Y(Sd@T)  [true positives / all system positives]
# Recall@T = intersection(#Y(Sd@T), #Y(Gd))/#Y(Gd)  [true positives / gold positives]

# We measure static precision/recall (based on static ontology) @threshold as

# Precision@T = intersection(#Y(Ss@T), #Y(Gd))/#Y(Ss@T)  [true positives / all system positives]
# Recall@T = intersection(#Y(Ss@T), #Y(Gd))/#Y(Gs)  [true positives / gold positives]

# Note that the model used should be consistent across each set  of evaluations.
# However,  we should also run metrics on different models (built on different sized document sets)
    
"""
I named the doc_feats file sample1
bash-3.2$ cd /home/j/anick/patent-classifier/ontology/annotation/en/
bash-3.2$ ls
doc_feats.eval  phr_occ.cum  phr_occ.eval.unlab  phr_occ.lab  phr_occ.uct
bash-3.2$ cp doc_feats.eval sample1

To run tests, uyou can use one of the hard-coded versions:

    eval.ten(.8)  # run english test with threshold set to .9
    eval.tcn(.8)  # run Chinese test
    eval.tde(.8)  # run German test

To get the nubers with threshold set to 0, use 0.000000000001 (to avoid cofusion with no data)

"""

import train
#import mallet
import os
import collections
import codecs


####################################################################################
### Testing static classification 
### 

class PRA:

    """precision/recall/accuracy calculation"""
    
    def __init__(self, d_eval, d_system, threshold, s_log, debug_c=True):

        self.debug_p = False
        self.debug_c = debug_c
        self.d_eval = d_eval
        self.d_system = d_system
        self.true_pos = 0
        self.system_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_neg = 0
        self.correct = 0
        self.total = 0   # number terms to be evaluated
        self.score_count = 0 # number terms for which we have scores
        self.eval_pos = 0
        self.eval_labeled = 0
        
        i = 0
        none_count = 0

        for phrase in self.d_eval.keys():

            i += 1
            self.total += 1
            gold_label = self.d_eval.get(phrase)
            system_score = self.d_system.get(phrase)
            self.debug_phrase(i, phrase, gold_label, system_score)

            if system_score is None:
                # the gold phrase doesn't appear in the scored subset (data sample),
                # default the score to 0.0 and set the system label to 'u' (unknown).
                system_score = 0.0
                none_count += 1
                system_label = "u"
            elif system_score > threshold:
                system_label = "y"
                self.score_count += 1

            elif system_score > 0.0:
                system_label = "n"
                self.score_count += 1
            else:
                system_label = "n"

            if gold_label == "y" and system_label == "y": self.true_pos += 1
            elif gold_label == "y" and system_label == "n": self.false_neg += 1
            elif gold_label == "n" and system_label == "n": self.true_neg += 1
            elif gold_label == "n" and system_label == "y": self.false_pos += 1
            s_log.write("%s\t|%s|\t%f\t%s\n" % (gold_label, system_label, system_score, phrase))
        
        self.log_missing_eval_phrases(threshold, s_log)
        self.correct = self.true_pos + self.true_neg
        self.debug_counts(d_eval, d_system, i, none_count, s_log)


    def precision(self):
        try:
            return float(self.true_pos) / (self.true_pos + self.false_pos)
        except ZeroDivisionError:
            print "[WARNING: precision] true_pos: %i, false_pos: %i" % (self.true_pos, self.false_pos) 
            return -1

    def recall(self):
        res = float(self.true_pos) / (self.true_pos + self.false_neg)
        return(res)

    def accuracy(self):
        #res = float(self.correct) / self.total
        res = float(self.correct) / self.score_count
        return(res)

    def debug_phrase(self, i, phrase, gold_label, system_score):
        if self.debug_p and i < 10:
            print "[PRA] '%s'" % phrase
            found = 'y' if self.d_system.has_key(phrase) else 'n'
            print "[PRA] gold_label=%s found=%s system_score=%s\n" % (gold_label, found, system_score)

    def debug_counts(self, d_eval, d_system, i, none_count, logfile):
        counts_string = "\nCounts\n" + \
                        "   size of d_eval: %d\n" % len(d_eval) + \
                      "   size of d_system: %d\n" % len(d_system) + \
                      "   non-matches (not in d_system): %i\n" % none_count
        logfile.write(counts_string)
        if self.debug_c:
            print counts_string

    def log_missing_eval_phrases(self, threshold, log):
        log.write("\nPHRASES NOT IN D_EVAL\n")
        for phrase, score in self.d_system.items():
            gold_label = self.d_eval.get(phrase)
            system_label = 'y' if score > threshold else 'n'
            if gold_label is None:
                log.write("u\t|%s|\t%f\t%s\n" % (system_label, score, phrase))
                

    
def default_n():
    """Initialize the label dictionaries to the label 'n'. """
    return("n")


class EvalData:

    """ Class to take an evaluation (gold standard) file (terms labeled with 'y', 'n') and
    the output of the mallet classifier (in the form of a scores file), and populate
    dictionaries to hold this information, keyed by term.  """
    
    # optional parameter to use min, max, or average score for thresholding
    def __init__(self, eval_file, system_file, score_type="average", count_threshold=1):

        self.debug = False
        self.d_eval_phr2label = {}   # map from evaluation phrase to class
        self.d_system_phr2score = {} # map from phrase to score (between 0.0 and 1.0)
        s_eval = codecs.open(eval_file, "r", encoding='utf-8')
        s_system = codecs.open(system_file, "r", encoding='utf-8')

        # populate dictionaries, part 1: gold data: manually annotated file of random phrases
        for line in s_eval:
            if line.strip() == '': continue
            if line.lstrip()[0] == '#': continue
            # if line begins with tab, it has not been labeled, since y/n should appear in
            # col 1 before the tab.
            if line[0] != "\t":
                # also omit any header lines that don't contain a tab in column two
                if line[1] == "\t":
                    line = line.strip()
                    (label, phrase) = line.split("\t")
                    # normalize segmentation by removing all spaces from Chinese words
                    # phrase = phrase.replace(' ','')
                    self.d_eval_phr2label[phrase] = label
                    # NOTE how the phrase and label are printed out. First byte(s) of
                    # phrase seems lost or misplaced
                    # print "[EvalData] storing label/phrase: %s, %s" % (label, phrase)

        # output from mallet maxent classifier ("yes" score, averaged over multiple document instances)
        n = 0  # number of lines (terms)
        c = 0  # number of terms with count >= count_threshold

        for line in s_system:
            n += 1
            line = line.rstrip()

            # the following were removed at merge time, but may be needed (MV)
            #
            # # normalize segmentation by removing all spaces from Chinese words
            # # phrase = phrase.replace(' ','')
            # self.d_system_phr2score[phrase] = float(score)
            # self.debug_phrase(n, phrase, score)

            (phrase, average, count, min, max) = line.split("\t")
            
            count = int(count)

            # only use scorees for terms that appear in at least <count_threshold> docs
            if count >= count_threshold:
                # default score is the average score over docs
                score = average
                # count is the number of documents the term was found in
                count = int(count)

                # chose which score to use based on score_type parameter
                if int(count) > 1: 
                    if score_type == "max":
                        score = max
                    elif score_type == "min":
                        score = min

                #if count == '1': continue
                c += 1
                # normalize segmentation by removing all spaces from Chinese words
                #phrase = phrase.replace(' ','')

                self.d_system_phr2score[phrase] = float(score)

                if self.debug and n < 10:
                    print "[ED]phrase: %s, score: %s" % (phrase, score)
                    if self.d_system_phr2score.has_key(phrase):
                        print "[ED]Found key in d_system: %s" % phrase
                    else:
                        print "[ED]key not found in d_system: %s" % phrase
                    if self.d_eval_phr2label.has_key(phrase):
                        print "[ED]Found key in d_eval: %s" % phrase
                    else:
                        print "[ED]key not found in d_eval: %s" % phrase
                    print "[EvalData]Storing sys score, phrase: %s, score: %f, actual: %f" \
                          % (phrase, float(score), self.d_system_phr2score.get(phrase))

        if self.debug:
            print "Total scores: %i, scores with count >= %i: %i" % (n, count_threshold, c)
        s_eval.close()
        s_system.close()

        
# for a list of terms (eg. annotated terms), compare scores generated in different ways 
# (e.g. different chunker, filter on/off.)

def compare_scores(term_file, score_file_1, score_file_2, output_file):
    d_term2label = {}
    d_term2score1 = {}
    d_term2score2 = {}

    s_eval = open(term_file)
    s_score1 = open(score_file_1)
    s_score2 = open(score_file_2)
    s_output = open(output_file, "w")

    # gold data: manually annotated file of random phrases
    for line in s_eval:

        # if line begins with tab, it has not been labeled, since y/n should appear in col 1 before the tab.
        if line.strip() == '': continue
        if line.lstrip()[0] == '#': continue

        if line[0] != "\t":
            # also omit any header lines that don't contain a tab in column two
            if line[1] == "\t":
                line = line.strip()
                (label, phrase) = line.split("\t")

                d_term2label[phrase] = label
        
    for line in s_score1:

        line = line.rstrip()
        (phrase, score, count, min, max) = line.split("\t")

        if d_term2label.has_key(phrase):
            d_term2score1[phrase] = float(score)

    for line in s_score2:

        line = line.rstrip()
        (phrase, score, count, min, max) = line.split("\t")

        if d_term2label.has_key(phrase):
            d_term2score2[phrase] = float(score)
        
    # do the comparison of scores (and compute average difference in scores)
    count = 0
    diff_sum = 0
    for phrase in d_term2label.keys():
        label = d_term2label.get(phrase)
        score1 = d_term2score1.get(phrase)
        if type(score1) != float:
            score1 = 0.0
        score2 = d_term2score2.get(phrase)
        if type(score2) != float:
            score2 = 0.0

        #print "[COMP]%s\t%s\t%f\t%f" % (phrase, label, score1, score2)
        diff = score1 - score2
        diff_sum = diff + diff_sum
        s_output.write("%s\t%s\t%f\t%f\t%f\n" % (phrase, label, score1, score2, diff))
        #print "%s\t%s\t%f\t%f\t%f" % (phrase, label, score1, score2, diff)
        count += 1
    avg_diff = diff_sum / count
    print "\n[COMP]avg_diff: %f" % avg_diff
    
    s_eval.close()
    s_score1.close()
    s_score2.close()
    s_output.close()

def tcomp1():
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    score_file_1 = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents-20121111/en/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"

    score_file_2 = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.scores.nr.sum.nr"
    output_file = eval_dir + "tcomp1.log"
    compare_scores(eval_test_file, score_file_1, score_file_2, output_file)                           


# tests over the 500 doc patent databases for each language
def tcn(threshold):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_dir = "../eval/"
    eval_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/annotation/cn/phr_occ.eval.lab.txt"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/cn/phr_occ.eval.lab"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/cn/phr_occ.lab"

    system_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents-20121130/cn/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"
    log_file_name = eval_dir + "tcn_c1_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)

# use the annotation data for testing rather than the evaluation data
def tcna(threshold):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/annotation/cn/phr_occ.lab"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/cn/phr_occ.eval.lab"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/cn/phr_occ.lab"

    system_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents-20121130/cn/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"
    log_file_name = eval_dir + "tcna_c1_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)


def tde(threshold):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/annotation/de/phr_occ.eval.lab"
    system_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents-20121130/de/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500" 
    log_file_name = eval_dir + "tde_c1_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)

def ten_c1(threshold, score_type="average", count=1):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_dir = "../eval/"
    # data labeled for phrases chunked by the original rules, which included conjunction and "of"
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    # data labeled for more restrictive chunks"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.newchunk.lab"
    #system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.avg_scores.nr"
    system_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents-20121111/en/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"

    log_file_name = eval_dir + "ten_c1_"  + score_type + "_" + str(count) + "_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name, score_type, count)

# english chunker version 2 using average score
def ten_c2(threshold, score_type="average", count=1):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_dir = "../eval/"
    # data labeled for phrases from original chunker rules"
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    # data labeled for more restrictive chunks"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.newchunk.lab"
    #system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.avg_scores.nr"
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.scores.nr.sum.nr"

    log_file_name = eval_dir + "ten_c2_" + score_type + "_" + str(count) + "_"+ str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name, score_type, count)

# english chunker version 2, using max score
def ten_c2_max(threshold):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_dir = "../eval/"
    # data labeled for phrases from original chunker rules"
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    # data labeled for more restrictive chunks"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.newchunk.lab"
    #system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.avg_scores.nr"
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.scores.nr.sum.nr"

    log_file_name = eval_dir + "ten_c2_max_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name, "max")

# english chunker version 2, using max score
def ten_c2_min(threshold):
    eval_dir = "/home/j/anick/patent-classifier/ontology/eval/"
    eval_dir = "../eval/"
    # data labeled for phrases from original chunker rules"
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    # data labeled for more restrictive chunks"
    #eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.newchunk.lab"
    #system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.avg_scores.nr"
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.scores.nr.sum.nr"

    log_file_name = eval_dir + "ten_c2_max_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name, "min")


def t4(threshold):
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.newchunk.lab"
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.avg_scores.nr"
    log_file_name = "t4_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)


def t3(threshold):
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.1.MaxEnt.out.avg_scores.nr"
    log_file_name = "t3_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)


def t2(threshold):
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    system_test_file = "/home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents-20121111/en/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"
    log_file_name = "t2_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)


def t1(threshold):
    eval_test_file = "/home/j/anick/patent-classifier/ontology/annotation/en/phr_occ.eval.lab"
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/utest.9.MaxEnt.out.scores.sum.nr"
    log_file_name = "t1_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)


# use ctrl-q <tab> to put real tabs in file in emacs
def t0(threshold):
    eval_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/t0.phr_occ.eval.lab" 
    system_test_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en/test/t0.scores.sum.nr" 
    log_file_name = "t0_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)

       
# optional parameter to use min, max, or average score for thresholding
# count restricts scores to terms that appear in <count> documents
def test(eval_test_file, system_test_file, threshold, log_file_name,
         score_type="average", count=1, debug_c=True, command=None):
    edata = EvalData(eval_test_file, system_test_file, score_type, count)
    # open a log file to keep gold and system labels for each phrase
    s_log = codecs.open(log_file_name, "w", 'utf-8')
    if command is not None:
        s_log.write(command + "\n\n")

    pra = PRA(edata.d_eval_phr2label, edata.d_system_phr2score, threshold, s_log, debug_c)
    precision = pra.precision()
    recall = pra.recall()
    accuracy = pra.accuracy()
    total = pra.total

    results_string = "total:%d tp:%s fp:%s fn:%s tn:%s -- " % \
                     (total, pra.true_pos, pra.false_pos, pra.false_neg, pra.true_neg) + \
                     "precision: %.2f, recall: %.2f, accuracy: %.2f, threshold: %.2f" % \
                     (precision, recall, accuracy, threshold)

    print results_string
    s_log.write("\n" + results_string + "\n")
    s_log.close()
    return results_string


####################################################################################
### Testing dynamic classification (incomplete)

# input:doc_feats_file = doc_feats_path/file_name
# output: mallet_file = mallet_subdir/featname + "." + version + ".mallet" = test_dir...
def eval_sample(doc_feats_path, train_dir, test_dir, file_name, featname, version):
    train.make_unlabeled_mallet_file(doc_feats_path, test_dir, file_name, featname, version)

    #Mallet_test parameters: test_file_prefix, version, test_dir, train_file_prefix, train_output_dir

    # create an instance of Mallet_test class to do the rest
    # let's do the work in the test directory for now.
    mtest = mallet.Mallet_test(file_name, version , test_dir, "utrain", train_dir)
    # creates train_path = train_output_dir/train_file_prefix.version
    # Uses path to locate .vectors file for training

    # creates test_path = test_dir/test_file_prefix.version
    # Uses path to create test_mallet_file = test_path.mallet

    # create the mallet vectors file from the mallet file

    ### Not sure if the next line is necessary
    mtest.write_test_mallet_vectors_file()

    mtest.mallet_test_classifier("MaxEnt")

def test1():
    doc_feats_path = "/home/j/anick/patent-classifier/ontology/annotation/en"
    path_en = "/home/j/anick/patent-classifier/ontology/creation/data/patents/en"
    train_dir = os.path.join(path_en, "train")
    test_dir = os.path.join(path_en, "test")
    file_name = "sample1"
    featname = "sample1"
    version = "1"
    eval_sample(doc_feats_path, train_dir, test_dir, file_name, featname, version)

    
##########################


# MV versions of test routines, the first two below were used for the evaluations for the
# final report

def mten_all(test_file):
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        print
        mten(threshold, test_file)
    
def mten(threshold, system_test_file):
    """Evaluate a system output file using a set of previously labeled terms. For English,
    these terms were taken from the first nine random files of the 500 US sample patents
    (see ../evaluation/technology_classifier.txt for details). Note that the system output
    file could be on a disjoint set of terms. Even if the same nine files were used this
    could happen. Part of this is because the annotations were most likely done with the
    abstract/summary filter on. This explains why the set of labeled terms is about 1400
    and the set of system terms is almost 4400. What is not yet explained is why almost
    700 elements from the evaluation set do not occur in the system output."""
    log_dir = "../evaluation/logs/"
    # data labeled for phrases chunked by the original rules, which included conjunction and "of"
    eval_test_file = "../annotation/en/phr_occ.eval.lab"
    log_file_name = log_dir + "mten_final_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)

def mtcn(threshold):
    eval_dir = "../eval/"
    # data labeled for phrases chunked by the original rules, which included conjunction and "of"
    eval_test_file = "../annotation/cn/phr_occ.eval.lab.txt"
    # data labeled for more restrictive chunks"
    system_test_file = "data/patents/cn/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"
    log_file_name = eval_dir + "tcn_c1_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)

def mtde(threshold):
    eval_dir = "../eval/"
    # data labeled for phrases chunked by the original rules, which included conjunction and "of"
    eval_test_file = "../annotation/de/phr_occ.eval.lab.txt"
    # data labeled for more restrictive chunks"
    system_test_file = "data/patents/de/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000500"
    log_file_name = eval_dir + "tde_c1_" + str(threshold) + ".gs.log"
    test(eval_test_file, system_test_file, threshold, log_file_name)


# NOT USED
    
def mo():
    training_file = "../annotation/en/phr_occ.lab"
    fragment_file = "../annotation/en/ontology-evaluation-20121128.lab"
    get_overlap(training_file, fragment_file)
    
def get_overlap(training_set, ontology_fragment):
    training_y = {}
    for line in codecs.open(training_set, encoding='utf-8'):
        (boolean, term) = line.rstrip().split("\t")
        if boolean == 'y':
            training_y[term] = True
    terms = 0
    terms_in_training_set = 0
    for line in codecs.open(ontology_fragment, encoding='utf-8'):
        terms += 1
        (boolean, term) = line.rstrip().split("\t")
        if boolean == 'y':
            if training_y.has_key(term):
                terms_in_training_set += 1
    print "Terms in training set: %d/%d (%.0f%%)" % (terms_in_training_set, terms,
                                                     100*(terms_in_training_set/float(terms)))



if __name__ == '__main__':

    import sys

    test_file = "data/patents/yy/en/test/utest.1.MaxEnt.out.s5.scores.sum.nr.000000-000009"

    if len(sys.argv) == 2:
        test_file = sys.argv[1]
        mten_all(test_file)
    else:
        threshold = float(sys.argv[1])
        test_file = sys.argv[2]
        mten(threshold, test_file)
    
