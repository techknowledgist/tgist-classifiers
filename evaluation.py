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

import os, sys, collections, codecs
import train



####################################################################################
### Testing static classification 
### 

class PRA:

    """Precision, recall and accuracy calculation. Takes a dictionary with gold
    standard terms, mapped to 'y' or 'n', and a dictionary of system results,
    mapped to system scores. Depending on the threshold, system scores will be
    mapped to 'y' or 'n' and PR&A are calculated over those, using only those
    terms for which we have both gold standard and system data.

    Will use all terms in the intersection of gold standard and system, unless
    the terms parameter is set to 'single-token-terms' or 'multi-token-terms',
    in which case a subset pf the interseciton is taken.

    The filter parameter can be used to hand in a set of terms that should be
    excluded from the evaluation, this could be the list of terms in the
    training set.

    """

    def __init__(self, d_eval, d_system, threshold=None,
                 term_type="all", term_filter=None, debug_c=True):
        self.debug_p = False
        #self.debug_p = True
        self.debug_c = debug_c
        self.d_eval = d_eval
        self.d_system = d_system
        self.threshold = threshold
        self.eval_terms = None                # set or list of terms to evaluate, to be filled in
        self.eval_terms_type = term_type      # 'all', 'single-token-terms' or 'multi-token-terms'
        self.eval_terms_filter = term_filter  # file with terms not to include
        self.eval_terms_count1 = None         # number of terms to be evaluated before filter
        self.eval_terms_count2 = None         # number of terms to be evaluated after filter
        self.eval_terms_scores = []           # list of all scores used
        self.collect_terms_to_evaluate()
        self.reset_counts()
        if threshold is not None:
            self.calculate_counts(threshold)

    def collect_terms_to_evaluate(self):
        # these are the terms that are in both dictionaries and that can be evaluated
        self.eval_terms = set(self.d_eval).intersection(self.d_system)
        # select the subset if you want single or multi token terms only
        if self.eval_terms_type == 'single-token-terms':
            self.eval_terms = [t for t in self.eval_terms if len(t.split()) == 1]
        elif self.eval_terms_type == 'multi-token-terms':
            self.eval_terms = [t for t in self.eval_terms if len(t.split()) > 1]
        self.eval_terms_count1 = len(self.eval_terms)
        self.eval_terms_count2 = len(self.eval_terms)
        # remove terms that should not be included
        if self.eval_terms_filter is not None:
            fh = codecs.open(self.eval_terms_filter, encoding='utf-8')
            terms_to_filter = {}
            for line in fh:
                fields = line.strip().split("\t")
                if len(fields) == 2:
                    terms_to_filter[fields[1]] = True
            self.eval_terms = set(self.eval_terms).difference(terms_to_filter)
            self.eval_terms_count2 = len(self.eval_terms)

    def reset_counts(self):
        self.tp = 0   # true positives
        self.fp = 0   # false positives
        self.fn = 0   # false negatives
        self.tn = 0   # true negatives

    def calculate_counts(self, threshold):
        self.reset_counts()
        self.threshold = threshold
        i = 0
        for phrase in self.eval_terms:
            i += 1
            self.update_counts(i, phrase)

    def update_counts(self, i, phrase):
        gold_label = self.d_eval.get(phrase)
        system_score = self.d_system.get(phrase)
        self.debug_phrase(i, phrase, gold_label, system_score)
        if system_score is None:
            # the gold phrase doesn't appear in the scored subset (data sample),
            # default the score to 0.0 and set the system label to 'u' (unknown).
            system_score = 0.0
            system_label = "u"
        else:
            system_label = "y" if system_score > self.threshold else "n"
        if gold_label == "y" and system_label == "y": self.tp += 1
        elif gold_label == "y" and system_label == "n": self.fn += 1
        elif gold_label == "n" and system_label == "n": self.tn += 1
        elif gold_label == "n" and system_label == "y": self.fp += 1
        self.eval_terms_scores.append("%s\t|%s|\t%f\t%s\n" \
                                      % (gold_label, system_label, system_score, phrase))

    def total(self):
        return self.tp + self.tn + self.fp + self.fn

    def precision(self):
        try:
            return float(self.tp) / (self.tp + self.fp)
        except ZeroDivisionError:
            return -1

    def recall(self):
        try:
            return float(self.tp) / (self.tp + self.fn)
        except ZeroDivisionError:
            return -1

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return (2 * p * r) / (p + r)

    def accuracy(self):
        try:
            return float(self.tp + self.tn) / self.total()
        except ZeroDivisionError:
            return -1

    def debug_phrase(self, i, phrase, gold_label, system_score):
        if self.debug_p and i < 10:
            print "[PRA] '%s'" % phrase
            found = 'y' if self.d_system.has_key(phrase) else 'n'
            print "[PRA] gold_label=%s found=%s system_score=%s\n" \
                  % (gold_label, found, system_score)

    def log_missing_eval_phrases(self, threshold, log):
        log.write("\nPHRASES NOT IN D_EVAL\n")
        for phrase, score in self.d_system.items():
            gold_label = self.d_eval.get(phrase)
            system_label = 'y' if score > threshold else 'n'
            if gold_label is None:
                log.write("u\t|%s|\t%f\t%s\n" % (system_label, score, phrase))

    def count_terms_in_overlap(self):
        return len( set(self.d_eval).intersection(self.d_system))

    def results_string(self):
        return \
            "threshold: %.2f {%s %s} ==> " % \
            (self.threshold, self.term_type_as_short_string(),
             self.term_filter_as_short_string()) + \
            "p=%.2f r=%.2f f1=%.2f acc=%.2f - " % \
            (self.precision(), self.recall(), self.f1_score(), self.accuracy()) + \
            "(total:%04d tp:%04d tn:%04d fp:%04d fn:%04d)\n" % \
            (self.total(), self.tp, self.tn, self.fp, self.fn)

    def term_type_as_short_string(self):
        if self.eval_terms_type == 'all': return 'all'
        if self.eval_terms_type == 'single-token-terms': return 'stt'
        if self.eval_terms_type == 'multi-token-terms': return 'mtt'

    def term_filter_as_short_string(self):
        return 'ntf' if self.eval_terms_filter is None else 'ytf'

    def pp_counts(self, fh=sys.stdout):
        fh.write(self.results_string())

    def pp_counts_long(self, fh=sys.stdout):
        fh.write("true positives   %4d\n" % self.tp)
        fh.write("true negatives   %4d\n" % self.tn)
        fh.write("false positives  %4d\n" % self.fp )
        fh.write("false negatives  %4d\n" % self.fn)
        fh.write("total            %4d\n\n" % self.total())
        fh.write("precision        %.2f\n" % self.precision())
        fh.write("recall           %.2f\n" % self.recall())
        fh.write("f1-score         %.2f\n" % self.f1_score())
        fh.write("accuracy         %.2f\n\n" % self.accuracy())

    
class EvalData:

    """ Class to take a gold standard file (terms labeled with 'y', 'n') and the
    output of the mallet classifier (in the form of a scores file), and populate
    dictionaries to hold this information, indexed by term. """
    
    def __init__(self, eval_file, system_file, score_type="average", count_threshold=1):
        self.d_eval_phr2label = {}   # map from evaluation phrase to class
        self.d_system_phr2score = {} # map from phrase to score (between 0.0 and 1.0)
        self._populate_gold_data_dictionary(eval_file)
        self._populate_system_dictionary(system_file, score_type, count_threshold)

    def _populate_gold_data_dictionary(self, eval_file):
        """Populate dictionaries, part 1: gold data from a manually annotated
        file of random phrases."""
        s_eval = codecs.open(eval_file, "r", encoding='utf-8')
        for line in s_eval:
            self._add_eval_line(line)
        s_eval.close()

    def _add_eval_line(self, line):
        """Add the label for a phrase to the eval dictionary."""
        if line.strip() == '': return
        if line.lstrip()[0] == '#': return
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

    def _populate_system_dictionary(self, system_file, score_type, count_threshold):
        """Takes the output from the mallet maxent classifier with the 'y'
        score, averaged over multiple document instances."""
        self.debug = False
        #self.debug = True
        s_system = codecs.open(system_file, "r", encoding='utf-8')
        n = 0  # number of lines (terms)
        c = 0  # number of terms with count >= count_threshold
        for line in s_system:
            n += 1
            # count is the number of documents the term was found in
            (phrase, average, count, min, max) = line.rstrip().split("\t")
            count = int(count)
            # only use scores for terms that appear in at least <count_threshold> docs
            if count >= count_threshold:
                score = self._set_score(count, average, min, max, score_type)
                c += 1
                phrase = normalize_phrase(phrase)
                self.d_system_phr2score[phrase] = score
                if self.debug and n < 10:
                    self._debug_system_dictionary(phrase, score)
        if self.debug:
            print "Total scores: %i, scores with count >= %i: %i" % (n, count_threshold, c)
        s_system.close()

    def _set_score(self, count, average, min, max, score_type):
        # default score is the average score over docs
        score = float(average)
        # chose which score to use based on score_type parameter
        if int(count) > 1:
            if score_type == "max": score = float(max)
            elif score_type == "min": score = float(min)
        return score

    def _debug_system_dictionary(self, phrase, score):
        print "[EvalData] phrase: %s, score: %s" % (phrase, score)
        #if self.d_system_phr2score.has_key(phrase):
        #    print "[EvalData]   in d_system"
        if self.d_eval_phr2label.has_key(phrase):
            print "[EvalData]   in d_eval"
        print "[EvalData]   storing sys score, phrase: %s, score: %f, actual: %f" \
              % (phrase, float(score), self.d_system_phr2score.get(phrase))


def normalize_phrase(phrase):
    """Perhaps needed to normalize segmentation by removing all spaces from
    Chinese words by using phrase.replace(' ',''), now just return the
    phrase."""
    return phrase


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


# test(eval_test_file, system_test_file, threshold, log_file_name)
def test(eval_file, system_file, threshold, log_file=None,
         term_type='all', term_filter=None,
         score_type="average", count=1, debug_c=True, command=None):

    """Compare the system results to the gold standard results. score_type is an
    optional parameter to use min, max, or average score for thresholding.
    count restricts scores to terms that appear in <count> or more documents."""

    edata = EvalData(eval_file, system_file, score_type, count)
    pra = PRA(edata.d_eval_phr2label, edata.d_system_phr2score, threshold,
              term_type=term_type, term_filter=term_filter, debug_c=debug_c)

    s_log = codecs.open(log_file, "w", 'utf-8')
    if command is not None:
        s_log.write('$ ' + command.replace('--', "\n     --") + "\n\n")
    s_log.write("threshold    =  %s\n" % threshold)
    s_log.write("term_type    =  %s\n" % term_type)
    s_log.write("term_filter  =  %s\n\n" % term_filter)
    s_log.write("terms in gold standard:        %4d\n" % len(pra.d_eval))
    s_log.write("terms in system response:      %4d\n" % len(pra.d_system))
    s_log.write("terms in both (the overlap):   %4d\n" % pra.count_terms_in_overlap())
    s_log.write("terms after term_type:         %4d\n" % pra.eval_terms_count1)
    s_log.write("terms after term_filter:       %4d\n\n" % pra.eval_terms_count2)
    pra.pp_counts_long(s_log)
    #pra.log_missing_eval_phrases(s_log)
    s_log.write("\nList of used gold labels and system responses:\n\n")
    for (x, y, name) in (('n', 'y', 'false positives'), ('y', 'n', 'false negatives'),
                         ('y', 'y', 'true positives'), ('n', 'n', 'true negatives')):
        s_log.write("$ grep -e '^%s' %s | grep '|%s|'   # to get %s\n" % (x, os.path.basename(log_file), y, name))
    s_log.write("\ngold\tsystem\tscore\tterm\n\n")
    for score in pra.eval_terms_scores: s_log.write(score)
    s_log.close()

    print pra.results_string(),
    return pra.results_string()



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
    
