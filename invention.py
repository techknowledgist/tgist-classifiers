# -*- coding: utf-8 -*-

# PGA NOTE: moved to classifier directory 12/3/14 where other mallet functions reside

# TODO: add chinese invention types to functions:
# merge_scores
# eval_iclassify

# module for machine learning of invention keywords

# Create a mallet training file from two input files:
# annotation file based on phr_occ 
# features file based on phr_feats

# By default, we use the features file named invention.features to specify the set of features to use

# Note that the creation of unlabeled files for annotion is done by key.sh and key.py

# invention.create_mallet_training_file(annot_file, phr_feats_file, mallet_training_file)
# fa = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/key.ta.20130510.lab"
# ff = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/keyfeats.ta.20130509.dat"
# fo = "/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/data/workspace/i.train.mallet"

# to test:
# invention.imallet() to create a mallet instance file for training
# invention.itrain() to create a model

# next: run model over a file (1st 30 chunks from title/abstract) 

# invention.create_mallet_training_file(fa, ff, fo)

import commands
import os
import sys
import config
import mallet
import codecs
from collections import defaultdict
from ontology.utils.file import get_year_and_docid, open_input_file

from signal import signal, SIGPIPE, SIG_DFL 

# Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
# Without this, we get some "broken pipe" messages in the output.
signal(SIGPIPE,SIG_DFL) 


d_chunkid2label = {}
# track counts of annotations and mallet lines output to verify that there is a match
chunkid2label_count = 0
output_count = 0


# These functions mirror methods of mallet.MalletTraining
# Eventually, invention training should be merged fully into mallet.py
# The main difference is how it merges annotations with feature data (in this case phr_feats)
def populate_feature_dictionary(features):

    """Populate the d_filter_feat dictionary with all the features used for the
    model. If no features are added, the dictionary will remain empty, which
    downstream will be taken to mean that all features will be used. The argument can
    either be a filename or an identifier that points to a file in the features
    directory."""

    d_filter_feat = {}

    try:
        if os.path.isfile(features):
            filter_filename = features
        else:
            filter_filename = os.path.join("features", features + ".features")
        with open(filter_filename) as s_filter:
            print "[MalletTraining] Using features file: %s" % filter_filename
            for line in s_filter:
                feature_prefix = line.strip()
                d_filter_feat[feature_prefix] = True
    except IOError as e:
        print "[MalletTraining] No features file found: %s" % filter_filename

    return(d_filter_feat)

def remove_filtered_feats(feats, d_filter_feat):
    """Given a list of features, return a line with all non-filtered features removed
    We filter on the prefix of the feature (the part before the '='). Return the
    original line if the features dictionary is empty."""
    if not d_filter_feat:
        return feats
    return [f for f in feats if d_filter_feat.has_key(f.split("=")[0])]



# NOTE: since we use | as a separator, we cannot allow a phrase to contain a "|"
# We replace it with "!".  Of course it would be better to have checked for this case
# earlier in the process and dealt with it there.
def make_instance_key(chunkid, year, phrase):
    key = year + "|" + chunkid + "|" +  phrase.replace(" ", "_").replace("|", "!")
    return(key)

# Note that it is a good idea after creating the training file to check the feature counts to
# make sure that there are no misspellings among them.  e.g. list a sorted count:
# cat i.train.mallet | cut -f2 -d" " | sort | uniq -c | sort -nr

# To create the mallet file, we need to combine the category label in the annotation file (.lab) with the features
# for the same chunk in the phr_feats file.  We also need to remove any features that are not included in the 
# list of features in the .features file specified in the features parameter.  For example, we don't want to include
# the sent_loc feature.  Currently, the features files reside in the /features subdirectory below the code directory.
def create_mallet_training_file(annot_file, phr_feats_file, mallet_training_file, features=None, version="1", xval=0):

    global chunkid2label_count
    global output_count

    d_filter_feat = {}

    s_mallet_training = codecs.open(mallet_training_file, "w", encoding='utf-8')
    s_annot = codecs.open(annot_file, encoding='utf-8')
    s_feats = codecs.open(phr_feats_file, encoding='utf-8')

    if features is not None:
        d_filter_feat = populate_feature_dictionary(features)

    #print "chunkid2label_count: %i" % chunkid2label_count

    # Create a dictionary storing each non-empty label by chunkid
    for line in s_annot:
        #print "[invention] label line is: %s" % line
        line = line.strip("\n")
        (label, chunkid, year, phrase, sentence) = line.split("\t")
        if label != "":
            d_chunkid2label[chunkid] = label
            chunkid2label_count += 1

    #print "[invention]Found %i labels." % (chunkid2label_count)

    # Now output a mallet training features instance for each labeled feature instance in phr_feats file
    for line in s_feats:
        line = line.strip("\n")
        l_data = line.split("\t")
        chunkid = l_data[0]
        phrase = l_data[2]
        l_feats = l_data[3:]
        # keep only features in the .features file
        l_feats = remove_filtered_feats(l_feats, d_filter_feat)

        key = make_instance_key(chunkid, year, phrase)
        if d_chunkid2label.has_key(chunkid):
            #print "[invention]chunkid: %s, label: %s" % (chunkid, d_chunkid2label[chunkid])
            instance_line = key + " " + d_chunkid2label[chunkid] + " " + " ".join(l_feats) + "\n"
            output_count += 1
            s_mallet_training.write(instance_line)
    if chunkid2label_count != output_count:
        print("[invention]WARNING: mismatch between labels and instances created.")
    print("[invention]Found %i labels, created %i feature instances") % (chunkid2label_count, output_count)

    s_annot.close()
    s_feats.close()
    s_mallet_training.close()

# root_dir specifies where the pipeline processed files are located (ie. the d3_phr_feats dir)
# file_list_file provides a list of the files (in the root_dir/) that we want to classify.
#    This is a tab separated file with year, full path, year/filename.xml
# iclassify_dir is where the mallet files will go.  The model must be in this dir.  Intermediate and output
#    files will be placed here under standard naming conventions.
def create_mallet_classify_file(root_dir, file_list_file, iclassify_dir,
                                features=None, version="1", verbose=False):

    global chunkid2label_count
    global output_count

    num_lines_output = 0
    mallet_classify_file = os.path.join(iclassify_dir, "iclassify.mallet")

    # open output file (this will be the mallet instance file to be classified)
    s_mallet_classify = codecs.open(mallet_classify_file, "w", encoding='utf-8')

    # loop through files in file_list_file
    s_file_list = open(file_list_file)

    # keep track of the number of annotations we throw away because they are beyond the first 30
    overflow = 0

    for line in s_file_list:
        line = line.strip("\n")
        # get the date/filename portion of path
        l_line_fields = line.split("\t")
        rel_file = l_line_fields[2]
        phr_feats_file = os.path.join(root_dir, 'data', 'd3_phr_feats', '01', 'files', rel_file)
        if verbose:
            print "[invention]opening phr_feats: %s" % phr_feats_file
        #s_phr_feats = codecs.open(phr_feats_file, encoding='utf-8')
        # handle compressed or uncompressed files
        s_phr_feats = open_input_file(phr_feats_file)
        # keep first 30 chunks, if they are from title/abstract
        num_chunks = 0
        for line in s_phr_feats:
            if num_chunks >= 30:
                overflow += 1
                break
            else:
                line = line.strip("\n")
                # check whether line is in title or abstract
                if line.find("TITLE") > 0 or line.find("ABSTRACT") > 0:
                    l_data = line.split("\t")
                    chunkid = l_data[0]
                    year = l_data[1]
                    phrase = l_data[2]
                    l_feats = l_data[3:]
                    key = make_instance_key(chunkid, year, phrase)

                    # add dummy "n" as class label
                    instance_line = key + " n " + " ".join(l_feats) + "\n"
                    output_count += 1
                    s_mallet_classify.write(instance_line)


                    num_chunks += 1
                    num_lines_output += 1

        s_phr_feats.close()

    print "[invention]%i lines written to %s" % (num_lines_output, mallet_classify_file)

    s_mallet_classify.close()
    s_file_list.close()


# Given a mallet training file, create a model
# invention equivalent to patent_utraining_data3
# We assume mallet training file already exists, with labels
# invention parallel to patent_training_data3()
def patent_invention_train(mallet_file,
                           features="invention", version="1", xval=0,
                           verbose=False, stats_file=None, training_portion=0):

    """Wrapper around mallet.py functionality to create a classifier
    model. The .mallet training instances file must exist and full path passed in.  Other files needed
    for mallet processing will be placed in the same directory (train_output_dir).
    and creates an instance of MalletTraining class to do the rest: 
    creating the .vectors file from the mallet file, and
    creating the model.
    """

    #d_phr2label = load_phrase_labels3(annotation_file, annotation_count)
    train_output_dir = os.path.dirname(mallet_file)
    mconfig = mallet.MalletConfig(
        config.MALLET_DIR, 'itrain', 'iclassify', version, train_output_dir, '/tmp',
        classifier_type="MaxEnt", number_xval=xval, training_portion=training_portion,
        prune_p=False, infogain_pruning="5000", count_pruning="3")
    mtr = mallet.MalletTraining(mconfig, features)
    mtr.write_train_mallet_vectors_file()
    mtr.mallet_train_classifier()
    # todo: add the following line
    ###write_training_statistics(stats_file, mtr)


def patent_invention_classify(mallet_file, train_dir="", test_dir="",
                           features="invention", version="1",
                           verbose=False, stats_file=None):

    if train_dir == "":
        train_dir = os.path.dirname(mallet_file)
    if test_dir == "":
        test_dir = os.path.dirname(mallet_file)
    mallet_config = mallet.MalletConfig(config.MALLET_DIR, "itrain", "iclassify",
                                        version, train_dir, test_dir, classifier_type="MaxEnt")
    mallet_classifier = mallet.MalletClassifier(mallet_config)
    #mallet_classifier.write_mallet_vectors_file()
    mallet_classifier.mallet_classify()
    if verbose:
        print "[patent_invention_classify]After applying classifier"
    #return(mallet_config)


# Retrieve the title of a patent, which is on line 2 of the files in the txt directory.
# here root_path is the directory for the txt file, up to the year subdirectory.
# filename is year/file.xml
def patent_title(root_path, filename):
    file_path = os.path.join(root_path, filename)
    # files may be compressed or not
    compressed_file = file_path + ".gz"
    qualifier = ""
    if os.path.exists(compressed_file):
        cat_version = "zcat"
        qualifier = ".gz"
    else:
        cat_version = "cat"
    cat_command = cat_version + " " + file_path + qualifier + " | head -2 | tail -1"
    #print "cat_command: %s" % cat_command
    # note we have to decode the title byte-string into unicode so that it can be
    # appended to a unicode string later and the entire string will be encodable from utf-8.
    title = commands.getoutput(cat_command).decode('utf-8')
    #print "title: %s" % title
    return(title)

# take a list and create a string separated by "," with "_" replaced by blank.
def list_to_csv_string(l_items):
    return( ", ".join(l_items).replace("_", " "))

# Output a human readable summary of title and keyterms for a patent to an open stream
def output_doc_summary(doc, title, d_label2terms, s_merged):
    s_merged.write(  "title: [%s]%s\n" % (doc, title))
    s_merged.write(  "invention type: %s\n" % list_to_csv_string(d_label2terms["t"]))
    s_merged.write( "invention descriptors: %s\n" % list_to_csv_string(d_label2terms["i"]))
    s_merged.write( "contextual terms: %s\n" % list_to_csv_string(d_label2terms["m"]))
    s_merged.write( "components/attributes: %s\n" % list_to_csv_string(d_label2terms["c"]))
    #s_merged.write( "related: %s\n" % list_to_csv_string(d_label2terms["r"]))
    s_merged.write(  "\n" )

def output_cat_summary(doc, d_label2terms, s_cat):
    for cat in ["t", "i", "m", "c", "r"]:
        if d_label2terms.has_key(cat):
            for phr in d_label2terms[cat]:
                #print "cat: %s" % cat
                #print "phr: %s" % ( phr)
                s_cat.write("%s\t%s\n" % (phr, cat))

def output_raw_eval_summary(last_doc, l_iclassify_all, d_key2chunkinfo_manual, s_raw_eval):
    for ci_iclassify in l_iclassify_all:
        # There may be docs in the system data output that are not included in the manual evaluation set,
        # so check that the chunk id exists before trying to include it in the output
        if d_key2chunkinfo_manual.has_key(ci_iclassify.key):
            #print "output_raw: key found: %s: " % ci_iclassify.key
            ci_manual = d_key2chunkinfo_manual[ci_iclassify.key]

            # adjust the manual category for i(nvention) if the adjusted cat is "t(ype)"
            if ci_iclassify.cat == "t":
                ci_manual.cat = "t"

            s_raw_eval.write("%s %s\t%s\t%s\t%s\n" % (ci_manual.cat, ci_iclassify.cat, ci_manual.key, ci_manual.chunk , ci_manual.sentence))

# outputs the summary info but only for the first occurrence of a term in each doc
def output_adj_eval_summary(last_doc, l_iclassify_first, d_key2chunkinfo_manual,  s_adj_eval):
    for ci_iclassify in l_iclassify_first:
        # There may be docs in the system data output that are not included in the manual evaluation set,
        # so check that the chunk id exists before trying to include it in the output
        if d_key2chunkinfo_manual.has_key(ci_iclassify.key):
            
            ci_manual = d_key2chunkinfo_manual[ci_iclassify.key]

            # adjust the manual category for i(nvention) if the adjusted cat is "t(ype)"
            if ci_iclassify.cat == "t":
                ci_manual.cat = "t"
            s_adj_eval.write("%s %s\t%s\t%s\t%s\n" % (ci_manual.cat, ci_iclassify.cat, ci_manual.key, ci_manual.chunk , ci_manual.sentence))



# Merge keyword info for each doc (in label_file) so that each keyword has only
# one label (the first that occurs in the doc).
# source_path is location of source txt directory (needed only to print the text of the title)
# iclassify_dir is location of invention classification output
# label_file is name of label file, by convention iclassify.<classifier>.label
# format: 2004|US6776488B2.xml_0|camera_crane     i       0.866316499353
#
# creates iclassify.<classifier>.label.merged
# and iclassify.<classifier>.label.cat
# invention.merge_scores  

def merge_scores(source_path, iclassify_path, label_file, lang="en"):
    if lang == "en":
        l_invention_type = ['assembly', 'means', 'compositions', 'composition',
                            'method', 'methods', 'apparatus', 'system', 'use',
                            'process', 'device', 'technique']
    elif lang == "cn":
        l_invention_type = ["系统", "装置", "方法", "特点", "特征", "包括", "基于", "属于"]
    else:
        print "[merge_scores]unknown language: %s.  Exiting." % lang
        return()


    # put invention types into a dictionary for easy testing
    d_invention_type = {}
    d_invention_type = d_invention_type.fromkeys(l_invention_type)

    # full path of label and merged (output) file

    # todo: This should be set to the proper location for invention data
    # once we decide where that should be.  Use the workspace dir for now.
    # MV: this first one used to have data/workspace appended
    iclassify_dir = iclassify_path
    label_path = os.path.join(iclassify_dir, label_file)
    output_path = os.path.join(iclassify_dir, label_file + ".merged")
    cat_path = os.path.join(iclassify_dir, label_file + ".cat")

    s_labels = codecs.open(label_path, encoding='utf-8')
    s_merged = codecs.open(output_path, "w", encoding='utf-8')
    s_cat = codecs.open(cat_path, "w", encoding='utf-8')

    last_doc = ""
    last_phrase = ""

    # for each doc, track which phrases have been seen
    d_seen = {}
    d_label2terms = defaultdict(list)
    
    title = ""
    year = ""
    last_year = ""

    # format of label file is:
    # 1994|US5318556A.xml_0|fluid_bag i       0.864621951173

    # The same phrase can occur multiple times with different labels.  We
    # will choose the first label (heuristic).
    # The label i(nvention) includes some invention type terms.  We will detect
    # these and relable them as type.  We will also look at the last term in a 
    # multiword i phrase to see if it contains a type term (in case one does not occur
    # independently.
    line_no = 1
    for line in s_labels:
        #print "starting line: %i " % line_no
        line_no += 1

        #line = line.decode('utf-8').strip("\n")
        line = line.strip("\n")
        # get out all the pieces
        (key, label, score) = line.split("\t")
        (year, chunkid, term) = key.split("|")
        (doc, chunk_no) = chunkid.split("_")
        ###print "doc: %s, last_doc: %s, chunk_no: %s" % (doc, last_doc, chunk_no)
        if doc != last_doc:
            # print out the summary
            # Don't include the year if it has defaulted to 9999 (meaning no year subdirectory exists)
            if last_year == "9999":
                txt_path = os.path.join(source_path, 'data', 'd1_txt', '01', 'files')
            else:
                # include the year in the directory path for the txt files
                txt_path = os.path.join(source_path,
                                        'data', 'd1_txt', '01', 'files', last_year)
            title = patent_title(txt_path, last_doc)
            if last_doc != "":
                output_doc_summary(last_doc, title, d_label2terms, s_merged)
                output_cat_summary(last_doc, d_label2terms, s_cat)
            last_doc = doc
            last_year = year
            d_seen = {}
            d_label2terms = defaultdict(list)
        
        # more info on the current doc
        # if term hasn't been seen, store under its first label
        if not d_seen.has_key(term):
            d_seen[term] = True
            if d_invention_type.has_key(term):
                # set the label to "t"
                label = "t"
            d_label2terms[label].append(term)

    # for end of file...
    if last_year == "9999":
        txt_path = os.path.join(source_path, 'data', 'd1_txt', '01', 'files')
    else:
        # include the year in the directory path for the txt files
        txt_path = os.path.join(source_path, 'data', 'd1_txt', '01', 'files', last_year)

    title = patent_title(txt_path, last_doc)
    output_doc_summary(last_doc, title, d_label2terms, s_merged)
    output_cat_summary(last_doc,  d_label2terms, s_cat)

    s_labels.close()
    s_merged.close()
    s_cat.close()


# read in data from manual and iclassify files, index by key (file + chunk-id)
# generate raw comparison of categories for chunks.
# Then extract invention type terms and post-process the other iclassify chunks by keeping only the first
# label for each chunk in the file.  Compare these to the manual annotations using the key index.
                            
class chunkinfo:
    def __init__(self, key, chunk, cat, sentence):
        self.key = key
        self.chunk = chunk
        self.cat = cat
        self.sentence = sentence

# output two files containing raw and adjusted label pairs for system output compared to manual labels.
# The adjusted label pairs make two adjustments:
# 1. add a "t" label for invention terms that refer to generic invention words (ie. invention types, like
# system or method"
# 2. if a term appears multiple times in a single document, use the first label as the assigned label for 
# all occurrences of the term.
# The adjusted file only contains output for the first occurrence of a term.  Hence it will be smaller than
# the raw file.
# output is in the form:
# gold label, system label, unique id for term, term, sentence
# i m     US7241624B2.xml_1       biochips        Dendrimer-based DNA extraction methods...
# input for manual (gold label) .lab or .mcipo file is in the form:
# m       US7844571B2.xml_5       9999    image data      To increase the efficiency of...
# input for the iclassify_file (system output) is .label, of the form:
# 9999|US7241624B2.xml_17|tips    m       0.713193784696 

def eval_iclassify(manual_path, manual_file, iclassify_path, iclassify_file, lang="en"):
    if lang == "en":
        l_invention_type = ['assembly', 'means', 'compositions', 'composition', 'method', 'methods', 'apparatus', 'system', 'use', 'process', 'device', 'technique']
    elif lang == "cn":
        l_invention_type = ["系统", "装置", "方法", "特点", "特征", "包括", "基于", "属于"]
    else:
        print "[eval_iclassify]unknown language: %s.  Exiting." % lang
        return()

    # put invention types into a dictionary for easy testing
    d_invention_type = {}
    d_invention_type = d_invention_type.fromkeys(l_invention_type)

    d_key2chunkinfo_manual = {}
    d_key2chunkinfo_iclassify = {}

    # todo: This should be set to the proper location for invention data
    # once we decide where that should be.  Use the workspace dir for now.
    iclassify_filename = iclassify_path + iclassify_file
    manual_filename = manual_path + manual_file

    eval_filename = iclassify_filename + ".raw_eval"
    adj_filename = iclassify_filename + ".adj_eval"

    s_manual = codecs.open(manual_filename, encoding='utf-8')
    s_iclassify = codecs.open(iclassify_filename, encoding='utf-8')
    s_raw_eval = codecs.open(eval_filename, "w", encoding='utf-8')
    s_adj_eval = codecs.open(adj_filename, "w", encoding='utf-8')
                            
    last_doc = ""
    last_phrase = ""

    # load the manual annotations into a dict keyed by chunk_id
    # we'll include any annotated line, even if it exceeds 30 per doc, since
    # the iclassify file will only include the first 30 and we will use that to 
    # control the output.
    manual_count = 0

    for line in s_manual:
        line = line.strip("\n")
        # i       US20020078204A1.xml_0   2002    method  <np> Method </np> and system for controlling presentation of information to a user based on the user 's condition
        # make sure line is not blank
        if line != "":
            (cat, key, year, chunk, sent) = line.split("\t")
            #key = key

            if cat != "":
                ci = chunkinfo(key, chunk, cat, sent)
                #print "in s_manual: adding key: %s" % key
                d_key2chunkinfo_manual[key] = ci
                manual_count += 1

    print "%i keys in d_key2chunkinfo_manual:" % manual_count
    #for key in d_key2chunkinfo_manual.keys():
    #    print key


    # Now process the system classifications (iclassify)
    # for each doc, track which phrases have been seen
    d_seen = {}
    #d_key2chunkinfo_iclassify = {}
    # list of chunkinfo in order for a single doc
    l_iclassify_all = [] 
    l_iclassify_first = [] 

    # format of label file is:
    # 1994|US5318556A.xml_0|fluid_bag i       0.864621951173

    # The same phrase can occur multiple times with different labels.  We
    # will choose the first label (heuristic).
    # The label i(nvention) includes some invention type terms.  We will detect
    # these and re-label them as type.  We will also look at the last term in a 
    # multiword i phrase to see if it contains a type term (in case one does not occur
    # independently.
    line_no = 1
    for line in s_iclassify:
        #print "starting line: %i " % line_no
        line_no += 1

        #line = line.decode('utf-8').strip("\n")
        line = line.strip("\n")
        # get out all the pieces
        #print "line in s_iclassify: %s" % line
        (key, label, score) = line.split("\t")
        (year, chunkid, term) = key.split("|")
        (doc, chunk_no) = chunkid.split("_")
        ###print "doc: %s, last_doc: %s, chunk_no: %s" % (doc, last_doc, chunk_no)
        # when we finish all lines in a document, output info for that doc

        # switch label from i to t if the invention term is a generic type
        if d_invention_type.has_key(term):
            # set the label to "t"
            label = "t"

        ci = chunkinfo(chunkid, term, label, "")
        
        if doc != last_doc:
            # print out the data for the previous doc when we reach the first line of a new one
            # initially last_doc is "".
            if last_doc != "":
                output_raw_eval_summary(last_doc, l_iclassify_all, d_key2chunkinfo_manual, s_raw_eval)
                output_adj_eval_summary(last_doc, l_iclassify_first, d_key2chunkinfo_manual,  s_adj_eval)

            last_doc = doc
            d_seen = {}
            # remember to clear the list 
            l_iclassify_all = []
            l_iclassify_first = []

        # While we are still processing the same doc, save up info
        # if term hasn't been seen, store under its first label
        if not d_seen.has_key(term):
            d_seen[term] = True
            l_iclassify_first.append(ci)

        # we add the data to the raw list regardless of whether the term has appeared before
        l_iclassify_all.append(ci)

    # for end of file...
    output_raw_eval_summary(last_doc, l_iclassify_all, d_key2chunkinfo_manual, s_raw_eval)
    output_adj_eval_summary(last_doc, l_iclassify_first, d_key2chunkinfo_manual,  s_adj_eval)

    print "Number lines in iclassify file: %i" % line_no

    s_manual.close()
    s_iclassify.close()
    s_raw_eval.close()
    s_adj_eval.close()

def cmtf():
    fa = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/key.ta.20130510.lab"
    ff = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/keyfeats.ta.20130509.dat"
    fo = "/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/data/workspace/itrain.mallet"
    features_file = "invention"
    create_mallet_training_file(fa, ff, fo, features_file)
    print ("Output: %s" % fo)



# for cs_2002_subset

def cmtf_cs():
    # new labeled file .lab
    fa = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/cs_2002/key.ta.20130720.lab"


    #ff = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/cs_2002/keyfeats.ta.20130812.dat"
    # modified features and reran the phr_feats step
    # use the new .dat file with updated features (prev_V2)
    ff = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/cs_2002/keyfeats.ta.20130813.dat"

    # put it in workspace and move it to a subdir later
    fo = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/itrain.mallet"
    features_file = "invention"
    create_mallet_training_file(fa, ff, fo, features_file)
    print ("Output: %s" % fo)

# test functions on sample data
def cmcf():
    root_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/"

    # loc of phr_feats files:  data/d3_phr_feats/01/files/
    # each line contains: date full_path_spec path_spec_from_root_dir
    file_list_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/config/key_test_files.txt"

    iclassify_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/data/workspace/"
    create_mallet_classify_file(root_dir, file_list_file, iclassify_dir, "invention", "1")

# To set up the data to classify the speech subset:
# Make a directory for this application.  Within it, create config and data subdirectories
# Within data, create workspace subdirectory
# Put the filelist into config subdirectory:
# cp /home/j/marc/Desktop/fuse/code/patent-classifier/ontology/creation/data/patents/201306-speech-recognition/config/files.txt /home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/config
# Create a mini set for testing:
# head -10 files.txt > files.10.txt
# Go to the directory where we built the invention classifier model:
# cd /home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/data/workspace
# Copy the mallet training files to the new target directory:
# cp itrain.* /home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/data/workspace

# 
def cmcf_speech():
    root_dir = "/home/j/marc/Desktop/fuse/code/patent-classifier/ontology/creation/data/patents/201306-speech-recognition/"

    # loc of phr_feats files:  data/d3_phr_feats/01/files/
    # each line contains: date full_path_spec path_spec_from_root_dir
    file_list_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/config/files.txt"

    iclassify_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/data/workspace/"
    create_mallet_classify_file(root_dir, file_list_file, iclassify_dir, "invention", "1")

# TBD
def cmcf_chinese():
    root_dir = "/home/j/marc/Desktop/fuse/code/patent-classifier/ontology/creation/data/patents/201306-speech-recognition/"

    # loc of phr_feats files:  data/d3_phr_feats/01/files/
    # each line contains: date full_path_spec path_spec_from_root_dir
    file_list_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/config/files.txt"

    iclassify_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/data/workspace/"
    create_mallet_classify_file(root_dir, file_list_file, iclassify_dir, "invention", "1")


def cmcf_cs():
    root_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/"

    # loc of phr_feats files:  data/d3_phr_feats/01/files/
    # each line contains: date full_path_spec path_spec_from_root_dir
    file_list_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/config/files.txt"

    iclassify_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/"
    create_mallet_classify_file(root_dir, file_list_file, iclassify_dir, "invention", "1")

# parallel corpus
def cmcf_par():
    root_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/"

    # loc of phr_feats files:  data/d3_phr_feats/01/files/
    # each line contains: date full_path_spec path_spec_from_root_dir
    file_list_file = "/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/config/files.txt"

    iclassify_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/data/workspace/"
    create_mallet_classify_file(root_dir, file_list_file, iclassify_dir, "invention", "1")

def itrain():
    patent_invention_train("/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/data/workspace/itrain.mallet")

def itrain_cs():
    patent_invention_train("/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/itrain.mallet")


# make sure an itrain.model file is in the workspace directory before running iclassify().  
# The model can be copied from another workspace where training was done.
# e.g. cp /home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/cs_2002_20130723_model/itrain.model .
def iclassify_cs():
    patent_invention_classify("/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/iclassify.mallet")

def iclassify_par():
    patent_invention_classify("/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/data/workspace/iclassify.mallet")

def iclassify():
    patent_invention_classify("/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/data/workspace/iclassify.mallet")

def iclassify_speech():
    patent_invention_classify("/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/data/workspace/iclassify.mallet")

# obsolete
def ms():
    root_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/ts500/"
    label_file = "iclassify.MaxEnt.label"
    merge_scores(root_path, label_file)

# Before running this, you need to create the .label file from .out.  See the command line below.
# import invention
# invention.ms_speech()
def ms_speech():
    # path to corpus where the txt files reside (for getting titles)
    source_path = "/home/j/marc/Desktop/fuse/code/patent-classifier/ontology/creation/data/patents/201306-speech-recognition/"
    # path to corpus where mallet files and output reside (which could be the same as or different from source_path)
    iclassify_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/"
    label_file = "iclassify.MaxEnt.label"
    merge_scores(source_path, iclassify_path, label_file)

# invention.ms_cs()
def ms_cs():
    # path to corpus where the txt files reside (for getting titles)
    source_path = "/home/j/marc/Desktop/fuse/code/patent-classifier/ontology/creation/data/patents/data/patents/201306-computer-science"

    # path to corpus where mallet files and output reside (which could be the same as or different from source_path)
    iclassify_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/"
    label_file = "iclassify.MaxEnt.label"
    merge_scores(source_path, iclassify_path, label_file)

# invention.ms_par()
def ms_par():
    # path to corpus where the txt files reside (for getting titles)
    source_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/"

    # path to corpus where mallet files and output reside (which could be the same as or different from source_path)
    iclassify_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/"
    label_file = "iclassify.MaxEnt.label"
    merge_scores(source_path, iclassify_path, label_file)


# invention.eval_cs()
# manual_file is hand labeled annotations in the form:
# US20020040345A1.xml_16   c       2002    card    A home banking system for receiving bank service...

# iclassify_file is machine labeled file of the form:
# 2002|US6470395B1.xml_0|method   i       0.980269938379


def eval_cs():
    manual_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/"
    iclassify_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/data/workspace/"
    manual_file = "key.ta.20130723.lab"
    iclassify_file = "iclassify.MaxEnt.label"
    eval_iclassify(manual_path, manual_file, iclassify_path, iclassify_file)

# evaluate the parallel corpus output (4 docs)
def eval_para():
    manual_path = "/home/j/anick/patent-classifier/ontology/annotation/en/invention/mini_parallel/"
    iclassify_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/parallel/us/data/workspace/"
    manual_file = "key.ta.201307.lab"
    iclassify_file = "iclassify.MaxEnt.label"
    eval_iclassify(manual_path, manual_file, iclassify_path, iclassify_file)


# note: to create .label from .out
# run this in workspace directory
# cat iclassify.MaxEnt.out | egrep -v '^name' | egrep '\|.*\|' | python /home/j/anick/patent-classifier/ontology/creation/invention_top_scores.py > iclassify.MaxEnt.label

# for the speech data:
# cat iclassify.MaxEnt.out | egrep -v '^name' | egrep '\|.*\|' | python /home/j/anick/patent-classifier/ontology/creation/invention_top_scores.py > iclassify.MaxEnt.label


# todo: fix cat: broken pipe errors.  Currently, you can remove them from output using;
# cat iclassify.MaxEnt.label.merged | sed -e 's/^cat:.*$//' > iclassify.MaxEnt.label.merged2

def test_encoding():
    in_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/data/workspace/one_fifth.txt"
    out_path = "/home/j/anick/patent-classifier/ontology/creation/data/patents/speech_subset/data/workspace/one_fifth.out"
    s_in = codecs.open(in_path, encoding='utf-8')
    s_out = codecs.open(out_path, "w", encoding='utf-8')

    d_test = {}

    for line in s_in:
        #line = line.decode('utf-8').strip("\n")
        line = line.strip("\n")
        l_line = line.split("\t")
        oline = " ".join(l_line) + "the end"

        d_test[oline] = True
        for key in d_test.keys():
            s_out.write(key)

# Below are functions for creating a Chinese keyterm classifier
 
# To create a training model for chinese keyterms, once we have an annotated file, we need to generate a file which contains
# the phr_feats data for all files annotated.  This is what we did for the Chinese data (12/3/14).
# for chinese annotated chunk files, get the corresponding phr_feats files and concatenate their contents into a single file
# After running this function, to concatenate the contents of the list of phr_feats files, run from output dir 
# (/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/):
# { xargs zcat < files.ln-cn-all-600k.all.phr_feats.filelist ; } > ln-cn-all-600k.all.phr_feats.dat

#This creates in out_path a list of the full paths for the source phr_feats files in the fuse corpus directory
def get_cn_phr_feats_data():
    # list of 100 files to be used in annotation
    in_path = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/files.ln-cn-all-600k.all.tail.20"
    # phr_feats data for those 100 files
    out_path = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/files.ln-cn-all-600k.all.phr_feats.filelist"

    # /home/j/corpuswork/fuse/FUSEData/corpora/ln-cn-all-600k/subcorpora/1998/data/d3_phr_feats/01/files/2004
    file_path_prefix = "/home/j/corpuswork/fuse/FUSEData/corpora/ln-cn-all-600k/subcorpora/"
    file_path_infix = "/data/d3_phr_feats/01/files/"

    
    s_in = codecs.open(in_path, encoding='utf-8')
    s_out = codecs.open(out_path, "w", encoding='utf-8')

    file_no = 0
    for line in s_in:
        file_no += 1
        line = line.strip("\n")
        (app_year, source_path, year_file) = line.split("\t")
        file_path = file_path_prefix + str(app_year) + file_path_infix + year_file + ".gz"
        print "[%i]file path: %s" % (file_no, file_path)
        s_out.write("%s\n" % file_path)
        print "Output in %s" % out_path
    s_in.close()
    s_out.close()

# create a mallet training file by merging annotation file and phr_feats data 
# cmtf = create mallet training file
# .mcipo is the subset of the annotated data labeled mcipo, leaving out any ambiguous labels.

def cmtf_chinese():
    fa = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn/cn.annotate.inventions.lab.txt.mcipo"
    ff = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn/ln-cn-all-600k.all.phr_feats.dat"
    fo = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn/itrain.mallet"
    features_file = "/home/j/anick/patent-classifier/ontology/classifier/features/invention.features"
    create_mallet_training_file(fa, ff, fo, features_file)
    print ("Output: %s" % fo)

# invention.itrain_chinese()
def itrain_chinese():
    mallet_file = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn/itrain.mallet"
    patent_invention_train(mallet_file, training_portion=.75)

"""
# To do Chinese evaluation
# in /home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn75
# create training file using 75% of the annotated files (75 out of 100)
# We will evaluate it on the remaining 25%

# split the data
# in /home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms
head -75 files.ln-cn-all-600k.all.phr_feats.filelist > files.ln-cn-all-600k.75.phr_feats.filelist
tail -25 files.ln-cn-all-600k.all.phr_feats.filelist > files.ln-cn-all-600k.25.phr_feats.filelist

# separate the .mcipo data into 2 files based on the 75/25 file split
head -75 files.ln-cn-all-600k.all.phr_feats.filelist > files.ln-cn-all-600k.75.phr_feats.filelist
tail -25 files.ln-cn-all-600k.all.phr_feats.filelist > files.ln-cn-all-600k.25.phr_feats.filelist
# find the line number of the 76th file in .mcpio (first file in files.ln-cn-all-600k.25.phr_feats.filelist)
grep -n CN101060929A cn.annotate.inventions.lab.txt.mcipo | more
# put the subsets into subdirectories cn75 and cn25
head -1736 cn.annotate.inventions.lab.txt.mcipo > cn75/cn.annotate.inventions.lab.txt.mcipo
tail -675 cn.annotate.inventions.lab.txt.mcipo > cn25/cn.annotate.inventions.lab.txt.mcipo

# Do the same for phr_feats data
grep -n CN101060929A ln-cn-all-600k.all.phr_feats.dat | head -2
head -89468 ln-cn-all-600k.all.phr_feats.dat > cn75/ln-cn-all-600k.all.phr_feats.dat
tail -21070 ln-cn-all-600k.all.phr_feats.dat > cn25/ln-cn-all-600k.all.phr_feats.dat
# in python, train a model using 75 docs
>>> import invention
>>> invention.cmtf_cn75() 
>>> invention.itrain_cn75()  
# create an iclassify file for the 25 test docs
invention.cmcf_cn25() 
# note that there may be a message indicating a count mismatch between the 2 files.  As long as this 
# is small, it can be ignored, since we discard a few annotations from .mcipo which don't fall into the 
# set of labels used.

# copy the model into the cn25 directory before running classification there
cp ../cn75/itrain.model .
# run classifier in python to create iclassify.MaxEnt.out
>>> invention.class_cn25()  

# to create .label from .out                                                                                                             
# run this in workspace directory (cn25)                                                                                                              
# cat iclassify.MaxEnt.out | egrep -v '^name' | egrep '\|.*\|' | python /home/j/anick/patent-classifier/ontology/creation/invention_top_scores.py > iclassify.MaxEnt.label

# Now do evaluation
>>> invention.eval_cn25() 
# To see the gold-system label pairs and their counts:
#  cut -f1 iclassify.MaxEnt.label.adj_eval | sort | uniq -c | sort -nr
"""

# invention.cmtf_cn75(train_labels="mci"):
# assume there is an annotation file whose extension is the set of labels to be used (e.g. mcipo or mci) for training 
def cmtf_cn75(train_labels="mcipo"):
    fa = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn75/cn.annotate.inventions.lab.txt." + train_labels
    ff = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn75/ln-cn-all-600k.all.phr_feats.dat"
    fo = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn75/itrain.mallet"
    features_file = "/home/j/anick/patent-classifier/ontology/classifier/features/invention.features"
    create_mallet_training_file(fa, ff, fo, features_file)
    print ("Output: %s" % fo)

# invention.itrain_cn75()
def itrain_cn75():
    mallet_file = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn75/itrain.mallet"
    patent_invention_train(mallet_file, training_portion=0)

# invention.cmcf_cn25(test_labels="mcipo"):
# assume there is an annotation file whose extension is the set of labels to be used (e.g. mcipo or mci) for testing 
def cmcf_cn25(test_labels="mcipo"):
    fa = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25/cn.annotate.inventions.lab.txt." + test_labels
    ff = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25/ln-cn-all-600k.all.phr_feats.dat"
    fo = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25/iclassify.mallet"
    features_file = "/home/j/anick/patent-classifier/ontology/classifier/features/invention.features"
    create_mallet_training_file(fa, ff, fo, features_file)
    print ("Output: %s" % fo)


# invention.class_cn25()
def class_cn25():
    mallet_file = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25/iclassify.mallet"
    patent_invention_classify(mallet_file, train_dir="", test_dir="",
                           features="invention", version="1",
                           verbose=False, stats_file=None)

# invention.eval_cn25()
def eval_cn25():
    manual_path = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25/"
    manual_file = "cn.annotate.inventions.lab.txt.mcipo"
    iclassify_path = "/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25/"
    iclassify_file = "iclassify.MaxEnt.label"
    language = "cn"
    eval_iclassify(manual_path, manual_file, iclassify_path, iclassify_file, lang=language)
    
"""
On 1/9/15 PGA moved the cn75 and cn25 directories created before fixing the double last_word bug into
/home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn/cn_model_mcipo_20150106
On 2/4/2015, did the same for cn/cn_model_mcipo_20150204, in order to use phr_feats data with a correction
that removes parens from tokens.  Previously a bug in Stanford tagger labeled some parens as N, causing them
to be included in terms.

How many term instances were affected by parens:
cat  ln-cn-all-600k.all.phr_feats.dat | cut -f3 | grep "[()]" | wc -l
10089
Moved the old .dat file (with the parens) into cn/cn_model_mcipo_20150204:
mv ln-cn-all-600k.all.phr_feats.dat cn/cn_model_mcipo_20150204

Note that in the file name, _mcipo_ indicates which labels were included

Then created new subdirectories cn75 and cn25 in /home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms
From that dir, ran 
{ xargs zcat < files.ln-cn-all-600k.all.phr_feats.filelist ; } > ln-cn-all-600k.all.phr_feats.dat

# Repeated the following steps using the new phr_feats data
# put the subsets into subdirectories cn75 and cn25
head -1736 cn.annotate.inventions.lab.txt.mcipo > cn75/cn.annotate.inventions.lab.txt.mcipo
tail -675 cn.annotate.inventions.lab.txt.mcipo > cn25/cn.annotate.inventions.lab.txt.mcipo

# Do the same for phr_feats data
grep -n CN101060929A ln-cn-all-600k.all.phr_feats.dat | head -2
head -89468 ln-cn-all-600k.all.phr_feats.dat > cn75/ln-cn-all-600k.all.phr_feats.dat
tail -21070 ln-cn-all-600k.all.phr_feats.dat > cn25/ln-cn-all-600k.all.phr_feats.dat
# in python, train a model using 75 docs
cd /home/j/anick/patent-classifier/ontology/creation
python2.7
>>> import invention
>>> invention.cmtf_cn75() 
>>> invention.itrain_cn75()  
# create an iclassify file for the 25 test docs
invention.cmcf_cn25() 
# note that there may be a message indicating a count mismatch between the 2 files.  As long as this 
# is small, it can be ignored, since we discard a few annotations from .mcipo which don't fall into the 
# set of labels used.

# copy the model into the cn25 directory before running classification there
cd /home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25
# from cn25 directory:
cp ../cn75/itrain.model .
# run classifier in python to create iclassify.MaxEnt.out
>>> invention.class_cn25()  

# to create .label from .out                                                                                                             
# run this in workspace directory (cn25)                                                                                                              
# cat iclassify.MaxEnt.out | egrep -v '^name' | egrep '\|.*\|' | python /home/j/anick/patent-classifier/ontology/creation/invention_top_scores.py > iclassify.MaxEnt.label

This gives us a .label file of the form:
2007|CN101060929A.xml_0|改性_催化剂_载体        i       0.939536099456
2007|CN101060929A.xml_2|磨耗性  m       0.536137197147
...


# Now do evaluation
>>> invention.eval_cn25() 
# To see the gold-system label pairs and their counts:
#  cut -f1 iclassify.MaxEnt.label.adj_eval | sort | uniq -c | sort -nr

# Next we run with training data that excludes labels o (other) and p (parse error)
# move up to directory /home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms
# NOTE!!!  The following line has a tab in the grep expression - this cannot be cut and pasted.  Type it in using
# ctl-V <tab>
# NOTE 2: The tab was unnecessary and was removed from the grep, so cut and paste will work now.
cat cn.annotate.inventions.lab.txt.mcipo | egrep -v '^[po]' > cn.annotate.inventions.lab.txt.mci
# verify it worked using wc.  This file should be smaller than the .mcipo file
wc -l cn.annotate.inventions.lab.txt.mci2105 cn.annotate.inventions.lab.txt.mci

head -1736 cn.annotate.inventions.lab.txt.mcipo | egrep -v '^[po]' > cn75/cn.annotate.inventions.lab.txt.mci
tail -675 cn.annotate.inventions.lab.txt.mcipo | egrep -v '^[po]' > cn25/cn.annotate.inventions.lab.txt.mci      

Note that both training and eval files will be smaller than in the mcipo case since certain labels were removed.

cp cn.annotate.inventions.lab.txt.mci cn75
cp cn.annotate.inventions.lab.txt.mcipo cn25
[anick@sarpedon keyterms]$ head -89468 ln-cn-all-600k.all.phr_feats.dat > cn75/ln-cn-all-600k.all.phr_feats.dat   
[anick@sarpedon keyterms]$ tail -21070 ln-cn-all-600k.all.phr_feats.dat > cn25/ln-cn-all-600k.all.phr_feats.dat 

# in python, train a model using 75 docs
# This will overwrite our previous files (generated for the mcipo model)
>>> import invention
>>> invention.cmtf_cn75(train_labels="mci")
>>> invention.itrain_cn75()  
///

This creates a gold label file (itrain.mallet) with all labels (of type mcipo).  
To create a version of the file with just mci labels, we
will remove the p and o labeled lines with grep 
[anick@sarpedon cn75]$ cp itrain.mallet itrain.mallet.mcipo
[anick@sarpedon cn75]$ cat itrain.mallet.mcipo | egrep -v " [po] " > itrain.mallet

Now train the model without the po labels using itrain.mallet
>>> invention.itrain_cn75()                                                                                               
# create an iclassify file for the 25 test docs.  Note we leave in the po labels for the test file even
# though they were not included in training data since we need to include them in the evaluation.
>>> invention.cmcf_cn25(test_labels="mcipo")
# note that there may be a message indicating a count mismatch between the 2 files.  As long as this 
# is small, it can be ignored, since we discard a few annotations from .mcipo which don't fall into the 
# set of labels used.

# copy the model into the cn25 directory before running classification there
cd /home/j/anick/patent-classifier/ontology/roles/data/annotation/keyterms/cn25
# from cn25 directory:
cp ../cn75/itrain.model .
# run classifier in python to create iclassify.MaxEnt.out
>>> invention.class_cn25()  

# to create .label from .out
# run this in workspace directory (cn25)
# cat iclassify.MaxEnt.out | egrep -v '^name' | egrep '\|.*\|' | python /home/j/anick/patent-classifier/ontology/creation/invention_top_scores.py > iclassify.MaxEnt.label

# Now do evaluation
>>> invention.eval_cn25() 
# To see the gold/system label pairs and their counts:
#  cut -f1 iclassify.MaxEnt.label.adj_eval | sort | uniq -c | sort -nr

TODO: Check on whether the gold data needs adjustment (setting the label to the value of the first occurrence)
Fix chunker to split on the tag for parenthesis.  20% of terms contain a paren!

"""
