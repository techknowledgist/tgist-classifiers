"""

Script to generate invention keyterm scores. 

This is based on inventions.py, but the interface and some methods are tweaked
to look more like the run_tclassify.py. Overall, much of the bookkeeping done in
the latter is not done here yet.

From the command line, you can now run this script as a classifier. The most
typical invocation is:

    $ python run_iclassify.py --classify --corpus DIRECTORY1 --batch DIRECTORY2

    This classifies all files in the corpus in DIRECTORY1 and writes the
    classification results to DIRECTORY2. This includes (i) creating
    iclassify.mallet, the input vector file, (ii) creating the result files of
    the classifier, (iii) creating the label file from the classifier output,
    (iv) creating the .cat and .merged files.
    
There are a few more options of interest:

    --model DIRECTORY:
        The model to use, defaults to data/models/inventions-standard-20130713,
        which was used for the FUSE phase 2 midterm evaluation.
        
    --filelist PATH:
        A file with all files from the corpus to be used as input to the
        classifier. The default is to use config/files.txt in the corpus.

Some examples:

    $ python run_iclassify.py \
         --classify \
         --corpus ../creation/data/patents/test-4 \
         --model data/models/inventions-standard-20130713 \
         --batch data/results/test-4

Currently, only the classification phase is done this way, training and
evaluation need to be added.

"""


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
import re
import config
import mallet
import codecs
import getopt
import subprocess

from collections import defaultdict
from signal import signal, SIGPIPE, SIG_DFL 

# TODO: why does this work?
from ontology.utils.file import get_year_and_docid, open_input_file, ensure_path
from ontology.utils.git import get_git_commit


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

# To create the mallet file, we need to combine the category label in the
# annotation file (.lab) with the features for the same chunk in the phr_feats
# file.  We also need to remove any features that are not included in the list
# of features in the .features file specified in the features parameter.  For
# example, we don't want to include the sent_loc feature.  Currently, the
# features files reside in the /features subdirectory below the code directory.

def create_mallet_training_file(annot_file, phr_feats_file, mallet_training_file,
                                features=None, version="1", xval=0):

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
    #print "[invention] Found %i labels." % (chunkid2label_count)

    # Now output a mallet training features instance for each labeled feature
    # instance in phr_feats file
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
    print("[invention]Found %i labels, created %i feature instances" \
          % chunkid2label_count, output_count)

    s_annot.close()
    s_feats.close()
    s_mallet_training.close()

    
# root_dir specifies where the pipeline processed files are located (ie. the
# d3_phr_feats dir) file_list_file provides a list of the files (in the
# root_dir/) that we want to classify.  This is a tab separated file with year,
# full path, year/filename.xml iclassify_dir is where the mallet files will go.
# The model must be in this dir.  Intermediate and output files will be placed
# here under standard naming conventions.

def create_mallet_classify_file(root_dir, file_list_file, iclassify_dir,
                                features=None, version="1", verbose=False):

    print "[create_mallet_classify_file] creating mallet file..."
    # output file (the mallet instance file to be classified)
    mallet_file = os.path.join(iclassify_dir, "iclassify.mallet")
    num_lines_output = 0

    with open(file_list_file) as s_file_list, \
         codecs.open(mallet_file, "w", encoding='utf-8') as s_mallet:
        file_count = 0
        for line in s_file_list:
            file_count += 1
            line = line.strip("\n")
            # get the date/filename portion of path
            rel_file = line.split("\t")[2]
            phr_feats_file = os.path.join(root_dir, 'data', 'd3_phr_feats', '01',
                                          'files', rel_file)
            if verbose:
                print "[create_mallet_classify_file] %05d reading phr_feats from %s" \
                      % (file_count, os.path.basename(phr_feats_file))
            num_lines_output += add_phr_feats_file(phr_feats_file, s_mallet)
        print "[create_mallet_classify_file] %i lines written to %s" \
              % (num_lines_output, os.sep.join(mallet_file.split(os.sep)[-4:]))


def add_phr_feats_file(phr_feats_file, s_mallet):
    """Loop through phr_feats_file and add the first 30 lines to s_mallet. Only
    add the lines if the chunk is in the title or abstract."""
    # TODO: should maybe b ein separate utilities file (classifier_utlis.py)
    global output_count
    num_lines_output = 0
    # handle compressed or uncompressed files
    s_phr_feats = open_input_file(phr_feats_file)
    # keep first 30 chunks, if they are from title/abstract
    num_chunks = 0
    for line in s_phr_feats:
        if num_chunks >= 30:
            break
        line = line.strip("\n")
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
            s_mallet.write(instance_line)
            num_chunks += 1
            num_lines_output += 1
    s_phr_feats.close()
    return num_lines_output



# Given a mallet training file, create a model
# invention equivalent to patent_utraining_data3. 
# We assume mallet training file already exists, with labels
# invention parallel to patent_training_data3()
def patent_invention_train(mallet_file,
                           features="invention", version="1", xval=0,
                           verbose=False, stats_file=None):

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
        classifier_type="MaxEnt", number_xval=xval, training_portion=0,
        prune_p=False, infogain_pruning="5000", count_pruning="3")
    mtr = mallet.MalletTraining(mconfig, features)
    # we can't use make_utraining_file3 since we do not base our annotations on doc_feats.
    #mtr.make_utraining_file3(fnames, d_phr2label, features=features)
    mtr.write_train_mallet_vectors_file()
    mtr.mallet_train_classifier()
    # todo: add the following line
    ###write_training_statistics(stats_file, mtr)


def patent_invention_classify(train_dir, test_dir, features="invention",
                              version="1", verbose=False, stats_file=None):
    if verbose:
        print '[patent_invention_classify] train_dir =', train_dir
        print '[patent_invention_classify] test_dir  =', test_dir
    mallet_config = mallet.MalletConfig(config.MALLET_DIR, "itrain", "iclassify",
                                        version, train_dir, test_dir, classifier_type="MaxEnt")
    mallet_classifier = mallet.MalletClassifier(mallet_config)
    mallet_classifier.mallet_classify(verbose=verbose)


# Retrieve the title of a patent, which is on line 2 of the files in the txt directory.
# here root_path is the directory for the txt file, up to the year subdirectory.
# filename is year/file.xml
def patent_title(root_path, filename, verbose=False):
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
    if verbose:
        print "cat_command: %s" % cat_command
    # note we have to decode the title byte-string into unicode so that it can be
    # appended to a unicode string later and the entire string will be encodable from utf-8.
    title = commands.getoutput(cat_command).decode('utf-8')
    #print "title: %s" % title
    return (title)

# take a list and create a string separated by "," with "_" replaced by blank.
def list_to_csv_string(l_items):
    return( ", ".join(l_items).replace("_", " "))

# Output a human readable summary of title and keyterms for a patent to an open stream
def output_doc_summary(year, doc, title, d_label2terms, s_merged):
    s_merged.write("[%s %s]\n" % (year, doc))
    s_merged.write("title: %s\n" % title)
    s_merged.write("invention type: %s\n" % list_to_csv_string(d_label2terms["t"]))
    s_merged.write("invention descriptors: %s\n" % list_to_csv_string(d_label2terms["i"]))
    s_merged.write("contextual terms: %s\n" % list_to_csv_string(d_label2terms["m"]))
    s_merged.write("components/attributes: %s\n" % list_to_csv_string(d_label2terms["c"]))
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

def merge_scores(source_path, iclassify_path, label_file, runtime=False, verbose=True):

    if verbose:
        print "[merge_scores] merging scores from label file"

    d_invention_type = _get_invention_types()
    (s_labels, s_merged, s_cat) = _get_file_handles(iclassify_path, label_file)
    
    last_doc = ""
    last_year = ""
    title = ""
    year = ""

    # for each doc, track which phrases have been seen
    d_seen = {}
    d_label2terms = defaultdict(list)

    # The same phrase can occur multiple times with different labels. We will
    # choose the first label (heuristic). The label i(nvention) includes some
    # invention type terms. We will detect these and relable them as type. We
    # will also look at the last term in a multiword i phrase to see if it
    # contains a type term (in case one does not occur independently.
    # Format of label file is: 1994|US5318556A.xml_0|fluid_bag i 0.864621951173
    line_no = 1
    for line in s_labels:
        #print "starting line: %i " % line_no
        line_no += 1
        (key, label, score, year, chunk_id, term, doc, chunk_no) = \
              _parse_feats_line(line, runtime)
        #print "doc: %s, last_doc: %s, chunk_no: %s" % (doc, last_doc, chunk_no)
        if doc != last_doc:
            _print_summary(source_path, last_year, runtime, last_doc,
                           d_label2terms, s_merged, s_cat)
            last_doc = doc
            last_year = year
            d_seen = {}
            d_label2terms = defaultdict(list)
        # if term hasn't been seen, store under its first label
        if not d_seen.has_key(term):
            d_seen[term] = True
            if d_invention_type.has_key(term):
                label = "t"
            d_label2terms[label].append(term)

    # for end of file...
    _print_summary(source_path, last_year, runtime, last_doc,
                   d_label2terms, s_merged, s_cat)

    close_files(s_labels, s_merged, s_cat)


def _print_summary(source_path, last_year, runtime, last_doc,
                   d_label2terms, s_merged, s_cat):
    txt_path = _get_text_path(source_path, last_year, runtime)
    title = patent_title(txt_path, last_doc)
    if last_doc != "":
        output_doc_summary(last_year, last_doc, title, d_label2terms, s_merged)
        output_cat_summary(last_doc, d_label2terms, s_cat)

def _parse_feats_line(line, runtime):
    line = line.strip("\n")
    (key, label, score) = line.split("\t")
    (year, chunkid, term) = key.split("|")
    (doc, chunk_no) = chunkid.split("_")
    # this is a bit of a hack to make this work in the runtime setting where
    # files have added suffixes (MV)
    if runtime and doc.endswith('.tag'):
        doc = doc[:-4] + '.txt'
    return (key, label, score, year, chunkid, term, doc, chunk_no)

def _get_invention_types():
    l_invention_type = ['assembly', 'means', 'compositions', 'composition',
                        'method', 'methods','apparatus', 'system', 'use',
                        'process', 'device', 'technique']
    # put invention types into a dictionary for easy testing
    d_invention_type = {}
    d_invention_type = d_invention_type.fromkeys(l_invention_type)
    return d_invention_type

def _get_file_handles(iclassify_path, label_file):
    # full path of label and merged (output) file
    label_path = os.path.join(iclassify_path, label_file)
    output_path = os.path.join(iclassify_path, label_file + ".merged")
    cat_path = os.path.join(iclassify_path, label_file + ".cat")
    s_labels = codecs.open(label_path, encoding='utf-8')
    s_merged = codecs.open(output_path, "w", encoding='utf-8')
    s_cat = codecs.open(cat_path, "w", encoding='utf-8')
    return (s_labels, s_merged, s_cat)

def _get_text_path(source_path, last_year, runtime=False):
    if runtime:
        return source_path
    # Don't include the year if it has defaulted to 9999 (meaning no year subdirectory exists)
    if last_year == "9999":
        return os.path.join(source_path, 'data', 'd1_txt', '01', 'files')
    # include the year in the directory path for the txt files
    return os.path.join(source_path, 'data', 'd1_txt', '01', 'files', last_year)

def close_files(*args):
    for arg in args:
        arg.close()




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

                            
def eval_iclassify(manual_path, manual_file, iclassify_path, iclassify_file):
    l_invention_type = ['assembly', 'means', 'compositions', 'composition', 'method', 'methods', 'apparatus', 'system', 'use', 'process', 'device', 'technique']
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
            key = key

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
    # these and relable them as type.  We will also look at the last term in a 
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
 


def read_opts():
    # copied from run_classifier, but most options are not used here
    longopts = ['corpus=', 'language=', 'train', 'classify', 'evaluate',
                'pipeline=', 'filelist=', 'annotation-file=', 'annotation-count=',
                'batch=', 'features=', 'xval=', 'model=', 'eval-on-unseen-terms',
                'verbose', 'show-batches', 'show-data', 'show-pipelines',
                'gold-standard=', 'threshold=', 'logfile=']
    try:
        return getopt.getopt(sys.argv[1:], '', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))

            
def run_iclassifier(corpus, filelist, model, classification,
                    label_file='iclassify.MaxEnt.label', verbose=False):
    """Run the invention classifier on the corpus using the model specified and
    create a classification."""
    print '\n[run_iclassifier] corpus =', corpus
    print '[run_iclassifier] files  =', filelist
    print '[run_iclassifier] model  =', model
    print '[run_iclassifier] class  =', classification
    ensure_path(classification)
    create_info_file(corpus, model, filelist, classification)
    # create classification/iclassify.mallet from given files in the corpus
    create_mallet_classify_file(corpus, filelist, classification, "invention", "1",
                                verbose=verbose)
    # create result files in the classification
    patent_invention_classify(train_dir=model, test_dir=classification)
    # creates the label file from the classifier output
    print "[run_iclassifier] creating the .label file"
    command = "cat %s/%s | egrep -v '^name' | egrep '\|.*\|' | python %s > %s/%s" \
              % (classification, 'iclassify.MaxEnt.out', 'invention_top_scores.py',
                 classification, label_file)
    print '   $', command
    subprocess.call(command, shell=True)
    # creates the .cat and .merged files
    merge_scores(corpus, classification, label_file)

    
def create_info_file(corpus, model, filelist, classification):
    with open(os.path.join(classification, 'iclassify.info.general'), 'w') as fh:
        fh.write("$ python %s\n\n" % ' '.join(sys.argv))
        fh.write("corpus          =  %s\n" % corpus)
        fh.write("file_list       =  %s\n" % filelist)
        fh.write("model           =  %s\n" % model)
        fh.write("classification  =  %s\n" % classification)
        fh.write("git_commit      =  %s" % get_git_commit())


def generate_tab_format(classification):
    """Creates the format that is input to the BAE triple store. Should probbaly
    just be added to run_iclassifier """

    print "[generate_tab_format] creating tab file\n"
    with open(os.path.join(classification, 'iclassify.info.merged.tab'), 'w') as fh:
        fh.write("$ python %s\n\n" % ' '.join(sys.argv))
        fh.write("classification        =  %s\n" % classification)
        fh.write("git_commit            =  %s\n" % get_git_commit())

    infile = os.path.join(classification, 'iclassify.MaxEnt.label.merged')
    outfile = os.path.join(classification, 'iclassify.MaxEnt.label.merged.tab')
    fh_in = codecs.open(infile)
    fh_out = codecs.open(outfile, 'w')

    patent_id = None
    c = 0
    for line in fh_in:
        c += 1
        if c % 10000 == 0: print c
        #if c > 10000: break
        if line.startswith('['):
            year, filename = line.strip("\n\r[]").split()
            patent_id = get_patent_id_from_filename(filename)
        elif line.strip() == '':
            patent_id = None
        else:
            for field, abbrev in [('invention type', 't'),
                                  ('invention descriptors', 'i'),
                                  ('contextual terms', 'ct'),
                                  ('components/attributes', 'ca')]:
                if line.startswith(field):
                    vals = line[len(field)+1:].strip()
                    #field_print_name = field.replace(' ', '_').replace('/', '_')
                    if patent_id and vals:
                        for val in vals.split(', '):
                            fh_out.write("%s\t%s\t%s\t%s\t%s\n" \
                                         % (year, patent_id, filename, abbrev, val))



# regular expression to pick the id out of a patent filename, this was tested
# with check_regexp_on_index()
PATENT_ID_EXP = re.compile('^(US)?(\D*)(\d+)')


def get_patent_id_from_filename(fname):
    fname = os.path.basename(fname)
    result = PATENT_ID_EXP.search(fname)
    if result is None:
        return None
    return result.group(2) + result.group(3)
    
    
def check_regexp_on_index():
    """Check whether the regular expression is the one used for generating the
    index file. Print a line if the id in the file does not match the one
    extracted from the filename."""
    exp = re.compile('^(US)?(\D*)(\d+)')
    f = '/home/j/corpuswork/fuse/FUSEData/lists/ln_uspto.all.index.txt'
    fh = open(f)
    fh.readline()
    for line in fh:
        id, fname = line.split()
        fname = os.path.basename(fname)
        result = exp.search(fname)
        if result is None:
            print line,
        id_in_fname = exp.search(fname).group(2) + exp.search(fname).group(3)
        if id != id_in_fname:
            print id, id_in_fname, fname





if __name__ == '__main__':

    # first set some defaults, many of these will never be overwritten
    
    # loc of phr_feats files inside corpus:  data/d3_phr_feats/01/files/
    # each line of filelist in config/files.txt contains: date full_path_spec
    # path_spec_from_root_dir
    corpus = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/"

    # default locations of the statistical model (itrain.model) and the
    # classification
    model = 'data/models/inventions-standard-20130713'
    classification = os.path.join(os.getcwd(), 'ws')

    train = False
    classify = False
    create_bae_tabfile = False
    filelist = None
    verbose = False

    # now read options and call the main method
    (opts, args) = read_opts()
    for opt, val in opts:
        if opt == '--train': train = True
        elif opt == '--classify': classify = True
        elif opt == '--corpus': corpus = val
        elif opt == '--model': model = val
        elif opt == '--batch': classification = val
        elif opt == '--filelist': filelist = val
        elif opt == '--verbose': verbose = True

    if filelist is None:    
        filelist = os.path.join(corpus, "config/files.txt")

    if classify:
        run_iclassifier(corpus, filelist, model, classification, verbose=verbose)
        generate_tab_format(classification)
    else:
        print "WARNING: nothing to do."
