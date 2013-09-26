"""
library for dealing with mallet feature production
PGA 10/72012

python 2.6 or higher required

Rewritten 12/19/12 PGA to remove file naming from inner functions

We assume that training data files will all be in placed in a single directory $train_dir
file names for use with mallet are constructed using
a user supplied file_prefix (e.g. test_data)
a user supplied trainer (which must be a case sensitive Mallet trainer keyword (e.g. MaxEnt)
system supplied qualifiers (such as .out, .mallet, .vectors, .vectors.out)

import test_flib to test the flib methods with a small example.

For mallet command line documentation, see
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/cmt-40/Nice/Urdu-MT/code/Tools/POS/postagger/mallet_0.4/doc/command-line-classification.html

"""

import os
import sys
import re
import codecs
from collections import defaultdict
import config
import inspect

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
os.chdir('../..')
sys.path.insert(0, os.getcwd())
os.chdir(script_dir)

from ontology.utils.batch import generate_doc_feats
from ontology.utils.file import compress, uncompress, get_year_and_docid, open_input_file



def parse_doc_feats_line(line):
    """Parse a doc_feats line and return phrase, identifier and feature list."""
    fields = line.strip("\n").split("\t")
    phrase, uid = fields[0], fields[1]
    feats = unique_list(fields[2:])
    return (phrase, uid, feats)

def unique_list(non_unique_list):
    # TODO: copied from train.py, move this to a utilities file or directory
    return (list(set(non_unique_list)))

def cattrs(inst):
    """View the attributes and values of a class instance."""
    attributes = inspect.getmembers(inst, lambda a:not(inspect.isroutine(a)))
    for a in attributes:
        if not(a[0].startswith('__') and a[0].endswith('__')):
            print a

def run_command(cmd):
    print_command(cmd)
    os.system(cmd)

def print_command(cmd):
    cmd = cmd.replace(' --', "\n      --")
    cmd = cmd.replace(' > ', "\n      > ")
    cmd = cmd.replace(' 2> ', "\n      2> ")
    cmd = cmd.replace(' | ', "\n      | ")
    print '   $', cmd




############################################################################
class MalletConfig(object):

    # TODO: add self.classifier_type to all the file names!  So far, only added
    # it to the test output files.

    def __init__(self, mallet_dir, train_file_prefix, test_file_prefix, version,
                 train_dir, test_dir, classifier_type="MaxEnt", number_xval=0,
                 training_portion=0, prune_p=False, infogain_pruning="5000",
                 count_pruning="3"):

        # TBD: Add test to constrain number_xval or training_portion to 0.
        self.mallet_dir = mallet_dir
        self.train_file_prefix = train_file_prefix
        self.test_file_prefix = test_file_prefix
        self.version = version
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.classifier_type = classifier_type
        self.number_xval = int(number_xval)
        self.training_portion = float(training_portion)
        self.infogain_pruning = infogain_pruning
        self.prune_p = prune_p
        self.count_pruning = count_pruning
        
        # mallet output files
        # data
        self.train_mallet_file = os.path.join(train_dir, train_file_prefix + ".mallet")
        self.train_vectors_file = os.path.join(train_dir, train_file_prefix + ".vectors")
        self.train_vectors_out_file = os.path.join(train_dir, train_file_prefix + ".vectors.out")

        # Currently, pruning is not working, due to an error in the mallet code
        # itself, which doesn't set up the pipes correctly.  train_prune_file
        # will only exist if the prune command is run.  If it's value is "", we
        # will assume it was not run and the train_vectors file should be used
        # as input to the classifier.  If its value is not "", it will be used
        # as the train_vectors file for training the classifier.
        self.pruned_vectors_file = os.path.join(train_dir, train_file_prefix + ".pruned.vectors")

        self.test_mallet_file = os.path.join(test_dir, test_file_prefix + ".mallet")
        self.test_vectors_file = os.path.join(test_dir, test_file_prefix + ".vectors")
        self.test_vectors_out_file = os.path.join(test_dir, test_file_prefix + ".vectors.out")
        
        self.test_mallet_file = os.path.join(test_dir, test_file_prefix + ".mallet")

        # training output
        self.model_file = os.path.join(train_dir, train_file_prefix + ".model")
        self.train_out_file = os.path.join(train_dir, train_file_prefix + ".out")
        self.train_stderr_file = os.path.join(train_dir, train_file_prefix + ".stderr")

        # feature values for model
        self.cinfo_file = os.path.join(train_dir, train_file_prefix + ".cinfo")
        self.cinfo_sorted_file = os.path.join(train_dir, train_file_prefix + ".cinfo.sorted")
        
        # classifier output
        self.classifier_out_file =  os.path.join(test_dir, test_file_prefix +
                                                 "." + self.classifier_type + ".out")
        self.classifier_stderr_file =  os.path.join(test_dir, test_file_prefix + "." +
                                                    self.classifier_type + ".stderr")


        # vectors
        self.cmd_csv2vectors_train = "sh " + self.mallet_dir + "/csv2vectors" + \
                                     " --token-regex '[^ ]+'" + \
                                     " --input " + self.train_mallet_file + \
                                     " --output " + self.train_vectors_file + \
                                     " --print-output TRUE" + \
                                     " > " + self.train_vectors_out_file

        self.cmd_csv2vectors_test = "sh " + self.mallet_dir + "/csv2vectors" + \
                                    " --input " + self.test_mallet_file + \
                                    " --output " + self.test_vectors_file + \
                                    " --print-output TRUE" + \
                                    " --use-pipe-from " + self.get_vectors_file()  + \
                                    " > " + self.test_vectors_out_file

        # pruning
        self.cmd_prune = "sh " + self.mallet_dir + "/mallet prune" + \
                         " --input " + self.train_vectors_file + \
                         " --output " + self.pruned_vectors_file + \
                         " --prune-infogain " + self.infogain_pruning + \
                         " --prune-count " +  self.count_pruning

        # training/testing
        # --report test:raw option provides <id> <actual> <predicted> labels, e.g.
        # 2 OUT OUT:0.6415563874015857 IN:0.3584436125984143 

        if self.training_portion > 0.0:
            print "[mallet_train_classifier] " + \
                  "setting mallet command with portions for testing and training"
            self.cmd_train_classifier = "sh " + self.mallet_dir + "/mallet train-classifier" + \
                                        " --input " + self.get_vectors_file() + \
                                        " --trainer " + self.classifier_type + \
                                        " --output-classifier " + self.model_file + \
                                        " --training-portion " + str(training_portion) + \
                                        " --report test:accuracy test:confusion train:raw" + \
                                        " > " + self.train_out_file + \
                                        " 2> " + self.train_stderr_file

        elif self.number_xval < 2: 
            print "[mallet_train_classifier] " + \
                  "setting mallet command without cross validation or portions"
            self.cmd_train_classifier = "sh " + self.mallet_dir + "/mallet train-classifier" + \
                                        " --input " + self.get_vectors_file() + \
                                        " --trainer " + self.classifier_type + \
                                        " --output-classifier " + self.model_file + \
                                        " --report test:accuracy test:confusion test:raw" + \
                                        " > " + self.train_out_file + \
                                        " 2> " + self.train_stderr_file

        else:
            # using cross-validation
            print "[mallet_train_classifier] setting mallet command with cross validation"
            self.cmd_train_classifier = "sh " + self.mallet_dir + "/mallet train-classifier" + \
                                        " --input " + self.get_vectors_file() + \
                                        " --trainer " + self.classifier_type + \
                                        " --output-classifier " + self.model_file + \
                                        " --cross-validation " + str(self.number_xval) + \
                                        " --report test:accuracy test:confusion test:raw" + \
                                        " > " + self.train_out_file + \
                                        " 2> " + self.train_stderr_file

            # Remove low values (negative /// why are there multiple scores for
            # the same features?)
            self.cmd_sort_model = "cat " + self.cinfo_file + \
                                  " | egrep -v '^FEAT|^ <default'" + \
                                  " | egrep -v 'E-[0-9]+$'" + \
                                  " | sed -e 's/^ //'" + \
                                  " | python reformat_uc.py" + \
                                  " | awk '{print $2,$1}'" + \
                                  " | egrep -v '^-'" + \
                                  " | sort -nr" + \
                                  " > " + self.cinfo_sorted_file

        # classification
        self.cmd_classify_file = "sh " + self.mallet_dir + "/mallet classify-file" + \
                                 " --line-regex \"^(\S*)[\s,]*(\S*)[\s]*(.*)$\"" + \
                                 " --name 1 --data 3" + \
                                 " --input " + self.test_mallet_file + \
                                 " --classifier " + self.model_file + \
                                 " --output -" + \
                                 " > " + self.classifier_out_file + \
                                 " 2> " + self.classifier_stderr_file

        # create readable versions of model values for features
        self.cmd_classifier2info = "sh " + self.mallet_dir + "/classifier2info" + \
                                   " --classifier " + self.model_file + \
                                   " > " + self.cinfo_file

        self.cmd_cinfo_sorted = "cat " + self.cinfo_file + \
                                " | egrep -v '^FEAT|^ <default'" + \
                                " | egrep -v 'E-[0-9]+$'" + \
                                " | sed -e 's/^ //'" + \
                                " | python reformat_uc.py" + \
                                " | awk '{print $2,$1}'" + \
                                " | sort -nr" + \
                                " > " + self.cinfo_sorted_file


    # use pruned or unpruned vectors file depending on the value of parameter prune_p
    def get_vectors_file(self):
        if self.prune_p:
            return(self.pruned_vectors_file)
        else:
            return(self.train_vectors_file)


    def print_vars(self):
        print "\nINSTANCE VARIABLES:"
        print "   mallet_dir          %s" % self.mallet_dir
        print "   train_file_prefix   %s" % self.train_file_prefix
        print "   test_file_prefix    %s" % self.test_file_prefix
        print "   version             %s" % self.version
        print "   train_dir           %s" % self.train_dir
        print "   test_dir            %s" % self.test_dir
        print "   classifier_type     %s" % self.classifier_type
        print "   number_xval         %s" % self.number_xval
        print "   training_portion    %s" % self.training_portion
        print "   infogain_pruning    %s" % self.infogain_pruning
        print "   prune_p             %s" % self.prune_p
        print "   count_pruning       %s" % self.count_pruning

    def print_files(self):
        print "\nFILES:"
        for f in [self.train_mallet_file, self.train_vectors_file,
                  self.train_vectors_out_file, self.pruned_vectors_file,
                  self.test_mallet_file, self.test_vectors_file,
                  self.test_vectors_out_file, self.test_mallet_file,
                  self.model_file, self.train_out_file,
                  self.train_stderr_file, self.cinfo_file,
                  self.cinfo_sorted_file, self.classifier_out_file,
                  self.classifier_stderr_file]:
            print '  ', f


############################################################################
# MalletInstance class encapsulates data and methods to capture one line of a mallet instance input file
# Each line consists of: name, label, data
# data (features) are in the form fname=fvalue (e.g. next_pos=NN)
# used to be called Mallet_instance
class MalletInstance:

    # leave id as "" to generate numeric ids for instance names
    # or pass in an id (e.g., tlink.ds_id)
    # meta is optional string of information to be stored with the instance that will
    # not affect mallet processing (ie. for storing notes about the instance).
    def __init__(self, id, label, meta):
        self.id = str(id)
        self. label = label
        self.l_feat = []
        # meta data string used to capture info for future debugging
        # e.g., the sentence, event strings, etc.
        self.meta = meta

    # add a feature to the mallet instance.
    # Each feature is composed of a name and value
    # This way we can post-filter features by name to test different feature sets

    def add_feat(self, fname, fvalue):
        # If fvalue is not a string, we need to convert it to string
        str_value = str(fvalue)
        #feat = fname + "=" + str_value
        feat = fname + "=" + str_value
        self.l_feat.append(feat)


###################################################################
# MalletTraining class encapsulates data and methods for adding instances and running mallet.
# Note: Need to allow an instance list to be reused with a filter for ablation testing
# test sets require different parameters, maybe have a separate class?

class MalletTraining:

    def __init__(self, mallet_config, features=None):

        self.mallet_config = mallet_config

        # id's are 0 based
        self.next_instance_id = 0
        
        self.train_mallet_file = mallet_config.train_mallet_file

        self.l_instance = []

        # table of instances indexed by predicted and actual labels
        self.d_labels2uid = defaultdict(list)
        self.d_uid2labels = {}

        # a table of feature (prefixes) to use from the phr_feats lines
        self.d_features = {}
        if features is not None:
            self.populate_feature_dictionary(features)


    def populate_feature_dictionary(self, features):
        """Populate the d_features dictionary with all the features used for the
        model. If no features are added, the dictionary will remain empty, which
        downstream will be taken to mean that all features will be used. The
        argument can either be a filename or an identifier that points to a file
        in the features directory."""
        if os.path.isfile(features):
            filter_filename = features
        else:
            filter_filename = os.path.join("features", features + ".features")
        try:
            with open(filter_filename) as s_filter:
                print "[MalletTraining] Using features file: %s" % filter_filename
                for line in s_filter:
                    feature_prefix = line.strip()
                    self.d_features[feature_prefix] = True
        except IOError as e:
            print "[MalletTraining] No features file found: %s" % filter_filename

    def remove_filtered_feats(self, feats):
        """Given a list of features, return a line with all non-filtered
        features removed We filter on the prefix of the feature (that is, the
        part before the '='). Return the original line if the features
        dictionary is empty."""
        if not self.d_features:
            return feats
        return [f for f in feats if self.d_features.has_key(f.split("=")[0])]


    def make_utraining_file3(self, fnames, d_phr2label, verbose=False):

        """ Create a file with training instances for Mallet. The list of
        feature files to use is given in fnames and the annotated terms in
        d_phr2label. This method is based on a similarly named function in
        train.py. It was moved here for consistency. This version should
        eventually make train.make_utraining_file() obsolete."""

        version = self.mallet_config.version
        mallet_file = self.mallet_config.train_mallet_file
        print "[make_utraining_file3] writing to", mallet_file
        print "[make_utraining_file3] features used:", \
              sorted(self.d_features.keys())

        self.stats_labeled_count = 0
        self.stats_unlabeled_count = 0
        file_count = 0

        with codecs.open(mallet_file, "w", encoding='utf-8') as s_train:
            for phr_feats_file in fnames:
                if verbose:
                    print "%05d %s" % (file_count, phr_feats_file)
                year, doc_id = get_year_and_docid(phr_feats_file)
                with open_input_file(phr_feats_file) as fh:
                    # this hard-wires the use of union train
                    docfeats = generate_doc_feats(fh, doc_id, year)
                    for term in sorted(docfeats.keys()):
                        feats = docfeats[term][2:]
                        feats = self.remove_filtered_feats(feats)
                        uid = "%s|%s|%s" % (year, doc_id, term.replace(' ','_'))
                        if d_phr2label.has_key(term):
                            label = d_phr2label.get(term)
                            if label == "":
                                print "[make_utraining_file3] " + \
                                      "WARNING: term with null label: %s" % term
                            else:
                                # mallet line format: "uid label f1 f2 f3 ..."
                                mallet_line = " ".join([uid, label] + feats)
                                s_train.write(mallet_line + "\n")
                                self.stats_labeled_count += 1
                        else:
                            self.stats_unlabeled_count += 1
        
        print "[make_utraining_file3] labeled instances: %i, unlabeled: %i" \
              % (self.stats_labeled_count, self.stats_unlabeled_count)



    ## NOTE: The next two functions are used to build a .mallet file.  If this
    ## is built externally, they can be ignored.  However, the mallet file must
    ## consist of <uid> <label> <f1> <f2> ...
    ## and be named <train_file_prefix>.mallet

    # add an instance object to the list of instances in the MalletTraining object
    def add_instance(self, mallet_instance):
        # each mallet instance contains <id> <label> <feature>+
        # check if id is needed
        # if id parameter is "", we will default to ordered integer ids
        if mallet_instance.id == "":
            mallet_instance.id = str(self.next_instance_id)
            self.next_instance_id += 1
        self.l_instance.append(mallet_instance)

    # write out training instances file to file $train_file_prefix.ver.mallet
    def write_train_mallet_file(self):
        # TODO: this function does not appear to be used
        mallet_stream = open(self.mallet_config.train_mallet_file, "w")
        print "writing to: %s (with feature uniqueness enforced)" \
            %  self.mallet_config.train_mallet_file
        for instance in self.l_instance:
            mallet_stream.write("%s %s " % (instance.id, instance.label))
            # use (list(set(...))) to insure that each feature is only included once.
            # Some features, like prev_J, tend to occur multiple times.  Mallet will 
            # create a vector with value for the feature > 1.0 if it appears multiple times.
            # However, we are currently treating all features as binary (present/absent)
            mallet_stream.write(" ".join(list(set(instance.l_feat))))
            mallet_stream.write("\n")
        mallet_stream.close()

    # convert mallet instance file to mallet vectors format in file $file_prefix.vectors
    # This is required to run the classifier on the data.
    def write_test_mallet_vectors_file(self):
        cmd = self.mallet_config.cmd_csv2vectors_test
        print "[write_test_mallet_vectors_file]"
        run_command(cmd)

    # set vectors file name attribute directly (useful if testing on vectors
    # data created elsewhere), arg is full path for .vectors file
    def set_mallet_vectors_file(self, full_vectors_path):
        self.train_vectors_file = full_vectors_path
        self.train_vectors_out_file = full_vectors_path + ".out" 


    # train a mallet classifier
    # trainer is the case-sensitive name for a mallet classifier (e.g., "MaxEnt", "NaiveBayes")
    # To divide training data into a training and evaluation set, set training_portion to a value
    # between 0 and 1, (e.g., .7)
    # To do cross validation, set number_cross_val to some number >= 2 (e.g., 10)
    # To use all instances to train a classifier, leave both parameters empty (i.e., = 0)
    # Model will be in $train_path_prefix.<trainer>.model
    # Output (accuracy, confusion matrix, label predicted/actual) is in $train_path_prefix.<trainer>.out
    # Command line format: vectors2train --training-file train.vectors --trainer  MaxEnt --output-classifier foo_model --report train:accuracy train:confusion> foo.stdout 2>foo.stderr
    def mallet_train_classifier(self):

        commands = [self.mallet_config.cmd_csv2vectors_train,
                    self.mallet_config.cmd_train_classifier,
                    self.mallet_config.cmd_classifier2info,
                    self.mallet_config.cmd_cinfo_sorted]
        if self.mallet_config.prune_p:
            commands.append(self.mallet_config.cmd_prune)
        for cmd in commands:
            print "[mallet_train_classifier]"
            run_command(cmd)
        compress(self.mallet_config.cinfo_file,
                 self.mallet_config.cinfo_sorted_file,
                 self.mallet_config.train_vectors_file,
                 self.mallet_config.train_vectors_out_file,
                 self.mallet_config.train_mallet_file)


    def mallet_csv2vectors(self):
        cmd = self.mallet_config.cmd_csv2vectors_train
        print "[mallet_train_classifier]"
        run_command(cmd)



##################
# PGA 
# class for creating a test data set using the same features as a pre-existing classifier
# Test data contains <id> <feature>+

# This was previously called mallet_test
class MalletClassifier:

    def __init__(self, mallet_config):
        self.mallet_config = mallet_config
        # id's are 0 based
        self.next_instance_id = 0
        self.l_instance = []

    # skip next two functions if .mallet file is created elsewhere
    def add_instance(self, mallet_instance):
        # each mallet instance contains <id> <label> <feature>+
        # check if id is needed
        # if id parameter is "", we will default to ordered integer ids
        if mallet_instance.id == "":
            mallet_instance.id = str(self.next_instance_id)
            self.next_instance_id += 1
        self.l_instance.append(mallet_instance)

    # write out test instances file to test_dir
    def write_test_mallet_file(self):
        mallet_stream = open(self.mallet_config.test_mallet_file, "w")
        print "writing to: %s" %  self.test_mallet_file
        for instance in self.l_instance:
            # note the label is not included as a field in test data for mallet
            mallet_stream.write("%s %s " % (instance.id))
            # Make sure l_feat list is unique before creating the mallet output line.
            # We are treating each feature as binary(present/absent) and some features can
            # occur more than once in phr_feats file.
            mallet_stream.write(" ".join(list(set(instance.l_feat))))
            mallet_stream.write("\n")
        mallet_stream.close()

    # create test vectors file compatible with training vectors file (using
    # use-pipe-from option)
    def write_test_mallet_vectors_file(self):
        cmd = self.mallet_config.cmd_csv2vectors_test
        print "[write_test_mallet_vectors_file]"
        run_command(cmd)

    # trainer is parameter to the method to allow for multiple classifiers over
    # the same data However, models built with the trainer must already exist in
    # the training directory

    # Note also that training with xvalidation on will create multiple models,
    # one per trial.  You need to train with no xvalidation to generate a model
    # file name that will work with the tester here.

    # deprecated 6/6/13 PGA
    def mallet_test_classifier(self):
        print "[mallet_test_classifier] classifier_type is %s" \
              % self.mallet_config.classifier_type
        cmd = self.mallet_config.cmd_classify_file
        print "[mallet_test_classifier]"
        run_command(cmd)

    # replaces mallet_test_classifier (PGA, used by invention.py)
    def mallet_classify(self):
        print "[mallet_test_classifier] classifier_type is %s" \
              % self.mallet_config.classifier_type
        cmd = self.mallet_config.cmd_classify_file
        print "[mallet_test_classifier]"
        run_command(cmd)
        
    def compress_files(self):
        compress(self.mallet_config.test_mallet_file,
                 self.mallet_config.classifier_out_file)

    #create a vectors file from the .mallet file for the test data
    def mallet_csv2vectors_test(self):
        cmd = self.mallet_config.cmd_csv2vectors_test
        print "[mallet_test_classifier]"
        run_command(cmd)
            


#######################################################################
# analyze mallet results different ways
class ResultInspector():

    def __init__(self, ds, mallet_config):
        self.ds = ds
        # dictionaries for tlink data
        self.d_uid2features = {}
        self.d_uid2labels = {}
        self.d_labels2uid = defaultdict(list)
        # maps actual label and feature to list of so labeled tlinks containing the feature
        self.d_afeat2uid = defaultdict(list)
        self.load_raw_data()


    def load_raw_data(self):

        # extract the raw data section from the .out file
        in_raw_data = False

        # process the .out file for label information (output of report test:raw option)
        # NOTE: in the case of cross validation folds, this will only capture the test results of
        # all folds
        
        out_stream = open(self.mallet_config.classifier_out_file)
        for line in out_stream:
            if in_raw_data and re.match("^[^ ]+ [^ ]+ [^ ]+:", line):
                m = re.match("^(?P<id>[^ ]+) (?P<actual>[^ ]+) (?P<pred>[^ ]+):", line)
                uid = m.group("id")
                actual = m.group("actual")
                pred = m.group("pred")
                # index the line
                
                self.d_uid2labels[uid] = [actual, pred]
                #print "Indexed %s => [%s %s]" % (uid, actual, pred)
            else:
                # No more raw data
                in_raw_data = False
                
            if " Raw Testing Data" in line:
                in_raw_data = True

        out_stream.close()

        # now create a reverse index from label pairs to id lists
        for uid in self.d_uid2labels.keys():
            [actual, pred] = self.d_uid2labels.get(uid)
            labels_key = actual + "_" + pred
            self.d_labels2uid[labels_key].append(uid)

        # .mallet file example line: 61 AFTER c2_path__to c1_vrel__PRD c1_etype__PROBLEM c2_etype__TEST
        mallet_stream = open(self.mallet_config.train_mallet_file, "r")
        for line in mallet_stream:
            line = line.strip()
            line_fields = line.split(" ")
            uid = line_fields[0]
            actual = line_fields[1]
            feature_list = line_fields[2:]
            self.d_uid2features[uid] = feature_list

            # also index tlinks by actual label and feature (LABEL_feat)
            for feat in feature_list:
                key = actual + "_" + feat
                self.d_afeat2uid[key].append(uid)
                

        mallet_stream.close()

    # print a list of ids for a given label pair (for inspection of misclassified instances)
    def print_res(self, actual, pred):
        labels_key = actual + "_" + pred
        l_uid = self.d_labels2uid[labels_key]
        print "%s %s:" % (actual, pred)
        for uid in l_uid:
            print "%s" % uid
            # output more diagnostic info here
            # by linking id with actual i2b2 concept data
            
    def print_uid_res(self, uid):
        print self.d_uid2features[uid]
        print self.d_uid2labels[uid]

    # debug a feature, given actual label and feature, range within list of uids to display.
    def db_feat(self, actual, feat, start=0, end=4):
        key = actual + "_" + feat
        l_uid = self.d_afeat2uid[key][start:end]
        for uid in l_uid:
            print "\nTLINK: %s" % uid
            self.db_tlink(uid)
            
    # debug tlink with useful info on sentence, tlink, and mallet results
    def db_tlink(self, uid):
        # uid is composed of doc_id, "tl", and tlink id
        [doc_id, tlstr, tlink_id] = uid.split("_")
        [actual_label, pred_label] = self.d_uid2labels[uid] 
        print "Actual: %s. pred: %s" % (actual_label, pred_label)
        print "Features: %s" % self.d_uid2features[uid]

        # There is a bug that some keys are missing, so we include a workaround for now ///
        if self.ds.id2tlinks.has_key(uid):
            self.ds.id2docs[doc_id].display_tlink(self.ds.id2tlinks[uid])
            print "Path: %s" % self.ds.id2tlinks[uid].dep_path
        else:
            print "Missing key %s in ds.id2.tlinks" % uid


####################################################################
# small utility functions for creating features

def startswith_one(str, l_prefix):
    for p in l_prefix:
        #print "matching %s with %s" % (p, str)
        if str.startswith(p):
            return True
    return False

def verb_p(dep_node):
    if dep_node.pos == None:
        return False
    if dep_node.pos.startswith("VB"):
        return True
    else:
        return False

# if head is a verb, return the dep_node's deprel (e.g., SBJ)
def verb_rel(dep_node):
    if dep_node.pos == None:
        return False
    if verb_p(dep_node.head):
        return(dep_node.deprel)
    else:
        return None

#returns a list of lemmas for dep_nodes found in the path matching one of the
# parts of speech in pos_list.  The pos tags are matched as prefixes, such that
# VB will match all verb pos.
# verb = VB, prep = IN, conj = CC
# Path is a list of DependencyNode instances
def feat_path_contains_pos(path, pos_list):
    l_lemmas = []
    for dep_node in path:
        #print "Trying to match dep_node pos: %s, pos_list: %s" % (dep_node.pos, pos_list)
        if startswith_one(dep_node.pos, pos_list):
            #print "matched %s,  lemma is %s" % (dep_node.pos, dep_node.lemma)
            l_lemmas.append(dep_node.lemma)
    return(l_lemmas)

def tok_seq_contains_pos(seq, pos_list, append_pos_p=True):
    l_lemmas = []
    for tok in seq:
        #print "Trying to match dep_node pos: %s, pos_list: %s" % (tok.dep_node.pos, pos_list)
        if startswith_one(tok.dep_node.pos, pos_list):
            #print "matched %s,  lemma is %s" % (tok.dep_node.pos, tok.dep_node.lemma)
            lemma = tok.dep_node.lemma
            if append_pos_p:
                lemma = tok.dep_node.pos + "_" + tok.dep_node.lemma
            l_lemmas.append(lemma)
    return(l_lemmas)

# return True if string contains any part in l_parts
def string_contains(string, l_parts):
    for part in l_parts:
        if part in string:
            return(True)
    return(False)

# returns a string version of a sequence of tokens (separated by "_")
def l_tok2string(l_tok):
    l_tok_str = []
    joined_string = ""
    for tok in l_tok:
        l_tok_str.append(tok.form)
    joined_string =  "_".join(l_tok_str)
    return(joined_string)


# This test assumes that mallet training and test files already exist.
def test1():
    mallet_dir = config.MALLET_DIR
    train_file_prefix = "utrain_30K"  
    test_file_prefix =  "utest"
    version = "1"
    train_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/test1/train"
    test_dir = "/home/j/anick/patent-classifier/ontology/creation/data/patents/test1/test"
    mallet_config = MalletConfig(mallet_dir, train_file_prefix, test_file_prefix,
                                 version, train_dir, test_dir, classifier_type="MaxEnt",
                                 number_xval=0, training_portion=0)
    mallet_training = MalletTraining(mallet_config)
    mallet_training.mallet_train_classifier()
    print "[test1]After training classifier"
    return(mallet_config)

