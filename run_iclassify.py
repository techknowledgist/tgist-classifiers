"""

Script to generate invention keyterm scores. 

This is based on inventions.py, but the interface and some methods are tweaked
to look more like run_tclassify.py. Overall, much of the bookkeeping done in the
latter is not done here yet.


CLASSIFIER

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


MODEL CREATION

1. First find out what documents are in the annotation set:

    $ cut -f2 key.ta.20130510.lab | cut -f1 -d'_' | sort | uniq

2. Put these in a mini corpus and pre-process them. You can use find_patents.py and
create-file-list.py in ../creation/data/patents to help with that.

3. Now collect the phr_feats and put them in a keyfeats.ta.DATE.dat file. This
may be a simple unzip and cat, but it could be more complicated (offsets could
have shifted, may need to select just those that that were annotated, need to
reformat into mallet format, etcetera).

4. Create the mallet file using create_mallet_training_file().

5. Create the model using patent_invention_train().

Steps 3, 4 and 5 are done by running this script using the --train option:

    $ python_run_iclassify.py --train --corpus PATH --filelist PATH --annotations PATH

TODO: this needs to be tested.

"""

import commands
import os
import sys
import re
import time
import codecs
import getopt
import subprocess
import shutil
from signal import signal, SIGPIPE, SIG_DFL 

import config
import mallet
import invention

# this works because the mallet import has already amended the path
from ontology.utils.file import get_year_and_docid, open_input_file, ensure_path
from ontology.utils.git import get_git_commit


# Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
# Without this, we get some "broken pipe" messages in the output.
signal(SIGPIPE,SIG_DFL) 


# Given a mallet training file, create a model
# invention equivalent to patent_utraining_data3. 
# We assume mallet training file already exists, with labels
# invention parallel to patent_training_data3()
def patent_invention_train(mallet_file,
                           features="invention", version="1", xval=0,
                           verbose=False, stats_file=None):

    """Wrapper around mallet.py functionality to create a classifier model. The
    .mallet training instances file must exist and full path passed in. Other
    files needed for mallet processing will be placed in the same directory
    (train_output_dir). Creates an instance of MalletTraining class to do the
    rest: creating the .vectors file from the mallet file, and creating the
    model."""

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




def read_opts():
    # copied from run_classifier, but most options are not used here
    longopts = ['corpus=', 'language=', 'train', 'classify', 'evaluate',
                'pipeline=', 'filelist=', 'annotation-file=',
                'batch=', 'features=', 'xval=', 'model=', 'eval-on-unseen-terms',
                'verbose' ]
    try:
        return getopt.getopt(sys.argv[1:], '', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))


def run_iclassifier(corpus, filelist, model, classification,
                    label_file='iclassify.MaxEnt.label', verbose=False):
    """Run the invention classifier on the corpus using the model specified and
    create a classification."""
    print
    print '[run_iclassifier] corpus =', corpus
    print '[run_iclassifier] files  =', filelist
    print '[run_iclassifier] model  =', model
    print '[run_iclassifier] class  =', classification
    t1 = time.time()
    ensure_path(classification)
    create_info_files(corpus, model, filelist, classification)
    # create classification/iclassify.mallet from given files in the corpus
    # NOTE: got rid of verbose argument
    invention.create_mallet_classify_file(corpus, filelist, classification, "invention", "1")
    t2 = time.time()
    # create result files in the classification
    invention.patent_invention_classify(None, train_dir=model, test_dir=classification)
    t3 = time.time()
    # creates the label file from the classifier output
    print "[run_iclassifier] creating the .label file"
    command = "cat %s/%s | egrep -v '^name' | egrep '\|.*\|' | python %s > %s/%s" \
              % (classification, 'iclassify.MaxEnt.out', 'invention_top_scores.py',
                 classification, label_file)
    print '   $', command
    subprocess.call(command, shell=True)
    t4 = time.time()
    process_label_file(corpus, classification, label_file, verbose)
    create_processing_time_file(classification, t1, t2, t3, t4)
    print


def create_info_files(corpus, model, filelist, classification):
    with open(os.path.join(classification, 'iclassify.info.general'), 'w') as fh:
        fh.write("$ python %s\n\n" % ' '.join(sys.argv))
        fh.write("corpus          =  %s\n" % corpus)
        fh.write("file_list       =  %s\n" % filelist)
        fh.write("model           =  %s\n" % model)
        fh.write("classification  =  %s\n" % classification)
        fh.write("git_commit      =  %s\n" % get_git_commit())
    shutil.copyfile(filelist, os.path.join(classification, 'iclassify.info.files'))

def create_processing_time_file(classification, t1, t2, t3, t4):
    now = time.time()
    with open(os.path.join(classification, 'iclassify.info.processing_time'),'w') as fh:
        fh.write("Processing time in seconds:\n\n")
        fh.write("   total                      %6d\n\n" % (now - t1))
        fh.write("   creating mallet file       %6d\n" % (t2 - t1))
        fh.write("   classifying                %6d\n" % (t3 - t2))
        fh.write("   creating the label file    %6d\n" % (t4 - t3))
        fh.write("   processing label file      %6d\n\n" % (now - t4))

def process_label_file(corpus, classification, label_file, verbose):
    """Takes the file with the labels and generates various derived data."""
    if verbose:
        print "[process_label_file] processing the label file"
    invention.merge_scores(corpus, classification, label_file) # has lang=en keyword
    generate_tab_format(classification, verbose)
    generate_relations(classification, verbose)


def generate_tab_format(classification, verbose=False):
    """Creates iclassify.MaxEnt.label.merged.tab, the tabulated format of the
    merged file. It has the same information as the merged file except that it
    does not print the title of the patent."""

    fields =  [('invention type', 't'),
               ('invention descriptors', 'i'),
               ('contextual terms', 'ct'),
               ('components/attributes', 'ca')]

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
        if line.startswith('title: ['):
            idx = line.find(']')
            filename = line[8:idx]
            patent_id = get_patent_id_from_filename(filename)
        elif line.strip() == '':
            patent_id = None
        else:
            for field, abbrev in fields:
                if line.startswith(field):
                    vals = line[len(field)+1:].strip()
                    #print '>>>', patent_id, abbrev, vals
                    if patent_id is not None and vals:
                        for val in vals.split(', '):
                            fh_out.write("%s\t%s\t%s\t%s\n" \
                                         % (patent_id, filename, abbrev, val))


def generate_relations(classification, verbose=False):
    """Creates iclassify.MaxEnt.label.relations.tab, a file with relations
    between terms. Relations are 'i-ct' (relation is between an invention and a
    contextual term in the same patent), 'i-ca' (relation between invention and
    a component/attribute) and ca-ca (relation of two terms that are both
    components/attributes of the same invention)."""

    def print_rels(terms, fh):
        i_terms = sorted(terms['i'])
        ca_terms = sorted(terms['ca'])
        ct_terms = sorted(terms['ct'])
        for i in i_terms:
            for ca in ca_terms: fh.write("%s\t%s\t%s\n" % ('i-ca', i, ca))
            for ct in ct_terms: fh.write("%s\t%s\t%s\n" % ('i-ct', i, ct))
        for ca1 in ca_terms:
            for ca2 in ca_terms:
                if ca1 < ca2:
                    fh.write("%s\t%s\t%s\n" % ('ca-ca', ca1, ca2))

    infile = os.path.join(classification, 'iclassify.MaxEnt.label.merged.tab')
    outfile = os.path.join(classification, 'iclassify.MaxEnt.label.relations.tab')
    fh_in = codecs.open(infile)
    fh_out = codecs.open(outfile, 'w')

    rels = ['i', 'ca', 'ct']
    current_patent_id = None
    current_terms = { 'i':[], 'ct':[], 'ca':[] }

    for line in fh_in:
        patent_id, fname, rel, term = line.rstrip("\n\r").split("\t")
        #print (patent_id, fname, rel, term)
        if patent_id != current_patent_id:
            print_rels(current_terms, fh_out)
            current_patent_id = patent_id
            current_terms = { 'i':[], 'ct':[], 'ca':[] }
        if rel in rels:
            current_terms[rel].append(term)
    print_rels(current_terms, fh_out)


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


def run_itrainer(corpus, filelist, model, features, annotation_file,
                 phr_feats_file=None, verbose=False):

    mallet_file = os.path.join(model, 'itrain.mallet')
    phr_feats_file = os.path.join(model, 'keyfeats.ta.dat')
    ensure_path(model)
    _itrainer_create_info_file(corpus, model, filelist, features, annotation_file)
    _itrainer_create_dat_file(phr_feats_file, corpus, filelist)
    _itrainer_create_mallet_file(annotation_file, phr_feats_file, mallet_file)
    patent_invention_train(mallet_file)


def _itrainer_create_info_file(corpus, model, filelist, features, annotation):
    with open(os.path.join(model, 'itrain.info.general'), 'w') as fh:
        fh.write("$ python %s\n\n" % ' '.join(sys.argv))
        fh.write("corpus          =  %s\n" % corpus)
        fh.write("file_list       =  %s\n" % filelist)
        fh.write("model           =  %s\n" % model)
        fh.write("features        =  %s\n" % features)
        fh.write("anotation       =  %s\n" % annotation)
        fh.write("git_commit      =  %s\n" % get_git_commit())
    shutil.copyfile(annotation, os.path.join(model, 'itrain.info.annotations'))
    shutil.copyfile(filelist, os.path.join(model, 'itrain.info.files'))

def _itrainer_create_dat_file(phr_feats_file, corpus, filelist):
    """Create the keyfeats.ta.dat file, which is a concatenation of all the
    files in filelist, but using only the first 100 terms in each file (because
    annotation does not go beyond those 100)."""
    print "[_itrainer_create_dat_file] creating", phr_feats_file
    print "[_itrainer_create_dat_file] from", corpus
    phr_feats_fh = codecs.open(phr_feats_file, 'w', encoding='utf-8')
    for line in open(filelist):
        (year, full_path, short_path) = line.split()
        # TODO: this is a hack, change this to use the filename generator and
        # the default_config and such
        fname = os.path.join(corpus, 'data/d3_phr_feats/01/files', short_path) # + '.gz')
        fh = open_input_file(fname)
        for line in fh:
            term_no = int(line.split()[0].split('_')[1])
            # no need to get too far into the file
            if term_no > 100: break
            phr_feats_fh.write(line)
    phr_feats_fh.close()

def _itrainer_create_mallet_file(annot_file, phr_feats_file, mallet_file):
    print "[_itrainer_create_mallet_file] creating", mallet_file
    print "[_itrainer_create_mallet_file] using annotations in", os.path.basename(annot_file)
    create_mallet_training_file(annot_file, phr_feats_file, mallet_file)

def _itrainer_create_mallet_file_old(annot_file):
    # this is how Peter used to create a mallet file from a dat file, which was
    # created from phr_feats files
    annotation_dir = "/home/j/anick/patent-classifier/ontology/annotation"
    phr_feats_file = os.path.join(annotation_dir, "en/invention/general/keyfeats.ta.20130509.dat")
    mallet_training_file  = "data/models/itrain.mallet"
    create_mallet_training_file(annot_file, phr_feats_file, mallet_training_file)




if __name__ == '__main__':

    # some defaults from Peter's original code
    corpus = "/home/j/anick/patent-classifier/ontology/creation/data/patents/cs_2002_subset/"
    annotation_dir = "/home/j/anick/patent-classifier/ontology/annotation"
    annotation_file = os.path.join(annotation_dir, "en/invention/general/key.ta.20130510.lab")
    phr_feats_file = os.path.join(annotation_dir, "en/invention/general/keyfeats.ta.20130509.dat")

    # default locations of the statistical model (itrain.model) for
    # classification and the classification
    model = 'data/models/inventions-standard-20130713'
    classification = os.path.join(os.getcwd(), 'ws')

    train = False
    classify = False
    filelist = None
    features = 'invention.features'
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
        elif opt == '--features': features = val
        elif opt == '--annotation-file': annotation_file = val
        elif opt == '--verbose': verbose = True

    if filelist is None:    
        filelist = os.path.join(corpus, "config/files.txt")

    if train:
        run_itrainer(corpus, filelist, model, features, annotation_file,
                     phr_feats_file=phr_feats_file, verbose=verbose)
    elif classify:
        run_iclassifier(corpus, filelist, model, classification, verbose=verbose)
    else:
        print "WARNING: nothing to do."
