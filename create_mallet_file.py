"""Script to create a .mallet file from a corpus. The .mallet file is defined by
the corpus, a pipeline used to process the corpus, a file list (which defaults
to all files in the corpus) and a file with labeled instances.

Usage:

  $ python malletfile_create.py OPTIONS

Options:

  --model-dir PATH - the directory where the mallet file will be created; this
      is also the directory where derivative mallet files (after downsampling
      and feature selection) and models created from mallet files will be
      written

  --corpus PATH - corpus directory, d3_feats files are taken from this corpus

  --pipeline FILENAME - file with pipeline configuration; this is used to pick
      out the data set created with that pipeline; the file is assumed to be in
      the config directory of the corpus; the default is 'pipeline-default.txt'.

  --filelist FILENAME - contains files to process, that is, the elements from
      the data set used to create the model; this is an absolute or relative
      path to a file (if there is no path, the config directory in the corpus is
      used)

  --annotation-file FILENAME - this specifies a path to the file with labeled
      terms, these terms are used to created positive and negative instances
      from terms and contexts in the --filelist file

  --annotation-count INTEGER - number of lines to take from the annotation file,
      the default is to use all labeled terms

  --verbose - switches on verbose mode

Note that unlike with the document processing step there is no language option,
this reflects the mostly language independent nature of this step. Of course,
the corpus itself has the lanuage in its configuration file. Also, as we will
see below, language-specific information like annotated terms can be handed in
to the process.

Typical invocation:

   $ python create_mallet_file.py \
       --corpus ../creation/data/patents/201312-en-500 \
       --model-dir data/models/technologies-201312-en-500-010 \
       --annotation-file ../annotation/en/technology/phr_occ.lab \
       --filelist files-010.txt
    
This takes the corpus in ../creation/data/patents/201312-en-500 and creates a
mallet file in data/models/en-010 as well as a directry with information
files. As a convention, it is probably a good idea to let the name of the model
directory reflect the corpus, the subset of the corpus, and perhaps the
annotation set. The labels used are from ../annotation/en/technology/phr_occ.lab
and the default to use all files in the corpus is overuled by using the list in
../creation/data/patents/201312-en-500/config/files-010.txt.

"""

import os, sys, shutil, getopt, codecs, time

sys.path.append(os.path.abspath('../..'))

import mallet

from ontology.utils.batch import RuntimeConfig
from ontology.utils.batch import find_input_dataset, check_file_availability
from ontology.utils.file import filename_generator, ensure_path
from ontology.utils.git import get_git_commit


VERBOSE = False


class TrainerClassifier(object):

    """Abstract class with some common methods for the trainer and the
    classifier."""
    
    def _find_datasets(self):
        """Select data sets and check whether all files are available."""
        self.input_dataset = find_input_dataset(self.rconfig, 'd3_phr_feats')
        check_file_availability(self.input_dataset, self.file_list)

    def _find_filelist(self, file_list):
        if os.path.exists(file_list):
            return file_list
        return os.path.join(self.rconfig.config_dir, file_list)


class MalletFileCreator(TrainerClassifier):

    """Class that takes care of all the housekeeping around a call to the train
    module. Its purpose is to create a mallet mmodel while keeping track of
    processing configurations and writing statistics."""

    def __init__(self, rconfig, file_list, annotation_file, annotation_count):
        """Store parameters and initialize file names."""
        self.rconfig = rconfig
        self.file_list = self._find_filelist(file_list)
        self.annotation_file = annotation_file
        self.annotation_count = annotation_count
        self.model = rconfig.model
        self.data_dir = os.path.join(rconfig.corpus, 'data')
        self.train_dir = rconfig.model
        self.info_dir = os.path.join(self.train_dir, "info")
        self.info_file_general = os.path.join(self.info_dir, "train.info.general.txt")
        self.info_file_annotation = os.path.join(self.info_dir, "train.info.annotation.txt")
        self.info_file_config = os.path.join(self.info_dir, "train.info.config.txt")
        self.info_file_filelist = os.path.join(self.info_dir, "train.info.filelist.txt")
        self.info_file_stats = os.path.join(self.info_dir, "train.info.stats.txt")
        self.mallet_file = os.path.join(self.train_dir, "train.mallet")

    def run(self):
        """Run the trainer by finding the input data and building a model from it. Also
        writes files with information on configuration settings, features, gold standard
        term annotations and other things required to reproduce the model."""
        if os.path.exists(self.train_dir):
            exit("WARNING: Classifier model %s already exists" % self.train_dir)
        ensure_path(self.train_dir)
        ensure_path(self.info_dir)
        self._find_datasets()
        self._create_info_files()
        self._create_mallet_file()
        
    def _create_info_files(self):
        """Create the info files that together give a complete picture of the
        configuration of the classifier as it ran. This is partially done by copying
        external files into the local directory."""
        self._create_info_general_file()
        self._create_info_annotation_file()
        shutil.copyfile(self.rconfig.pipeline_config_file, self.info_file_config)
        shutil.copyfile(self.file_list, self.info_file_filelist)

    def _create_info_general_file(self):
        with open(self.info_file_general, 'w') as fh:
            fh.write("$ python %s\n\n" % ' '.join(sys.argv))
            fh.write("model             =  %s\n" % os.path.abspath(self.model))
            fh.write("file_list         =  %s\n" % os.path.abspath(self.file_list))
            fh.write("annotation_file   =  %s\n" % os.path.abspath(self.annotation_file))
            fh.write("annotation_count  =  %s\n" % self.annotation_count)
            fh.write("config_file       =  %s\n" % \
                     os.path.abspath(rconfig.pipeline_config_file))
            fh.write("timestamp         =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
            fh.write("git_commit        =  %s\n" % get_git_commit())

    def _create_info_annotation_file(self):
        with codecs.open(self.annotation_file) as fh1:
            with codecs.open(self.info_file_annotation, 'w') as fh2:
                written = 0
                for line in fh1:
                    if line.startswith('y') or line.startswith('n'):
                        written += 1
                        if written > self.annotation_count:
                            break
                        fh2.write(line)

    def _create_info_stats_file(self, labeled, unlabeled, terms, terms_y, terms_n):
        with codecs.open(self.info_file_stats, 'w') as fh:
            fh.write("Unlabeled instances     %6d\n" % unlabeled)
            fh.write("\nLabeled instances       %6d\n\n" % labeled)
            fh.write("  positive instances    %6d\n" % sum(terms_y.values()))
            fh.write("  negative instances    %6d\n" % sum(terms_n.values()))
            fh.write("\nLabeled types:          %6d\n\n" % len(terms))
            fh.write("  positive types        %6d\n" % len(terms_y))
            fh.write("  negative types        %6d\n" % len(terms_n))
            fh.write("\n\nTerms with positive training instances:\n\n")
            for term in sorted(terms_y.keys()):
                fh.write("   %6d   %s\n" % (terms_y[term], term))
            fh.write("\n\nTerms with negative training instances:\n\n")
            for term in sorted(terms_n.keys()):
                fh.write("   %6d   %s\n" % (terms_n[term], term))

    def _create_mallet_file(self):
        self._load_phrase_labels()
        mconfig = mallet.MalletConfig(
            self.model, 'train', 'classify', '0', self.model, '/tmp',
            classifier_type="MaxEnt", number_xval=0, training_portion=0,
            prune_p=False, infogain_pruning="5000", count_pruning="3")
        mtr = mallet.MalletTraining(mconfig)
        fnames = filename_generator(self.input_dataset.path, self.file_list)
        mtr.make_utraining_file3(fnames, self.d_phr2label, verbose=VERBOSE)
        self._create_info_stats_file(mtr.stats_labeled_count, mtr.stats_unlabeled_count,
                                     mtr.stats_terms, mtr.stats_terms_y, mtr.stats_terms_n)

    def _load_phrase_labels(self):
        """Use the label-term pairs in label_file to populate a dictionary of labeled
        phrases with their labels. Only labels used are 'y' and 'n', all others
        (that is, the empty string and the question mark) are ignored. """
        self.d_phr2label = {}
        with codecs.open(self.annotation_file, encoding='utf-8') as s_label_file:
            count = 0
            for line in s_label_file:
                count += 1
                if count > self.annotation_count:
                    break
                (label, phrase) = line.strip("\n").split("\t")
                # only useful labels are y and n
                if label in ("y", "n"):
                    self.d_phr2label[phrase] = label


def read_opts():
    longopts = ['corpus=', 'language=', 'pipeline=', 'filelist=',
                'annotation-file=', 'annotation-count=',
                'batch=', 'model-dir=', 'verbose' ]
    try:
        return getopt.getopt(sys.argv[1:], '', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))



if __name__ == '__main__':

    # default values of options
    model_path = None
    corpus_path = None
    file_list = 'files.txt'
    pipeline_config = 'pipeline-default.txt'
    annotation_file = None
    annotation_count = sys.maxint

    (opts, args) = read_opts()
    for opt, val in opts:
        if opt == '--corpus': corpus_path = val
        elif opt == '--model-dir': model_path = val
        elif opt == '--filelist': file_list = val
        elif opt == '--annotation-file': annotation_file = val
        elif opt == '--annotation-count': annotation_count = int(val)
        elif opt == '--pipeline': pipeline_config = val
        elif opt == '--verbose': VERBOSE = True

    if corpus_path is None: exit("WARNING: no corpus specified, exiting...\n")
    if model_path is None: exit("WARNING: no model directory specified, exiting...\n")
    if annotation_file is None: exit("WARNING: no annotation file specified, exiting...\n")

        
    # there is no language to hand in to the runtime config, but it will be
    # plucked from the general configuration if needed
    rconfig = RuntimeConfig(corpus_path, model_path, None, None, pipeline_config)
    if VERBOSE: rconfig.pp()
    MalletFileCreator(rconfig, file_list, annotation_file, annotation_count).run()
