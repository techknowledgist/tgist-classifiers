"""run_tclassify.py

Script that lets you run the classifier on a dataset.

Usage:

   $ python run_tclassify.py --classify OPTIONS
   $ python run_tclassify.py --evaluate OPTIONS
   $ python run_tclassify.py --show-data --corpus PATH
   $ python run_tclassify.py --show-pipeline --corpus PATH

For the first form, we have the following options:

   --corpus PATH - corpus directory

   --pipeline FILENAME - file with pipeline configuration; this is used to pick
        out the data set created with that pipeline; the file is assumed to be in
        the config directory of the corpus; the default is 'pipeline-default.txt'.

   --filelist FILENAME - contains files to process, that is, the elements from
        the data set used to create the model; this is an absolute or relative
        path to a file; the default is to use files.txt in CORPUS_PATH/config

   --xval INTEGER - cross-validation setting for classifier, if set to 0 (which
        is the default) then no cross-validation will be performed

   --model FILENAME - selects the model used for classification

   --batch DIRECTORY - name of the current batch being created, this is the
       directory where all data will be written to.

   --verbose - switches on verbose mode

For the second form, we assume an existing classification and compare it to a
gold standard. Only two options are needed, a third is optional:

   --batch DIRECTORY - name of the current batch created with the --classify
       option, this is also the directory where all evaluation data will be
       written to.

   --gold-standard FILENAME - file with labeled terms for evaluations, if this
       is specified the system results will be compared to this list

   --filter FILENAME - a file with terms to be ignored in the evaluation,
       typically the file with annotated terms for creating the training data

There are two forms that are there purely to print information about the corpus:

  --show-data        print available datasets, then exit
  --show-pipelines   print defined pipelines, then exit

Both these options require the --corpus option but nothing else (all other
options handed in will be ignored).


Examples.

For running the classifier, you just pick your corpus and a model and run the
classifier on a set of files defined by a pipeline and a file list. Here is a
typical invocation:

   $ python run_tclassify.py \
     --classify \
     --corpus ../creation/data/patents/201312-en-500/ \
     --filelist ../creation/data/patents/201312-en-500/config/files-010.txt \
     --model data/models/technologies-201312-en-500-010/train.ds0005.standard.model \
     --batch data/classifications/test2 \
     --verbose

For evaluation of the above-created classification, simply use the batch and
compare it to a gold standard:

   $ python run_tclassify.py \
     --evaluate \
     --batch data/classifications/test2 \
     --gold-standard ../annotation/en/technology/phr_occ.eval.lab \
     --verbose

The system will select classify.MaxEnt.out.s4.scores.sum.nr in the selected
batch of the corpus and consider that file to be the system response. Ideally,
the gold standard was manually created over the same files as the one in the
batch. The log file will contain all terms with gold label, system response and
system score.

"""

import os, sys, shutil, getopt, subprocess, codecs, time

import config
import evaluation
import train
import mallet

sys.path.append(os.path.abspath('../..'))

from ontology.classifier.utils.find_mallet_field_value_column import find_column
from ontology.classifier.utils.sum_scores import sum_scores
from ontology.utils.batch import RuntimeConfig, show_datasets, show_pipelines
from ontology.utils.batch import find_input_dataset, check_file_availability, Profiler
from ontology.utils.file import filename_generator, ensure_path, open_output_file, compress
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


class Classifier(TrainerClassifier):

    def __init__(self, rconfig, file_list, model, classifier_type, batch, use_all_chunks_p):

        self.rconfig = rconfig
        self.file_list = file_list
        self.model = model
        self.classifier = classifier_type
        self.batch = batch
        self.use_all_chunks_p = use_all_chunks_p
        self.input_dataset = None

        self.mallet_file = self.batch + os.sep + 'classify.mallet'
        self.info_file_general = os.path.join(self.batch, "classify.info.general.txt")
        self.info_file_config = os.path.join(self.batch, "classify.info.config.txt")
        self.info_file_filelist = os.path.join(self.batch, "classify.info.filelist.txt")

        self.results_file = os.path.join(self.batch, "classify.%s.out" % (self.classifier))
        self.stderr_file = os.path.join(self.batch, "classify.%s.stderr" % (self.classifier))

        base = os.path.join(self.batch, "classify.%s.out" % (self.classifier))
        self.classifier_output = base
        self.scores_s1 = base + ".s1.all_scores"
        self.scores_s2 = base + ".s2.y_scores"
        self.scores_s3 = base + ".s3.scores.sum"
        self.scores_s4_nr = base + ".s4.scores.sum.nr"
        self.scores_s4_az = base + ".s4.scores.sum.az"


    def run(self):
        t1 = time.time()
        if os.path.exists(self.info_file_general):
            sys.exit("WARNING: already have classifier results in %s" % self.batch)
        ensure_path(self.batch)
        self._find_datasets()
        self._create_mallet_file()
        self._run_classifier()
        self._calculate_scores()
        self._create_info_files(t1)
        for fname in (self.results_file, self.mallet_file, self.scores_s1):
            print "Compressing", fname
            compress(fname)

    def _calculate_scores(self):
        """Use the clasifier output files to generate a sorted list of technology terms
        with their probabilities. This is an alternative way of using the commands in
        patent_tech_scores.sh."""
        self._scores_s1_select_score_lines()
        self._scores_s2_select_scores()
        self._scores_s3_summing_scores()
        self._scores_s4_sort_scores()

    def run_score_command(self, command, message):
        if VERBOSE:
            prefix = os.path.join(self.rconfig.target_path, 
                                  'data', 't2_classify', self.batch)
            print "[--scores]", message
            print "[--scores]", command.replace(prefix + os.sep, '')
        subprocess.call(command, shell=True)

    def _scores_s1_select_score_lines(self):
        message = "select the line from the classifier output that contains the scores"
        command = "cat %s | egrep '^[0-9]' > %s" % (self.classifier_output, self.scores_s1)
        self.run_score_command(command, message)

    def _scores_s2_select_scores(self):
        column = find_column(self.scores_s1, 'y')
        if VERBOSE:
            print "[--scores] select 'y' score from column %s of %s" % \
                  (column, os.path.basename(self.scores_s1))
        fh_in = codecs.open(self.scores_s1)
        fh_out = codecs.open(self.scores_s2, 'w')
        for line in fh_in:
            fields = line.split()
            id = fields[0]
            score = float(fields[int(column)-1])
            fh_out.write("%s\t%.6f\n" % (id, score))

    def _scores_s3_summing_scores(self):
        if VERBOSE:
            print "[--scores] summing scores into", os.path.basename(self.scores_s3)
        sum_scores(self.scores_s2, self.scores_s3)

    def _scores_s4_sort_scores(self):
        message1 = "sort on average scores"
        message2 = "sort on terms"
        command1 = "cat %s | sort -k2,2 -nr -t\"\t\" > %s" % (self.scores_s3, self.scores_s4_nr)
        command2 = "cat %s | sort > %s" % (self.scores_s3, self.scores_s4_az)
        self.run_score_command(command1, message1)
        self.run_score_command(command2, message2)

    def _create_info_files(self, t1):
        with open(self.info_file_general, 'w') as fh:
            fh.write("$ python %s\n\n" % ' '.join(sys.argv))
            fh.write("batch            =  %s\n" % os.path.abspath(self.batch))
            fh.write("file_list        =  %s\n" % self.file_list)
            fh.write("model            =  %s\n" % self.model)
            fh.write("features         =  %s\n" % ' '.join(self.features))
            fh.write("config_file      =  %s\n" % os.path.basename(rconfig.pipeline_config_file))
            fh.write("timestamp        =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
            fh.write("processing time  =  %ds\n" % int(time.time() - t1))
            fh.write("git_commit       =  %s" % get_git_commit())
        shutil.copyfile(self.rconfig.pipeline_config_file, self.info_file_config)
        shutil.copyfile(self.file_list, self.info_file_filelist)

    def _create_mallet_file(self):
        fnames = filename_generator(self.input_dataset.path, self.file_list)
        fh = open_output_file(self.mallet_file, compress=False)
        self._set_features()
        if VERBOSE:
            print "[create_mallet_file] features: %s" % (self.features)
        features = dict([(f,True) for f in self.features])
        stats = { 'labeled_count': 0, 'unlabeled_count': 0, 'total_count': 0 }
        count = 0
        for fname in fnames:
            count += 1
            if VERBOSE:
                print "[create_mallet_file] %05d %s" % (count, fname)
            train.add_file_to_utraining_test_file(
                fname, fh, {}, features, stats, use_all_chunks_p=self.use_all_chunks_p)
        fh.close()
        if VERBOSE:
            print "[create_mallet_file]", stats

    def _run_classifier(self):
        mclassifier = mallet.SimpleMalletClassifier(
            config.MALLET_DIR,
            classifier_type=self.classifier)
        mclassifier.run_classifier(
            self.model,
            self.mallet_file,
            self.results_file,
            self.stderr_file)

    def _set_features(self):
        # TODO: it is a bit unclear now how we get the features, in the past
        # they came from the model.info file, recursing to parent info files if
        # needed, but now we find them in the mallet.info file, the question is
        # whether the features are always there and if they are the correct ones
        # in case we selected features twice; maybe never allow feature
        # selection to happen twice.
        if VERBOSE:
            print "[get_features] model file =", self.model
        info_file = os.path.splitext(self.model)[0] + '.mallet.info'
        print self.model
        print info_file
        features = parse_info_file(info_file)
        if features.has_key('features'):
            feature_set = frozenset(features['features'].split())
        else:
            print "WARNING: no features found, exiting."
            exit()
        self.features = sorted(list(feature_set))


def parse_info_file(fname):
    """Parse an info file and return a dictionary of features. Return None if the
    file does not exist. Assumes that the last feature of note is always
    git_commit."""
    if VERBOSE:
        print "[parse_info_file]", fname
    features = {}
    try:
        for line in open(fname):
            if line.find('=') > -1:
                f, v = line.split('=', 1)
                features[f.strip()] = v.strip()
                if f.strip() == 'git_commit':
                    break
        return features
    except IOError:
        return None


def evaluate(batch, gold_standard, tfilter):
    """Evaluate results in batch given a gold standard. It is the responsibility
    of the user to make sure that it makes sense to compare this gold standard
    to the system results."""
    # TODO: now the log files have a lot of redundancy, fix this
    summary_file = os.path.join(batch, "eval-results-summary.txt")
    summary_fh = open(summary_file, 'w')
    system_file = os.path.join(batch, 'classify.MaxEnt.out.s4.scores.sum.nr')
    command =  "python %s" % ' '.join(sys.argv)
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    #for threshold in (0.5,):
        for term_type in ('all', 'single-token-terms', 'multi-token-terms'):
            ttstring = term_type_as_short_string(term_type)
            tfstring = term_filter_as_short_string(tfilter)
            log_file = os.path.join(batch, "eval-results-%s-%s-%.1f.txt" \
                                    % (ttstring, tfstring, threshold))
            result = evaluation.test(gold_standard, system_file, threshold, log_file,
                                     term_type=term_type, term_filter=tfilter,
                                     debug_c=False, command=command)
            summary_fh.write(result)

def term_type_as_short_string(term_type):
    if term_type == 'all': return 'all'
    if term_type == 'single-token-terms': return 'stt'
    if term_type == 'multi-token-terms': return 'mtt'

def term_filter_as_short_string(term_filter):
    return 'ntf' if term_filter is None else 'ytf'


def read_opts():
    longopts = ['classify', 'evaluate', 'show-data', 'show-pipelines',
                'corpus=', 'language=', 'pipeline=', 'filelist=',
                'batch=', 'features=', 'xval=', 'model=', 'eval-on-unseen-terms',
                'verbose', 'gold-standard=', 'threshold=', 'filter=', 'logfile=']
    try:
        return getopt.getopt(sys.argv[1:], '', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))



if __name__ == '__main__':

    # default values of options
    corpus_path = None
    file_list = 'files.txt'
    pipeline_config = 'pipeline-default.txt'
    classify_p, evaluate_p = False, False
    show_data_p, show_pipelines_p = False, False
    model, batch, xval, = None, None, "0"
    use_all_chunks = True
    gold_standard = None
    filter_terms = None
    threshold = None

    (opts, args) = read_opts()

    for opt, val in opts:

        if opt == '--evaluate': evaluate_p = True
        elif opt == '--classify': classify_p = True
        elif opt == '--show-data': show_data_p = True
        elif opt == '--show-pipelines': show_pipelines_p = True

        elif opt == '--corpus': corpus_path = val
        elif opt == '--model': model = val
        elif opt == '--batch': batch = val
        elif opt == '--filelist': file_list = val

        elif opt == '--pipeline': pipeline_config = val
        elif opt == '--xval': xval = val

        elif opt == '--gold-standard': gold_standard = val
        elif opt == '--threshold': threshold = float(val)
        elif opt == '--filter': filter_terms = val

        elif opt == '--verbose': VERBOSE = True
        elif opt == '--eval-on-unseen-terms': use_all_chunks = False

    if corpus_path is None and not evaluate_p:
        exit("WARNING: no corpus specified, exiting...\n")

    # there is no language to hand in to the runtime config, but it will be
    # plucked from the general configuration if needed
    if not evaluate_p:
        rconfig = RuntimeConfig(corpus_path, model, batch, None, pipeline_config)

    if show_data_p:
        show_datasets(rconfig, config.DATA_TYPES, VERBOSE)
    elif show_pipelines_p:
        show_pipelines(rconfig)
    elif evaluate_p:
        evaluate(batch, gold_standard, filter_terms)
    elif classify_p:
        if VERBOSE: rconfig.pp()
        # allow for the file_list to be just the filename in the config
        # directory of the corpus
        if not os.path.exists(file_list):
            file_list = os.path.join(corpus_path, 'config', file_list)
        # TODO: we now just hand in MaxEnt as the classifier type because that
        # is what we always use, but really the model info should store the
        # classifier type selected and the classifier should just use that
        Classifier(rconfig, file_list, model, 'MaxEnt', batch,
                   use_all_chunks_p=use_all_chunks).run()
    else:
        print "\nWARNING: missing option\n"
        print "Usage:"
        print "   $ python run_tclassify.py --classify OPTIONS"
        print "   $ python run_tclassify.py --evaluate OPTIONS"
        print "   $ python run_tclassify.py --show-data --corpus PATH"
        print "   $ python run_tclassify.py --show-pipeline --corpus PATH"
