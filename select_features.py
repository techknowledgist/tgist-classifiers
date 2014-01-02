"""select_features.py

Feature selection is an operation on a mallet file. The output is a mallet file
where all features that are not specified in the feature list are removed. The
feature list is given in a file, with one line for each feature. If the file is
empty, no features will be kept.

The name of the new file is derived from the old file, basically by expanding
the filename with the name of the feature file. For example, if the input file
is named train.ds00100.mallet and the feature file is 'extint, then the output
file will be named train.ds00100.extint.mallet.


Usage:

   $ python select_features.py OPTIONS


Options:

   --source-mallet-file PATH
       the input mallet file from which features are selected

   --features NAME
        name of the features file in features, the .features extension is not
        included

   --verbose
        switch on verbose mode


Example:

   $ python select_features.py \
        --source-mallet-file data/models/technologies-201312-en-500-010/train.ds0005.mallet \
        --features standard


Wishlist:

1- Add a way to keep all features, this is now only possible by specifying each
   feature generated for the phr_feats files.

2- Add a way to remove specified features, rather than keeping specified features.

"""

import os, sys, getopt, time, random, codecs

from mallet import MalletTraining

sys.path.append(os.path.abspath('../..'))

from ontology.utils.git import get_git_commit


VERBOSE = False


def select_features(source_mallet_file, features):
    t1 = time.time()
    target_mallet_file = _generate_output_filename(source_mallet_file, features)
    if os.path.exists(target_mallet_file):
        exit("WARNING: target file %s already exists, exiting...\n" % target_mallet_file)
    if VERBOSE:
        print "[select_features] creating", target_mallet_file
    fh_in = codecs.open(source_mallet_file, encoding='utf-8')
    fh_out = codecs.open(target_mallet_file, 'w', encoding='utf-8')
    # create an instance without a MalletCOnfig, it is just used for reading the features
    mtr = MalletTraining(None, features)
    if VERBOSE:
        print "[select_features]", sorted(mtr.d_features.keys())
    count = 0
    for line in fh_in:
        count += 1
        #if count > 30000: break
        if count % 10000 == 0: print count
        mallet_instance = MalletInstance(line)
        mallet_instance.filter(mtr.d_features)
        fh_out.write("%s\n" % mallet_instance.as_line())
    _write_info(source_mallet_file, target_mallet_file, mtr.features_file, mtr.d_features, t1)

def _generate_output_filename(mallet_file, features):
    """Returns an output name that incorporates the features."""
    # TODO: this is very similar to the same function in downsample.py, should
    # merge the two
    path, fname = os.path.split(mallet_file)
    prefix, suffix = fname.split('.mallet')
    feature_set = os.path.splitext(os.path.basename(features))[0]
    fname = os.path.join(path, "%s.%s.mallet%s" % (prefix, feature_set, suffix))
    return fname

def _write_info(source_file, target_file, feats_file, feats, t1):
    fh = open(target_file + '.info', 'w')
    fh.write("$ python %s\n\n" % ' '.join(sys.argv))
    fh.write("source file       =  %s\n" % os.path.abspath(source_file))
    fh.write("target file       =  %s\n" % os.path.abspath(target_file))
    fh.write("features file     =  %s\n" % feats_file)
    fh.write("features          =  %s\n" % ' '.join(sorted(feats.keys())))
    fh.write("timestamp         =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
    fh.write("processing time   =  %ds\n" % (time.time() - t1))
    fh.write("git_commit        =  %s" % get_git_commit())


class MalletInstance(object):

    """Implements a mallet instance with unique identifier, label and features."""

    # TODO: should probably be in mallet.py, also because this will also be used
    # by run_tclassifier.py (actually, there already is such a class in
    # mallet.py, merge them)

    def __init__(self, line):
        self.line = line
        fields = line.split()
        self.identifier = fields[0]
        self.label = fields[1]
        self.features = [f.split('=', 1) for f in fields[2:]]

    def __str__(self):
        return "<MalletInstance id=%s label=%s\n %s>" % (self.identifier, self.label, self.features)
        return self.line

    def unique_features(self):
        return sorted(dict(self.features).keys())

    def as_line(self):
        return "%s %s %s" % (self.identifier, self.label, ' '.join(["%s=%s" % (k,v) for k,v in self.features]))

    def filter(self, feature_dict):
        # this first line was added for those case where the feature is screwed up
        self.features = [pair for pair in self.features if len(pair) == 2]
        self.features = [(k,v) for (k,v) in self.features if feature_dict.has_key(k)]


def read_opts():
    longopts = ['features=', 'source-mallet-file=', 'verbose' ]
    try:
        return getopt.getopt(sys.argv[1:], '', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))



if __name__ == '__main__':

    features = None
    source_mallet_file = None

    (opts, args) = read_opts()
    for opt, val in opts:
        if opt == '--features': features = val
        elif opt == '--source-mallet-file': source_mallet_file = val
        elif opt == '--verbose': VERBOSE = True

    if features is None: exit("WARNING: no features specified, exiting...\n")
    if source_mallet_file is None: exit("WARNING: no source file specified, exiting...\n")

    select_features(source_mallet_file, features)

