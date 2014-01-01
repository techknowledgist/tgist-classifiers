"""downsample.py

Downsampling is an operation on a .mallet file. The input is a mallet file with
the .mallet extension, containg mallet instances. The output is also a mallet
file, but has the number of instances per term limited by a threshold. The name
of the output file is generated automatically, by adding a string representing
the threshold, It is written to the directory of the input file. For example, if
the input file is named train.mallet and the threshold is 5, then the output
file will be named train.ds0005.mallet.


Usage:

   $ python downsample.py OPTIONS

Options:

   --source-mallet-file PATH - the input mallet file on which downsampling is
        performed

   --threshold INTEGER - the maximum number of instances to keep for each term

   --verbose - switch on verbose mode

Example:

    $ python downsample.py \
      --source-mallet-file /home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/1997/train.mallet \
      --threshold 100 \
      --verbose &


Memory and space considerations.

Downsampling requires loading the entire input mallet file into memory. Here are
a few statistics from ln-all-600k/models/1997, which has a basic mallet file
created from about 15K patents:

    81M Dec 31 15:29 train.ds0100.mallet
   140M Jan  1 10:16 train.ds0200.mallet
   820M Dec 31 12:13 train.mallet

Memory used when downsampling train.mallet is about 4GB in this case and takes
83 seconds. For ln-all-600k/models/2005, the largest year with about 44K
patents, it 12GB of memory and 267 seconds, so the process seems to scale
reasonable well..

"""

import os, sys, getopt, time, random, codecs

sys.path.append(os.path.abspath('../..'))
from ontology.utils.git import get_git_commit


VERBOSE = False


def downsample(source_mallet_file, threshold):
    t1 = time.time()
    target_mallet_file = _generate_output_filename(source_mallet_file, threshold)
    if os.path.exists(target_mallet_file):
        exit("WARNING: target file %s already exists, exiting...\n" % target_mallet_file)
    fh_in = codecs.open(source_mallet_file, encoding='utf-8')
    fh_out = codecs.open(target_mallet_file, 'w', encoding='utf-8')
    terms = _read_mallet_file(fh_in)
    label_y_1, label_y_2 = 0, 0
    label_n_1, label_n_2 = 0, 0
    for t in sorted(terms.keys()):
        mallet_instances = terms[t]
        random.shuffle(mallet_instances)
        for label, line in mallet_instances:
            if label == 'y': label_y_1 += 1
            elif label == 'n': label_n_1 += 1
        if len(mallet_instances) > threshold:
            mallet_instances = mallet_instances[:threshold]
            #print len(mallet_instances)
        for label, line in mallet_instances:
            if label == 'y': label_y_2 += 1
            elif label == 'n': label_n_2 += 1
            fh_out.write(line)
    t2 = time.time()
    info_string = _info_string(terms, label_y_1, label_y_2, label_n_1, label_n_2)
    if VERBOSE: print "\n" + info_string
    _write_info(source_mallet_file, target_mallet_file, info_string, threshold, t1, t2)


def _generate_output_filename(mallet_file, threshold):
    """Returns an output name that incorporates the threshold."""
    path, fname = os.path.split(mallet_file)
    prefix, suffix = fname.split('.mallet')
    fname = os.path.join(path, "%s.ds%04d.mallet%s" % (prefix, threshold, suffix))
    return fname
    
def _read_mallet_file(fh):
    terms = {}
    count = 0
    for line in fh:
        count += 1
        if VERBOSE and count % 10000 == 0:
            print count
        identifier, label = line.split()[0:2]
        term = identifier.split('|',2)[2]
        terms.setdefault(term,[]).append((label, line))
    return terms

def _info_string(terms, label_y_1, label_y_2, label_n_1, label_n_2):
    return \
        "Terms in .mallet file: %d\n\n" % len(terms) \
        + "positive instances before downsampling  %8d\n" % label_y_1 \
        + "positive instances after downsampling   %8d\n\n" % label_y_2 \
        + "negative instances before downsampling  %8d\n" % label_n_1 \
        + "negative instances after downsampling   %8d\n\n" % label_n_2 \
        + "total instances before downsampling     %8d\n" % (label_y_1 + label_n_1) \
        + "total instances after downsampling      %8d\n" % (label_y_2 + label_n_2)

def _write_info(source_file, target_file, info_string, threshold, t1, t2):
    if target_file.endswith('.gz'):
        target_file = target_file[:-3]
    fh = open(target_file + '.info', 'w')
    fh.write("$ python %s\n\n" % ' '.join(sys.argv))
    fh.write("source file       =  %s\n" % os.path.abspath(source_file))
    fh.write("target file       =  %s\n" % os.path.abspath(target_file))
    fh.write("threshold         =  %d\n" % threshold)
    fh.write("timestamp         =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
    fh.write("seconds elapsed   =  %d\n" % int(t2 - t1))
    fh.write("git_commit        =  %s" % get_git_commit())
    fh.write("\n\n" + info_string + "\n")




def read_opts():
    longopts = ['threshold=', 'source-mallet-file=', 'verbose' ]
    try:
        return getopt.getopt(sys.argv[1:], '', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))



if __name__ == '__main__':

    threshold = None
    source_mallet_file = None
    
    (opts, args) = read_opts()
    for opt, val in opts:
        if opt == '--threshold': threshold = int(val)
        elif opt == '--source-mallet-file': source_mallet_file = val
        elif opt == '--verbose': VERBOSE = True

    if threshold is None: exit("WARNING: no threshold specified, exiting...\n")
    if source_mallet_file is None: exit("WARNING: no source file specified, exiting...\n")
    
    downsample(source_mallet_file, threshold)
               
