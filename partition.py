"""partition.py

Partitioning is an operation on a .mallet file. The input is a mallet file with
the .mallet extension, containg mallet instances. The output is two mallet
files, provding a exhaustive disjoint partition of the input file.

One of the output files, which has 'p-stt' as part of its name, has only the
single-token terms, and the other file, with 'p-mtt' as part of its name, has
all multi-token terms.

Usage:

   $ python downsample.py OPTIONS

Options:

   --mallet-file PATH - the input mallet file that is partitioned

   --verbose - switch on verbose mode

Example:

    $ python partition.py \
      --mallet-file /home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/1997/train.mallet \
      --verbose &


"""

import os, sys, getopt, time, codecs

sys.path.append(os.path.abspath('../..'))
from ontology.utils.git import get_git_commit


VERBOSE = False


def partition(mallet_file):
    t1 = time.time()
    out_file1, out_file2 = _generate_output_filenames(mallet_file)
    if os.path.exists(out_file1):
        exit("WARNING: target file %s already exists, exiting...\n" % out_file1)
    if os.path.exists(out_file2):
        exit("WARNING: target file %s already exists, exiting...\n" % out_file2)
    fh_in = codecs.open(mallet_file, encoding='utf-8')
    fh_out_stt = codecs.open(out_file1, 'w', encoding='utf-8')
    fh_out_mtt = codecs.open(out_file2, 'w', encoding='utf-8')
    count = 0
    for line in fh_in:
        count += 1
        #if count > 1000: break
        if VERBOSE and count % 10000 == 0:
            print count
        identifier, label = line.split()[0:2]
        term = identifier.split('|',2)[2]
        split_term = term.split('_')
        if len(split_term) == 1:
            fh_out_stt.write(line)
        elif len(split_term) > 1:
            fh_out_mtt.write(line)
        else:
            print "WARNING: unexpected term:", term
    _write_info(mallet_file, out_file1, out_file2, t1, time.time())


def _generate_output_filenames(mallet_file):
    """Returns an output name that incorporates the threshold."""
    path, fname = os.path.split(mallet_file)
    prefix, suffix = fname.split('.mallet')
    fname1 = os.path.join(path, "%s.p-stt.mallet%s" % (prefix, suffix))
    fname2 = os.path.join(path, "%s.p-mtt.mallet%s" % (prefix, suffix))
    return fname1, fname2
    
def _write_info(source_file, out_file1, out_file2, t1, t2):
    for fname in (out_file1, out_file2):
        fh = open(fname + '.info', 'w')
        fh.write("$ python %s\n\n" % ' '.join(sys.argv))
        fh.write("source file       =  %s\n" % os.path.abspath(source_file))
        fh.write("target file       =  %s\n" % os.path.abspath(fname))
        fh.write("timestamp         =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
        fh.write("seconds elapsed   =  %d\n" % int(t2 - t1))
        fh.write("git_commit        =  %s" % get_git_commit())




def read_opts():
    longopts = ['mallet-file=', 'verbose' ]
    try:
        return getopt.getopt(sys.argv[1:], 'm:v', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))



if __name__ == '__main__':

    threshold = None
    mallet_file = None
    
    (opts, args) = read_opts()
    for opt, val in opts:
        if opt in  ('-m', '--mallet-file'): mallet_file = val
        elif opt in ('-v', '--verbose'): VERBOSE = True

    if mallet_file is None: exit("WARNING: no source file specified, exiting...\n")
    
    partition(mallet_file)
               
