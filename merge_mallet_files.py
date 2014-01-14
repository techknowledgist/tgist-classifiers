"""

merge_mallet_files.py

Merge several mallet files into one. Creates a new directory with train.mallet
file and an info file.

Usage:

    $ python merge_mallet_files.py TARGET_DIR SOURCE1 SOURCE2 ...

    TARGET_DIR is the directory where the new merged mallet file is written as
    train.mallet. The directory will also hold an info file that lists all the
    sources.

    SOURCE1 etcetera are mallet files or unix regular expressions that can be
    expanded into lists of mallet files.

It is up to the user to make sure that these are models that can actually be
merged meaningfully. Most importantly, these need to be models that were created
with the same initial annotation settings in the create_mallet_file.py
script. That is, the mallet files must have been generated from the same
annotation set. It is also important that the mallet files have the same feature
set.

It is not necessary though that the mallet files are all from the same
downsample threshold, one could for example imagine that for some years we want
to have different thresholds (for example, higher thresholds for recent years if
we want to build a model for classifying current patents).


Examples:

    $ python merge_models.py \
        /home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/merged-1997-2007 \
        '/home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/????/train.ds0200.mallet'

        Merge the ds0200 models in model directories 1997-2007 (since they all match '????').

    $ python merge_mallet_files.py \
        /home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/merged-97-99 \
        '/home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/19[0-9]*/train.ds0200.mallet'

        Merge the ds0200 models in model directories 1997-1999 (matching '19[0-9]*').
        
    $ python merge_mallet_files.py \
        /home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/merged-97-98 \
        '/home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/1997/train.ds0200.mallet'
        '/home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/1998/train.ds0200.mallet'

        Merge the ds0200 models in model directories 1997 and 1998. The single
        quotes are optional here.

"""


import os, sys, glob, time

sys.path.append(os.path.abspath('../..'))

from ontology.utils.file import ensure_path
from ontology.utils.git import get_git_commit


def merge_mallet_files(target_dir, mallet_files):
    t1 = time.time()
    target_file = os.path.join(target_dir, 'train.mallet')
    info_file = os.path.join(target_dir, 'train.mallet.info')
    print "\nMerging"
    for f in mallet_files:
        print '  ', f
    print "Target mallet file\n  ", target_file
    merge_command = "cat %s > %s" % (' '.join(mallet_files), target_file)
    print "\n$", merge_command, "\n"
    ensure_path(target_dir)
    os.system(merge_command)
    write_info(info_file, mallet_files, t1)

def write_info(info_file, mallet_files, t1):
    fh = open(info_file, 'w')
    fh.write("$ python %s\n\n" % ' '.join(sys.argv))
    fh.write("timestamp         =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
    fh.write("seconds elapsed   =  %d\n" % int(time.time() - t1))
    fh.write("git_commit        =  %s\n\n" % get_git_commit())
    for f in mallet_files:
        fh.write("source            =  %s\n" % f)



if __name__ == '__main__':

    target_dir = sys.argv[1]
    mallet_files = []
    for exp in sys.argv[2:]:
        files = glob.glob(exp)
        mallet_files.extend(files)
    merge_mallet_files(target_dir, mallet_files)
