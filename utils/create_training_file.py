"""

Take the config/files.txt file and a config/files-testing.txt from a corpus and
create a file config/files-training.txt, which has all lines in files.txt except
for those in files-testing.txt.

Usage:

    $ python create_training_file.py CORPUS_DIR?

    The CORPUS_DIR argument is optional, if it is not given, the default used
    will be '/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-sample-500/'

Assumes that files.txt has three columns (year, source_path, short_path),
files-training.txt will just have short_path (similar to files-testing.txt).

"""


import os, sys

corpus_dir = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-sample-500/'

if len(sys.argv) > 1:
    corpus_dir = sys.argv[1]


filelist = os.path.join(corpus_dir, 'config/files.txt')
test_files = os.path.join(corpus_dir, 'config/files-testing.txt')
train_files = os.path.join(corpus_dir, 'config/files-training.txt')

test_files_d = {}
with open(test_files) as fh:
    for line in fh:
        test_file = os.path.basename(line.strip())
        test_files_d[test_file] = True

with open(filelist) as fh:
    with open(train_files, 'w') as fh_train:
        for line in fh:
            short_path = line.split()[2]
            fname = os.path.basename(short_path)
            if not test_files_d.has_key(fname):
                fh_train.write(short_path + "\n")
