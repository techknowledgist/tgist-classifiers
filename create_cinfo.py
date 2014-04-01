"""

Takes a Mallet model file and creates two cinfo files, one sorted and one unsorted.

Usage:

    $ python create_cinfo.py MODEL_FILE

    This creates two files: MODEL_FILE.cinfo and MODEL_FILE.cinfo.sorted
    
"""


import sys, config, time
from mallet import SimpleMalletTrainer, run_command


def create_cinfo(model_file):    
    t1 = time.time()
    cinfo_file = model_file + '.cinfo'
    cinfo_file_sorted = model_file + '.cinfo.sorted'
    mtrainer = SimpleMalletTrainer(config.MALLET_DIR)
    mtrainer.create_cinfo(model_file, cinfo_file, cinfo_file_sorted)
    print "time elapsed: %s seconds" % (time.time() - t1)


if __name__ == '__main__':
    model_file = sys.argv[1]
    create_cinfo(model_file)
