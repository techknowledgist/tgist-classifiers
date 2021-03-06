"""create_model.py

Create a model from a mallet file.

The name of the model is derived from the mallet file by replacing the .mallet
extension with a .model extension. Also creates a .info file and two .cinfo (the
last two in the info subdirectory). The intermediate vector file that is created
along the way is deleted after the model is created.


Usage:

   $ python create_model.py OPTIONS


Options:

   --mallet-file PATH - the input mallet file from which the model is created

   --verbose - switch on verbose mode, optional, defualt is verbose mode off

   --cinfo - create cinfo files with information on how informative features
        are, optional, the default is to not print this information

Several options not yet implemented, but that we need to add for completeness.

   --xval INT
   --classifier
   --training-portion
   --infogain-pruning
   --prune
   --count-pruning


Example:

   $ python create_model.py \
     --mallet-file data/models/technologies-201312-en-500-010/train.ds0005.mallet


Wishlist:
1- check whether the model and other files exist, give option to overwrite
2- maybe split off the --cinfo option and have a separate script
   create_classifier_info.py

"""

import os, sys, getopt, time 

import config
from mallet import SimpleMalletTrainer, run_command
from utils.git import get_git_commit
from utils.path import ensure_path

VERBOSE = False


def create_model(mallet_file, cinfo):
    t1 = time.time()
    vectors_file = _generate_filename(mallet_file, 'vectors')
    model_file = _generate_filename(mallet_file, 'model')
    out_file = _generate_filename(mallet_file, 'out')
    stderr_file = _generate_filename(mallet_file, 'stderr')
    cinfo_file = _generate_filename(mallet_file, 'cinfo')
    cinfo_file_sorted = _generate_filename(mallet_file, 'cinfo.sorted')
    mtrainer = SimpleMalletTrainer(config.MALLET_DIR)
    mtrainer.create_vectors(mallet_file, vectors_file)
    mtrainer.create_model(vectors_file, model_file, out_file, stderr_file)
    if cinfo:
        mtrainer.create_cinfo(model_file, cinfo_file, cinfo_file_sorted)
        _cleanup_cinfo(cinfo_file, cinfo_file_sorted)
    _write_info(mallet_file, model_file, mtrainer, out_file, stderr_file, t1)
    _cleanup(out_file, stderr_file, vectors_file)

def _generate_filename(mallet_file, extension):
    """Returns a file name by replacing the .mallet extension with another extension."""
    return mallet_file[:-6] + extension

def _write_info(mallet_file, model_file, mtrainer, out_file, stderr_file, t1):
    fh = open(model_file + '.info', 'w')
    fh.write("$ python %s\n\n" % ' '.join(sys.argv))
    fh.write("mallet file       =  %s\n" % os.path.abspath(mallet_file))
    fh.write("model file        =  %s\n" % os.path.abspath(model_file))
    fh.write("trainer settings  =  %s\n" % mtrainer.settings())
    fh.write("timestamp         =  %s\n" % time.strftime("%Y%m%d:%H%M%S"))
    fh.write("time elapsed      =  %ds\n" % (time.time() - t1))
    fh.write("git_commit        =  %s\n\n" % get_git_commit())
    fh.write("$ %s\n\n" % mtrainer.saved_create_vectors_command)
    fh.write("$ %s\n" % mtrainer.saved_create_model_command)
    fh.write("\nContents of .out file:\n\n")
    for line in open(out_file):
        fh.write("    %s" % line)
    fh.write("\nContents of .stderr file:\n\n")
    for line in open(stderr_file):
        line = line.replace("\f", "\n    ")
        line = line.replace("\r", "\n    ")
        fh.write("    %s" % line)
    for cmd in mtrainer.saved_create_cinfo_commands:
        fh.write("\n$ %s\n" % cmd)
        
def _cleanup_cinfo(cinfo_file, cinfo_file_sorted):
    run_command("gzip %s" % cinfo_file)
    run_command("gzip %s" % cinfo_file_sorted)
    info_dir = os.path.dirname(cinfo_file) + os.sep + 'info'
    ensure_path(info_dir)
    run_command("mv %s.gz %s" % (cinfo_file, info_dir))
    run_command("mv %s.gz %s" % (cinfo_file_sorted, info_dir))

def _cleanup(out_file, stderr_file, vectors_file):
    for f in (out_file, stderr_file, vectors_file):
        run_command("rm %s" % f)

def read_opts():
    longopts = ['mallet-file=', 'verbose', 'cinfo' ]
    try:
        return getopt.getopt(sys.argv[1:], 'm:v', longopts)
    except getopt.GetoptError as e:
        sys.exit("ERROR: " + str(e))



if __name__ == '__main__':

    mallet_file = None
    cinfo_p = False
    
    (opts, args) = read_opts()
    for opt, val in opts:
        if opt in ['-m', '--mallet-file']: mallet_file = val
        elif opt in ['-v', '--verbose']: VERBOSE = True
        elif opt == '--cinfo': cinfo_p = True
        # TODO: will need more options (xval etcetera)
        
    if mallet_file is None: exit("WARNING: no mallet file specified, exiting...\n")

    create_model(mallet_file, cinfo_p)

