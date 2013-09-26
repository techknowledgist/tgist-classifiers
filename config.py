"""

File with configuration settings. Intended to replace all previous configuration
files, which were named inconsistently and which duplicated some code. Used all
caps for all variables that are intended to be consumed by other scripts, which
makes it easier to recognize when variables from this file are used.

Configuration settings in this file:
- general settings
- mallet location

"""

import os, sys


# First some code to determine what machine we are running this on, will be used
# to determine locations.

# TODO: this feels a bit hackish, may want to think about a more elegant
# solution; this code is also repeated in ../creation/config.py.

script_path = os.path.abspath(sys.argv[0])
if script_path.startswith('/shared/home'):
    location = 'FUSENET'
elif script_path.startswith('/home/j/'):
    location = 'BRANDEIS'
elif script_path.startswith('/Users/'): 
    location = 'MAC'
elif script_path.startswith('/home/sean'):
    location = 'MAC'
else:
    print "WARNING: could not determine the location"
    location = None


### General settings
### -----------------------------------------------------------------------

DATA_TYPES = \
    ['d0_xml', 'd1_txt', 'd2_tag', 'd2_seg', 'd3_phr_feats']


### MALLET settings
### -----------------------------------------------------------------------

MALLET_RELEASE = '2.0.7'

# mallet directory, note that there should be no trailing slash in the directory name
if location == 'FUSENET':
    # location on the fuse VM
    MALLET_DIR = "/home/fuse/tools/mallet/mallet-2.0.7/bin"
elif location == 'MAC':
    # assumed location on any Mac
    MALLET_DIR = '/Applications/ADDED/nlp/mallet/mallet-2.0.7/bin'
else:
    # location on the department machines
    MALLET_DIR = "/home/j/corpuswork/fuse/code/patent-classifier/tools/mallet/mallet-2.0.7/bin"
