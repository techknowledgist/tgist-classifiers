"""

File with configuration settings. Intended to replace all previous configuration
files, which were named inconsistently and which duplicated some code. Used all
caps for all variables that are intended to be consumed by other scripts, which
makes it easier to recognize when variables from this file are used.

Configuration settings in this file:
- general settings
- processing pipeline
- mallet location
- stanford tool locations and settings
- settings for pipeline and patent_analyzer scripts

"""

import os, sys


# First some code to determine what machine we are running this on, will be used
# to determine locations.

# TODO: this feels a bit hackish, may want to think about a more elegant
# solution.

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

# default language
LANGUAGE = "en"

# annotations directory, general and for the language
ANNOTATION_DIRECTORY = "../annotation"
ANNOT_LANG_PATH = ANNOTATION_DIRECTORY + "/" + LANGUAGE


### Pipeline settings
### -----------------------------------------------------------------------

# definition of default configuration file and the default pipeline
# configurations

DEFAULT_PIPELINE_CONFIGURATION_FILE = 'pipeline-default.txt'

DEFAULT_PIPELINE = """
# This file contains the default pipeline configuration settings. Settings in
# here can be overruled by handing the step2_document_processing script the
# identifier for another configuration file. All pipeline configuration files
# live inside of the config directory configuration file.

--populate
--xml2txt
--txt2tag
--tag2chk --candidate-filter=off --chunker-rules=en
"""

DEFAULT_PIPELINE_CN = """
# This file contains the default pipeline configuration settings for Chinese. Settings in
# here can be overruled by handing the step2_document_processing script the identifier for
# another configuration file. All pipeline configuration files live inside of the config
# directory configuration file.

--populate
--xml2txt
--txt2seg
--seg2tag
--tag2chk --candidate-filter=off --chunker-rules=cn
"""

# definition of sub directory names for processing stages
DATA_TYPES = \
    ['d0_xml', 'd1_txt', 'd2_tag', 'd2_seg', 'd3_phr_feats']
PROCESSING_AREAS = \
    DATA_TYPES + ['t0_annotate', 't1_train', 't2_classify', 't3_test',
                  'o1_index', 'o2_matcher', 'o3_selector', 'workspace' ]


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


### Stanford parser/segmenter settings
### -----------------------------------------------------------------------

STANFORD_TAGGER_RELEASE = "2012-07-09"
STANFORD_SEGMENTER_RELEASE = "2012-07-09"

# tagger and segmenter location
if location == 'BRANDEIS':
    base_dir = "/home/j/corpuswork/fuse/code/patent-classifier/tools/stanford/"
    STANFORD_TAGGER_DIR = base_dir + "stanford-postagger-full-2012-07-09" 
    STANFORD_SEGMENTER_DIR = base_dir + "stanford-segmenter-2012-07-09"
elif location == 'FUSENET':
    STANFORD_TAGGER_DIR = "/home/fuse/tools/stanford-postagger-full-2012-07-09" 
    STANFORD_SEGMENTER_DIR = "/home/fuse/tools/stanford-segmenter-2012-07-09"
elif location == 'MAC':
    base_dir = '/Applications/ADDED/nlp/stanford/'
    STANFORD_TAGGER_DIR = base_dir + "stanford-postagger-full-2012-07-09"
    STANFORD_SEGMENTER_DIR = base_dir + "stanford-segmenter-2012-07-09"
else:
    # cobbel together a path local to the repository
    tools_path = os.path.join(script_path, '..', '..', 'tools')
    tools_path = os.path.abspath(tools_path)
    STANFORD_TAGGER_DIR = os.path.join(tools_path, "stanford-postagger-full-2012-07-09" )
    STANFORD_SEGMENTER_DIR = os.path.join(tools_path, "stanford-segmenter-2012-07-09")


# memory use for the stanford tagger and segmenter
STANFORD_MX = "2000m"

STANFORD_DEBUG_P = 1

STANFORD_SENTENCES = "newline"

# parameters for 3 kinds of output, the output format variable does not appear to be used
STANFORD_TAG_SEPARATOR = "_"
STANFORD_TOKENIZED = 0
STANFORD_OUTPUT_FORMAT = "wordsAndTags"
#STANFORD_OUTPUT_FORMAT = "wordsAndTags,penn,typedDependenciesCollapsed"

### use these parameters to test handling of pre-tagged input
# if tokenized = 1 and tag_separator = "/"
# then parser should use tags provided in input line
# tag_separator = "/"
# tokenized = 1
# output_format = "penn,typedDependenciesCollapsed"

# See discarded/sdp_config for Some older settings that are not used anymore



### configuration for data used by patent_analysis.py and pipeline.py
### -----------------------------------------------------------------------

DATA_ROOT = "data"

# for patent_analysis.py

# directory of patent xml files arranged into yearly subdirectories
EXTERNAL_PATENT_PATH = "/home/j/clp/chinese/corpora/fuse-patents/500-patents/DATA/Lexis-Nexis/US/Xml"

# location where patents are copied to local directory for processing steps.  The directory structure 
# arranges files by language/step/year
# The .xml qualifier is expected on file names in all subdirectories, so that the file
# names within the step subdirectories are identical.
WORKING_PATENT_PATH = os.path.join(DATA_ROOT, "patents")

# for pipeline.py

# For each RDG (related document group), there must be a filelist containing
# doc_id year external_file_path
EXTERNAL_RDG_FILELIST = os.path.join(DATA_ROOT, "external/en1.txt")

# For each RDG, create a local working directory
WORKING_RDG_PATH = os.path.join(DATA_ROOT, "working/rdg/en1")
