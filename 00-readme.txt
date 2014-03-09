
This directory contains code to create classifier models and run the classifiers
on unlabeled data.

There are several steps that need to be taken. Steps 1-3 are not needed once
satisfactory models are created.


1. creating a .mallet file

First step in creating a model is to generate a .mallet file from the d3_feats
files in a corpus, given an list of labeled examples. For this use the following
script:

$ python create_mallet_file.py OPTIONS

What it basically does is to (1) collect all lines from d3_feats files for all
labeled examples, (2) create the union of all instances for a term in the same
document, (3) write the results all to one big document. All features are
selected. See the script for more details.

The current version works for technology classification only, need a version for
inventions as well.


2. downsampling a mallet file

This takes a mallet file and creates a new mallet file by capping the number of
training instances for each term.

$ python downsample.py OPTIONS


3. feature selection on a mallet file

Takes a mallet file and creates a new mallet file where each feature vector only
contains the features specified in one of the options.

$ python select_features.py OPTIONS


4. creating a model

Takes a mallet file and creates a vector file and the model. Vector files are
deleted once the model is created.

$ python create_model OPTIONS


5. running the classifier

$ python run_tclassify.py OPTIONS



MEMORY

Memory settings are in the mallet bin directory, which on the Brandeis CS
system is in

/home/j/corpuswork/fuse/code/patent-classifier/tools/mallet/mallet-2.0.7/bin

The relevant files and the default settings are:

    classifier2info  -  2000m
    csv2vectors      -  4000m
    mallet           -  2g

When creating large models some of these may need to be edited. The settings
above are know to work for mallet files up to 267M (5k patents and 4k annotated
terms), but break on mallet files of 533M (10k patents). Here are some settings
for larger mallet files (10k, 20k and 50k patents):

   533M ==> mallet=4g
   1.1G ==> mallet=8g
   2.7G ==> csv2vectors=? mallet=?

The mallet script is used for both trainer and classfier and the same settings
can be used. Models build from larger mallet files do not appear to need more
memory, but for creating larger models memory needs to be increased as shown
above. Could consider adding a --memory option which can be used to overrule
this, but that is a tad complicated because the mallet scripts would also need
to be changed.
