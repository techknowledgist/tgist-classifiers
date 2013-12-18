
This directory contains code to create classifier models and run the classifiers
on unlabeled data.

There are several steps that need to be taken. Steps 1-3 are not needed once
satisfactory models are created.


1. creating a .mallet file

First step in creating a model is to generate a .mallet file from the d3_feats
files in a corpus, given an list of labeled examples For this use the following script:

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

$ python select_features.py OPTIONS


4. running the classifier

This is done with run

$ python run_tclassify.py OPTIONS
