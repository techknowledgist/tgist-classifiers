
This directory contains code to create classifier models and run the classifiers
on unlabeled data.

There are several steps that need to be taken. Steps 1-3 are not needed once
satifactory models are created.


1. creating a .mallet file

First step in creating a model is to generate a .mallet file from the d3_feats
files in a corpus. For this use the following script:

$ python create_mallet_file.py OPTIONS

See the script for more details.



2. downsampling a mallet file


3. feature selection on a mallet file



4. running th eclassifier
