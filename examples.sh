# Some examples for creating a mallet file and downsampling and running the classifier

# create a mallet file from 10 files of the 500 sample patents
echo

echo python create_mallet_file.py --corpus ../creation/data/patents/201312-en-500 --model-dir data/models/technologies-201312-en-500-010 --annotation-file ../annotation/en/technology/phr_occ.lab --filelist files-010.txt --verbose

# create a mallet file from all files of the 500 sample patents
echo
echo python create_mallet_file.py --corpus ../creation/data/patents/201312-en-500 --model-dir data/models/technologies-201312-en-500-all --annotation-file ../annotation/en/technology/phr_occ.lab --filelist files.txt --verbose

# quick downsample on 10-file mallet file with theshold of 5
echo
echo python downsample.py --source-mallet-file data/models/technologies-201312-en-500-010/train.mallet --threshold 5


# run classifier on 10 random files
echo
echo python run_tclassify.py --classify --corpus ../creation/data/patents/201312-en-500/ --filelist ../creation/data/patents/201312-en-500/config/files-010.txt --model data/models/technologies-201312-en-500-010/train.ds0005.standard.model --batch data/classifications/test2 --verbose

# run classifier on the 10 test files and add evaluation
echo
echo python run_tclassify.py --classify --corpus ../creation/data/patents/201312-en-500/ --filelist ../creation/data/patents/201312-en-500/config/files-testing.txt --model data/models/technologies-201312-en-500-010/train.ds0005.standard.model --batch data/classifications/test-eval --verbose --gold-standard ../annotation/en/technology/phr_occ.eval.lab

# old example of creating a model, code soon to be obsolete
echo
echo python run_tclassify.py --train  --corpus ../creation/data/patents/201312-en-500  --pipeline pipeline-default.txt  --filelist files-010.txt  --annotation-file ../annotation/en/technology/phr_occ.lab  --annotation-count 2000  --model data/models/test --features extint  --xval 0  --verbose

# old example of classifying using a model, code soon to be obsolete
echo
echo python run_tclassify.py --classify --corpus ../creation/data/patents/201312-en-500 --pipeline pipeline-default.txt --filelist files-010.txt --model data/models/test --batch data/classifications/testtttt --verbose


# run the technology classifier on a set of corpora

CORPORA=/home/j/corpuswork/fuse/FUSEData/corpora/wos-cs-520k/subcorpora
CORPORA=/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-cs-500k/subcorpora
CORPORA=/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-all-600k/subcorpora/

CLASSIFICATIONS=/home/j/corpuswork/fuse/FUSEData/corpora/wos-cs-520k/classifications
CLASSIFICATIONS=/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-cs-500k/classifications/technologies
CLASSIFICATIONS=/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-all-600k/classifications

MODELS=/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-all-600k/models
MODEL=$MODELS/technologies/pubyears-all/merged-97-07/train.ds1000.model
MODEL=$MODELS/technologies/random-50k/gt-10000/train.model

echo
for year in 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007;
do
    echo $ python run_tclassify.py --classify --corpus $CORPORA/$year --model $MODEL --output $CLASSIFICATIONS/$year --verbose
done
