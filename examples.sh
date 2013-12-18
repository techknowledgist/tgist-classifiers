# some examples for creating a mallet file and downsampling

# create a mallet file from 10 files of the 500 sample patents
#python create_mallet_file.py --corpus ../creation/data/patents/201312-en-500 --model-dir data/models/technologies-201312-en-500-010 --annotation-file ../annotation/en/technology/phr_occ.lab --filelist files-010.txt --verbose

# create a mallet file from all files of the 500 sample patents
#python create_mallet_file.py --corpus ../creation/data/patents/201312-en-500 --model-dir data/models/technologies-201312-en-500-all --annotation-file ../annotation/en/technology/phr_occ.lab --filelist files.txt --verbose

# quick downsample on 10-file mallet file with theshold of 5
#python downsample.py --source-mallet-file data/models/technologies-201312-en-500-010/train.mallet --threshold 5



python run_tclassify.py --classify --corpus ../creation/data/patents/201312-en-500/ --filelist ../creation/data/patents/201312-en-500/config/files-010.txt --model data/models/technologies-201312-en-500-010/train.ds0005.standard.model --batch data/classifications/test2 --verbose

# old example of creating a model, code soon to be obsolete
# python run_tclassify.py --train  --corpus ../creation/data/patents/201312-en-500  --pipeline pipeline-default.txt  --filelist files-010.txt  --annotation-file ../annotation/en/technology/phr_occ.lab  --annotation-count 2000  --model data/models/test --features extint  --xval 0  --verbose

# old example of classifying using a model, code soon to be obsolete
#python run_tclassify.py --classify --corpus ../creation/data/patents/201312-en-500 --pipeline pipeline-default.txt --filelist files-010.txt --model data/models/test --batch data/classifications/testtttt --verbose
