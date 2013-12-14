# some examples for creating a mallet file and downsampling

# create a mallet file from 10 files of the 500 sample patents
#python create_mallet_file.py --corpus ../creation/data/patents/201312-en-500 --model-dir data/models/technologies-201312-en-500-010 --annotation-file ../annotation/en/technology/phr_occ.lab --filelist files-010.txt --verbose

# create a mallet file from all files of the 500 sample patents
#python create_mallet_file.py --corpus ../creation/data/patents/201312-en-500 --model-dir data/models/technologies-201312-en-500-all --annotation-file ../annotation/en/technology/phr_occ.lab --filelist files.txt --verbose

# quick downsample on 10-file mallet file with theshold of 5
python downsample.py --source-mallet-file data/models/technologies-201312-en-500-010/train.mallet --threshold 5
