"""

This script takes a config/files.txt file from a corpus and creates a file
config/files-2007.txt which contains only those patents from files.txt that have
a publication date of 2007 or earlier.

This is to meet the MITRE phase 2 requirement that we do not look at data after
2007 and they use the publication date to determine the year.

Recall that the format of files.txt is as follows:

    APPLICATION_YEAR  FULL_PATH  SHORT_PATH

where SHORT_PATH is made up of a publication year and the name of the patent.

Usage:

    $ python select_years.py CORPUS_PATH

"""


import os, sys, codecs

corpus = sys.argv[1]

mallet_file_in = os.path.join(corpus, 'config', 'files.txt')
mallet_file_out = os.path.join(corpus, 'config', 'files-2007.txt')

fh_in = codecs.open(mallet_file_in, encoding='utf-8')
fh_out = codecs.open(mallet_file_out, 'w', encoding='utf-8')

count_in = 0
count_out = 0
for line in fh_in:
    count_in += 1
    pubyear = int(line.split()[2].split(os.sep, 1)[0])
    #print pubyear, line,
    if pubyear <= 2007:
        count_out += 1
        fh_out.write(line)
    #if count_in > 20: break

print "Patents checked  %5d" % count_in
print "Patents kept     %5d" % count_out
