"""

Take the summed and sorted score files and create two files with the terms only,
the first file has terms that are considered good and the second has terms that
were filered out. Filtering proceeds by checking the size of the term and
counting non-letters.

Usage:
    
    $ python filter.py 1997 1998 1999 2000 2001 ...

The directory and specific file names can be edited below.

TODO: add this to run_iclassifier

"""

# settings for ls-us-cs-500k
BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-cs-500k/classifications'
CLASSIFICATION_EXP = '%s-technologies-standard-1000'

# settings for ls-us-all-600k
BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-all-600k/classifications'
CLASSIFICATION_EXP = 'technologies-ds1000-all-%s'

# settings for ls-us-all-600k, time series v2
BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-us-all-600k/classifications/phase2-eval'
CLASSIFICATION_EXP = 'technologies-ds1000-all-%s'

# settings for ls-cn-all-600k, time series v2
BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-cn-all-600k/classifications/phase2-eval'
CLASSIFICATION_EXP = 'technologies-ds1000-all-%s'

# TODO: maybe use .sum instead of .sum.az
FILENAME1 = 'classify.MaxEnt.out.s4.scores.sum.az'
FILENAME2 = 'classify.MaxEnt.out.s6.terms.x.good'
FILENAME3 = 'classify.MaxEnt.out.s6.terms.x.bad'


FILENAME1 = 'classify.MaxEnt.out.s3.scores.sum'

# TODO: THIS ONLY MAKES SENSE FOR ENGLISH!!!!!


import os, sys, codecs


def filter_terms(infile, outfile1, outfile2, idx=0):
    """idx is the field in infile that has the term"""
    print '  ', outfile1
    print '  ', outfile2
    print
    fh_in = codecs.open(infile)
    fh_out1 = codecs.open(outfile1, 'w')
    fh_out2 = codecs.open(outfile2, 'w')
    count = 0
    for line in fh_in:
        count += 1
        if count > 100000: break
        if count % 100000 == 0: print '  ', count
        #print line.rstrip("\n\r").split("\t")
        term = line.rstrip("\n\r").split("\t")[idx]
        term = line.split("\t")[idx]
        term_size = len(term)
        letters = len([c for c in term if c.isalpha()])
        rest = term_size - letters
        #if letters > rest: print term_size, letters, rest, term
        #if rest == 0: print term_size, letters, rest, term
        if term_size > 75 or letters < rest:
            #print term_size, letters, rest, term
            fh_out2.write("%s\n" % term)
        else:
            fh_out1.write("%s\n" % term)

def filter_years():
    """FIlters the terms for classifications for a set of years"""
    years = sys.argv[1:]
    for year in years:
        infile = os.path.join(BASE_DIR, CLASSIFICATION_EXP % year, FILENAME1)
        outfile1 = os.path.join(BASE_DIR, CLASSIFICATION_EXP % year, FILENAME2)
        outfile2 = os.path.join(BASE_DIR, CLASSIFICATION_EXP % year, FILENAME3)
        print year
        filter_terms(infile, outfile1, outfile2)
        print

def filter_all_terms():
    """This runs the filter on all terms in the merge term file."""
    # NOTE: this is actually useless because the merged file was already filtered
    basedir = "/home/j/corpuswork/fuse/FUSEData/corpora/cs-500k/classifications"
    infile = os.path.join(BASE_DIR, 'all_terms.txt')
    outfile1 = os.path.join(BASE_DIR, 'terms.good.txt')
    outfile2 = os.path.join(BASE_DIR, 'terms.bad.txt')
    filter_terms(infile, outfile1, outfile2, idx=1)


if __name__ == '__main__':

    filter_years()
    #filter_all_terms()
