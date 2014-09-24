"""

Merges summed classifier outputs.

Usage:
    
$ python merge_classifier_results.py OUTPUT_DIRECTORY CLASSIFIER_RESULT+

The result are written to several files in OUTPUT_DIRECTORYand each file has two
fields: a total count and the term. One file has all terms and their
frequencies, whereas the others only have the terms that occur N times, where N
is given in the file name. In addition, an info file is written with the version
of the code used and all the names of the files that were merged.

CLASSIFIER_RESULT is a summed classifier result, typically a file with the base
name classify.MaxEnt.out.s3.scores.sum, but it can also be an expression with
unix wild cards.

Example:

$ python merge_classifier_results.py data/merged_terms /home/j/corpuswork/fuse/FUSEData/corpora/ln-us-all-600k/classifications/technologies-ds1000-all-*/classify.MaxEnt.out.s3.*

This file has code taken from classifier/utils/merge.py as well as from
classifier/utils/split_terms_on_frequency.py. Unlike the first it does not do
any term filtering. 

"""

import os, sys, glob, codecs, gzip
sys.path.append(os.path.abspath('../..'))
from ontology.utils.file import ensure_path
from ontology.utils.git import get_git_commit


def open_input_file(fname):
    if fname.endswith('.gz'):
        gzipfile = gzip.open(fname, 'rb')
        reader = codecs.getreader('utf-8')
        return reader(gzipfile)
    else:
        return codecs.open(fname, encoding='utf-8')                                

def merge_result_files(target_dir, result_files):
    terms = collect_terms(result_files)
    print "\nWriting all %d terms" % len(terms)
    outfile = target_dir + '/merged_term_frequencies.all.txt'
    fh_out = codecs.open(outfile, 'w', encoding='utf-8')
    thresholds = [5, 10, 25, 50, 100]
    fhs = {}
    for i in thresholds:
        outfile = "%s/merged_term_frequencies.%04d.txt" % (target_dir, i)
        fhs[i] = codecs.open(outfile, 'w', encoding='utf-8')
    for term in terms:
        frequency = terms[term]
        fh_out.write("%d\t%s\n" % (frequency, term))
        for i in thresholds:
            if frequency >= i:
                fhs[i].write("%d\t%s\n" % (frequency, term))

def collect_terms(result_files):
    terms = {}
    for fname in result_files:
        print fname
        fh_terms = open_input_file(fname)
        count = 0
        for line in fh_terms:
            count += 1
            if count > 100000: break
            if count % 500000 == 0: print '  ', count
            fields = line.split("\t")
            term = fields[0]
            term_count = int(fields[2])
            terms[term] = terms.get(term, 0) + term_count
    return terms


if __name__ == '__main__':
    
    target_dir = sys.argv[1]
    result_files = []
    for exp in sys.argv[2:]:
        files = glob.glob(exp)
        result_files.extend(files)

    ensure_path(target_dir)
    infofile = target_dir + '/merged_term_frequencies.info.txt'
    fh_info = codecs.open(infofile, 'w', encoding='utf-8')
    fh_info.write("git commit = %s\n\n" % get_git_commit())
    for fname in result_files:
        fh_info.write(fname + u"\n")

    merge_result_files(target_dir, result_files)
    
