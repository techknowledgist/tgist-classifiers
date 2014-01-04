"""

Merges all summed classifier outputs in sorted files into one big file. Assumes
the sorted files are files for each year.

Usage:
    
$ python merge.py 1997 1998 1999 2000 2001 ...

The result is written to all_terms.count.txt which has two fields: a total count
and the term.

Merging all files from BASE_DIR (which have a total of 51,752,285 terms) results
in an all_terms.txt file with 31,453,657 terms (on 10/24/2013).

"""


import os, sys, codecs

# settings for ln-cs-500k
BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-cs-500k/classifications'
CLASSIFICATION_EXP = '%s-technologies-standard-1000'
TERM_FILE = 'classify.MaxEnt.out.s5.scores.sum.az'
BAD_TERM_FILE = 'classify.MaxEnt.out.s6.terms.bad'

# settings for ln-all-600k
BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/classifications'
CLASSIFICATION_EXP = 'technologies-ds1000-all-%s'
TERM_FILE = 'classify.MaxEnt.out.s4.scores.sum.az'
BAD_TERM_FILE = 'classify.MaxEnt.out.s6.terms.x.bad'



def simple_merge(years):

    terms = {}

    for year in years:
        
        print "\n%s" % year

        term_file = os.path.join(BASE_DIR, CLASSIFICATION_EXP % year, TERM_FILE)
        bad_term_file = os.path.join(BASE_DIR, CLASSIFICATION_EXP % year, BAD_TERM_FILE)
        print '  ', term_file
        print '  ', bad_term_file
        fh_terms = codecs.open(term_file)
        fh_bad_terms = codecs.open(bad_term_file)

        bad_terms = {}
        for line in fh_bad_terms:
            bad_terms[line.rstrip('\n\r')] = True
        print "   read %d bad terms" % len(bad_terms)
        
        count = 0
        for line in fh_terms:
            count += 1
            if count % 500000 == 0:
                print '  ', count
            fields = line.split("\t")
            term = fields[0]
            term_count = int(fields[2])
            if not bad_terms.has_key(term):
                terms[term] = terms.get(term, 0) + term_count

    print "\nWriting all %d terms" % len(terms)
    fh_out = codecs.open('all_terms.count.txt', 'w')
    for term in terms:
        fh_out.write("%d\t%s\n" % (terms[term], term))


if __name__ == '__main__':

    years = sys.argv[1:]
    simple_merge(years)




"""

Here is some code that attempted to merge in the smart way, by having pointers
at alphabetically sorted files. This will never get you into memory
problems. The thing was that the unix sort we had used did not seem to agree
with the python sort using '<' so we got weird results.


BASE_DIR = '/home/j/corpuswork/fuse/FUSEData/corpora/cs-500k/classifications'
CLASSIFICATION_EXP = '%s-technologies-standard-1000'
FILENAME = 'classify.MaxEnt.out.s5.scores.sum.az'

        
def merge(years):
    cm = ClassificationMerger()
    for year in years:
        sc = SortedClassification(year)
        #sc.pp()
        cm.add_classification(sc)


    x = {}
    for sc in cm.classifications:
        print sc
        while sc.next_term is not None:
            x.setdefault(sc.next_term,[]).append(999)
        
    return
        
    cm.merge()


class ClassificationMerger(object):

    def __init__(self):
        self.classifications = []
        
    def add_classification(self, sorted_classification):
        self.classifications.append(sorted_classification)

    def merge(self):
        next, sc = self.get_next()
        while next is not None:
            print next
            self.collect(next)
            self.pop(next)
            next, sc = self.get_next()
            
    def get_next(self):
        next = None
        next_sc = None
        for sc in self.classifications:
            t = sc.next_term
            if next is None or next > t:
                next = t
                next_sc = sc
        return next, next_sc
    
    def collect(self, term):
        for sc in self.classifications:
            if sc.next_term == term:
                print '  ', sc

    def pop(self, term):
        for sc in self.classifications:
            if sc.next_term == term:
                sc.pop()


class SortedClassification(object):

    def __init__(self, year):
        self.year = year
        self.filename = os.path.join(BASE_DIR, CLASSIFICATION_EXP % year, FILENAME)
        self.fh = codecs.open(self.filename)
        term, score = self.read_next()
        self.next_term = term
        self.next_score = score

    def __str__(self):
        return "<SortedClassification on %s>" % self.year
    
    def pp(self):
        print "<SortedClassification>"
        print "    year = %s" % self.year
        print "    file = %s" % self.filename
        print "    next = %s\n" % self.next_term
    
    def pop(self):
        term, score = self.read_next()
        self.next_term = term
        self.next_score = score
        
    def read_next(self):
        line = self.fh.readline()
        if line:
            fields = line.split("\t")
            term, score = fields[:2]
            return term, score
        else:
            return None, None
        

"""
