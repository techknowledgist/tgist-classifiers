"""count_terms.py

Counts the number of terms in a classification.

Usage:

    $ python count_tokens.py CLASSIFICATION_PATH

The CLASSIFICATION_PATH holds a classification as created by run_tclassify.py in
ontology/classifier. The default for CLASSIFICATION_PATH is the example
classification.

Prints the terms to the standard output.

This works off of the classify.MaxEnt.out.s3.scores.sum file in the
classification directory.

It takes about X minutes to run this scripts on a 6MB, 500K token corpus.

"""


import os, sys, time
sys.path.append(os.path.abspath('../../..'))
from ontology.utils.file import open_input_file


CLASSIFICATION = '../data/classifications/sample-us'


def count_terms(classification):
    t1 = time.time()
    term_count = 0
    inst_count = 0
    tech_term_count = 0
    tech_inst_count = 0
    done = 0
    for line in open(os.path.join(classification, 'classify.MaxEnt.out.s3.scores.sum')):
        #if done > 50: break
        fields = line.rstrip().split("\t")
        techscore = float(fields[1])
        instances = int(fields[2])
        #print done, techscore, instances
        term_count += 1
        inst_count += instances
        if techscore > 0.5:
            tech_term_count += 1
            tech_inst_count += instances
        done += 1
    print os.path.basename(classification),
    print term_count, inst_count, tech_term_count, tech_inst_count,
    print "(%d seconds)" % (time.time() - t1)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        CLASSIFICATION = sys.argv[1]
    count_terms(CLASSIFICATION)
