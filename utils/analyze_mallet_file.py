
"""

Script to analyze the contents of a mallet file:
    
    - counts the types and instances,
    
    - counts the features and lists for each feature how often it occurred and
      how many different values there were,

    - lists the positive and negative terms with their counts

    
Usage:

    $ python analyze_mallet_file.py MALLET_FILE


Writes output to MALLET_FILE.stats.txt. This can run on both the mallet file
created for a model as well as a mallet file created for a classification, but
the output is less informative for the latter.

"""


import os, sys, codecs
sys.path.append(os.path.abspath('../../..'))
from ontology.utils.file import open_input_file


def parse_mallet_line(line):
    # Use split(' ') instead of split(). The Stanford tagger seems to insert
    # some characters in some cases, they look like underscores but they aren't,
    # and for some reason split() ends up splitting on those characters.
    fields = line.split(' ')
    id = fields[0]
    label = fields[1]
    features = fields[2:]
    #print line,
    #print len(features), features
    (year, fname, term) = id.split('|', 2)
    #print label, term
    return label, term, features


mallet_file = sys.argv[1]
info_file = mallet_file + '.stats.txt'

pos_terms = {}
neg_terms = {}
features = {}
featvals = {}


with open_input_file(mallet_file) as fh:
    count = 0
    for line in fh:
        count += 1
        #if count > 10000: break
        if count % 100000 == 0: print count
        label, term, feats = parse_mallet_line(line)
        if label == 'y':
            pos_terms[term] = pos_terms.get(term, 0) + 1
        elif label == 'n':
            neg_terms[term] = neg_terms.get(term, 0) + 1
        for featval in feats:
            feat, val = featval.split('=', 1)
            #if feat == '234_shore': print line
            features[feat] = features.get(feat,0) + 1
            if not featvals.has_key(feat):
                featvals[feat] = {}
            featvals[feat][val] = featvals[feat].get(val,0) + 1


total_positive_terms = len(pos_terms)
total_negative_terms = len(neg_terms)

pos_instances = pos_terms.values()
neg_instances = neg_terms.values()

total_pos_instances = sum(pos_instances)
total_neg_instances = sum(neg_instances)


with codecs.open(info_file, 'w', encoding='utf-8') as fh:

    fh.write("# Some statistics on %s\n#\n" % mallet_file)
    fh.write("# Created with ontology/classifier/utils/analyze_mallet_file.py\n\n")
    fh.write("positive types:               %8d\n"   % total_positive_terms)
    fh.write("negative types:               %8d\n\n" % total_negative_terms)
    fh.write("positive instances:           %8d\n"   % total_pos_instances)
    fh.write("negative instances:           %8d\n\n" % total_neg_instances)
    pos_ratio = 0 if len(pos_terms) == 0 else (total_pos_instances / len(pos_terms))
    neg_ratio = 0 if len(neg_terms) == 0 else (total_neg_instances / len(neg_terms))
    fh.write("instances per positive term:  %8d\n"   % pos_ratio)
    fh.write("instances per negative term:  %8d\n\n" % neg_ratio)

    for i in (100, 200, 1000, 2000):
        fh.write("positive instances with count capped at %4d: %8d\n"
                 % (i, sum([min(n,i) for n in pos_instances])))
        fh.write("negative instances with count capped at %4d: %8d\n\n"
                 % (i, sum([min(n,i) for n in neg_instances])))

    fh.write("\nFeatures:\n\n")
    fh.write("   feature           count     values\n\n")
    for feat in sorted(features.keys()):
        fh.write("   %-12s %10d %10d\n" % (feat, features[feat], len(featvals[feat])))
            
    fh.write("\n\nTerms with positive instances:\n\n")
    for term in sorted(pos_terms.keys()):
        fh.write("%8d  %s\n" % (pos_terms[term], term))
    fh.write("\nTerms with negitive instances:\n\n")
    for term in sorted(neg_terms.keys()):
        fh.write("%8d  %s\n" % (neg_terms[term], term))
