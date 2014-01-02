
"""

Script to analyze the contents of a mallet file. Counts the types and instances,
the features and lists the terms with the count of their instances.

Usage:

    $ python analyze_mallet_file.py MALLET_FILE

    Writes output to MALLET_FILE.stats.txt.

"""


import sys, codecs


def parse_mallet_line(line):
    fields = line.split()
    #print line
    #print fields
    id = fields[0]
    label = fields[1]
    features = fields[2:]
    (year, fname, term) = id.split('|', 2)
    #print label, term
    return label, term, features



mallet_file = sys.argv[1]
info_file = mallet_file + '.stats.txt'

pos_terms = {}
neg_terms = {}
features = {}

with codecs.open(mallet_file, encoding='utf-8') as fh:
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
            feat = featval.split('=')[0]
            #if feat == '234_shore': print line
            features[feat] = features.get(feat,0) + 1


total_positive_terms = len(pos_terms)
total_negative_terms = len(neg_terms)

pos_instances = pos_terms.values()
neg_instances = neg_terms.values()

total_pos_instances = sum(pos_instances)
total_neg_instances = sum(neg_instances)

with codecs.open(info_file, 'w', encoding='utf-8') as fh:

    fh.write("positive types:               %8d\n"   % total_positive_terms)
    fh.write("negative types:               %8d\n\n" % total_negative_terms)
    fh.write("positive instances:           %8d\n"   % total_pos_instances)
    fh.write("negative instances:           %8d\n\n" % total_neg_instances)
    fh.write("instances per positive term:  %8d\n"   % (total_pos_instances / len(pos_terms)))
    fh.write("instances per negative term:  %8d\n\n" % (total_neg_instances / len(neg_terms)))

    for i in (100, 200, 1000, 2000):
        fh.write("positive instances with count capped at %4d: %8d\n" % (i, sum([min(n,i) for n in pos_instances])))
        fh.write("negative instances with count capped at %4d: %8d\n\n" % (i, sum([min(n,i) for n in neg_instances])))

    fh.write("\nFeatures:\n\n")
    for feat in sorted(features.keys()):
        fh.write("  %8d  %s\n" % (features[feat], feat))

    fh.write("\n\nTerms with positive instances:\n\n")
    for term in sorted(pos_terms.keys()):
        fh.write("%8d  %s\n" % (pos_terms[term], term))
    fh.write("\nTerms with negitive instances:\n\n")
    for term in sorted(neg_terms.keys()):
        fh.write("%8d  %s\n" % (neg_terms[term], term))
