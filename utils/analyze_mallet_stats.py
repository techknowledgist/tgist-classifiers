"""

Print some analytics of the statistics file created when building the mallet
file from the phr_feats files. Takes 

$ python analyze_mallet_stats.py /home/j/corpuswork/fuse/FUSEData/corpora/ln-all-600k/models/1997/info/train.info.stats.txt


"""


import sys, codecs

stats_file = sys.argv[1]
fh = codecs.open(stats_file, encoding='utf-8')

pos_counts = []
neg_counts = []

print_header = True
reading_pos = False
reading_neg = False


for line in fh:

    if line.startswith('Terms with positive training instances'):
        reading_pos = True
        reading_neg = False
        print_header = False

    elif line.startswith('Terms with negative training instances'):
        reading_pos = False
        reading_neg = True
        print_header = False

    elif print_header:
        print line,

    elif reading_pos:
        try:
            count = int(line.strip().split()[0])
            pos_counts.append(count)
        except IndexError:
            pass

    elif reading_neg:
        try:
            count = int(line.strip().split()[0])
            neg_counts.append(count)
        except IndexError:
            pass


print "Average positive instances per term:", sum(pos_counts)/len(pos_counts)
print "Average negative instances per term:", sum(neg_counts)/len(neg_counts)

print
for i in (50, 100, 200, 1000):
    print "Total positive instances if count capped at %4d: %8d" \
          % (i, sum([(n if n <= i else i) for n in pos_counts]))

print
for i in (50, 100, 200, 1000):
    print "Total negative instances if count capped at %4d: %8d" \
          % (i, sum([(n if n <= i else i) for n in neg_counts]))
