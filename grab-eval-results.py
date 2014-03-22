
"""

Quick and dirty script to grab the results of an evaluation and pull out fscore,
precision and recall.

Edit the three globals near the top for specific settings:

    DIR - a directory with classifications,
    EXP - an expression to match a sub directory
    FNAME - the file you take data from in the subdirectory

Output is written in a latex tabular format.

"""


import glob, os

DIR = 'data/classifications/eval-lrec2014/en-all-eval-10/features'
EXP = 'gt*'
FNAME = 'eval-old-all-ytf.txt'

for dname in glob.glob(DIR + '/' + EXP):
    #print dname
    for fname in glob.glob(dname + '/' + FNAME):
        #print fname
        basename = os.path.basename(dname)
        feature = basename.split('-')[-1]
        for line in open(fname):
            if line.startswith('threshold: 0.5'):
                p, r, f = line.split()[5:8]
                p = p.split('=')[1]
                r = r.split('=')[1]
                f = f.split('=')[1]
                print "gt-10000 - %s \t& %s \t& %s \t& %s\t\\\\" % (feature, f, p, r)
