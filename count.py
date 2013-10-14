"""

Some methods to round up frequency counts and to filter the term list on
frequencies.

"""

import os, sys, codecs


def count_frequencies():
    """Takes as the first argukment a file with two tab-separated columns, count
    and some term, and then creates a frequency table of how often each count
    occurred. This information is written to the second argument. Doing this
    makes most sense on the merged file created with merge.py, which for the
    2013 computer science corpus lives in 
    /home/j/corpuswork/fuse/FUSEData/corpora/cs-500k/classifications """
    fh_in = codecs.open(sys.argv[1])
    fh_out = codecs.open(sys.argv[2], 'w')
    counts = {}
    c = 0
    for line in fh_in:
        c += 1
        if c % 1000000 == 0: print c
        # if c > 1000000: break
        count = int(line.split()[0])
        counts[count] = counts.get(count, 0) + 1
    for count in sorted(counts):
        fh_out.write("%d\t%d\n" % (count, counts[count]))


def filter_frequencies():
    """Takes as the first argument the file with all terms and then creates
    files in the same directry with all terms with frequencies of n or higher,
    where n is now set to one of 5, 10, 25, 50 and 100. Creates several output
    files."""
    fname = sys.argv[1]
    fh_in = codecs.open(fname)
    pathname, ext = os.path.splitext(fname)
    freqs = [5, 10, 25, 50, 100]
    fh_out = {}
    for freq in freqs:
        outfile = "%s.%04d%s" % (pathname, freq, ext)
        fh_out[freq] = codecs.open(outfile, 'w')
    c = 0
    for line in fh_in:
        c += 1
        if c % 1000000 == 0: print c
        #if c > 1000000: break
        freq = int(line.split("\t")[0])
        for f in freqs:
            if freq >= f:
                fh_out[f].write(line)
    

if __name__ == '__main__':

    #count_frequencies()
    filter_frequencies()
    
