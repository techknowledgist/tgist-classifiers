# reformat output of unix uniq -c | sort -nr
# to <count>\t<value>

import os
import sys

def reformat_uc():
    for line in sys.stdin:
        line = line.strip("\n")
        line = line.lstrip(" ")
        count = line[0:line.find(" ")]
        value = line[line.find(" ")+1:]
        #print "count: %s, value %s" % (count, value)
        newline = count + "\t" + value
        print newline

if __name__ == "__main__":
    reformat_uc()

