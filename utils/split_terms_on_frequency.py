"""

Take all_terms.count.txt and create files with only the terms with a minimum
number of occurrences.

"""

import codecs

terms_file = 'all_terms.count.txt'

thresholds = [5, 10, 25, 50, 100]

fh = codecs.open(terms_file, encoding='utf-8')
fhs = {}
for i in thresholds:
    fhs[i] = codecs.open("all_terms.%04d.txt" % i, 'w', encoding='utf-8')

lines = 0
for line in fh:
    lines += 1
    #if lines > 500000: break
    if lines % 100000 == 0: print lines
    #print line,
    count = int(line.split()[0])
    for i in thresholds:
        if count >= i:
            fhs[i].write(line)
