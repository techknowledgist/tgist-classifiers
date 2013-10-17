import sys

# usage:
# cat iclassify.MaxEnt.out | egrep -v '^name' | egrep '\|.*\|' | python /home/j/anick/patent-classifier/ontology/creation/invention_top_scores.py > iclassify.MaxEnt.label

# takes the mallet output and isolates the top scoring label, ignores features
# assumes lines of the form:
# 2011|US7869095B2.xml_16|pixel_value     i       0.015153757395538953    m       0.0184681787010246      c       0.9085227024373407       o       0.011836733588594574    r       0.046018627877501106

for line in sys.stdin:
    line = line.strip("\n")
    l_data = line.split("\t")
    key = l_data[0]
    l_rest = l_data[1:]
    max_score = 0.0
    max_label = "n"
    while l_rest != []:
        label = l_rest[0]
        score = float(l_rest[1])
        if score > max_score:
            max_score = score
            max_label = label
        l_rest = l_rest[2:]
    print "%s\t%s\t%s" % (key, max_label, str(max_score))

