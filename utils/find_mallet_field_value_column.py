# find_mallet_field_value_column.py
# assuming a line in mallet output of the form:
# 1995|US5402540A|water_resistant_bond    y       0.05596473591425579     n       0.9425655360420153      d       7.87658559655227E-4
#
# determine which column contains the value for the class and return that column number (1-based)
# python2.6 find_mallet_field_value_column.py /home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents/en/test/utest.2.MaxEnt.out.all_scores y 
# python2.6 find_mallet_field_value_column.py data/patents/en/test/utest.2.MaxEnt.out.all_scores y 

# MV: put code into a function so I can call this directly from Python, but behaviour when
# called from a linux shell script has stayed the same.

import sys

def find_column(mallet_all_scores_file, field_value):
    s_lines = open(mallet_all_scores_file)
    first_line = s_lines.readline()
    l_field = first_line.split("\t")
    field_column = 0
    try:
        field_column = l_field.index(field_value)
        # actual desired column is the next field.  We want it 1-based for unix cut
        # function rather than 0-based so we need to add 2
        field_column = field_column + 2
        # print "field_column: %i" % (field_column) 
    except Exception: 
        print "[Error in find_mallet_field_value_column.py] " + \
              "The desired field value [%s] was not found in the mallet out file: %s" \
              % (field_value, mallet_all_scores_file)
    s_lines.close()
    return str(field_column)


if __name__ == '__main__':
    mallet_all_scores_file = sys.argv[1]
    field_value = sys.argv[2]
    print find_column(mallet_all_scores_file, field_value)
    # use exit code to return integer to bash
    # sys.exit(field_column)
