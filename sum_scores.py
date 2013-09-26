# sum_scores.py

# PGA 10/19/12

# process a scores file to generate average scores

# scores file contains:
# <year>|<doc_id>|<phrase>\tscore
#. eg.
# 2011|US20110157681A1|pixel_electrode    0.999884251916764

# there can be more than one score per phrase in the input scores file if it occurs in multiple docs
# For each phrase, we compute max, min, count, average score

# cd /home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/data/patents/en/test
# head -2000 utest.7.MaxEnt.out.scores > utest.7.MaxEnt.out.scores.2000
# python /home/j/corpuswork/fuse/code/patent-classifier/ontology/creation/sum_scores.py utest.7.MaxEnt.out.scores.2000 utest.7.MaxEnt.out.scores.2000.sum
# cat utest.7.MaxEnt.out.scores.2000.sum | sort -k2,2 -nr -t"   " | more

# NOTE: Phrases input to this function may contain escaped forward slash ("\/").  By default, this will be converted to "/" unless
# remove_backslash_p parameter is set to False in sum_scores.

# error reported on 
# 0.00002(107723)=1.0

import sys
#from collections import defaultdict

def sum_scores(doc_scores_file, sum_scores_file, remove_backslash_p = True):

    # MV: Changed this so that we can use a generic python in patent_tech_scores.py
    # d_phr2score = defaultdict(list)
    d_phr2score = {}

    s_doc_scores_file = open(doc_scores_file)
    s_sum_scores_file = open(sum_scores_file, "w")
    unexpected_input = 0
    for line in s_doc_scores_file:
        line = line.strip("\n")
        fields = line.split("\t")
        try:
            # for chinese, we are getting lines like "0.00002(107723)=1.0"
            doc_score = fields[1]
        except IndexError:
            unexpected_input += 1
        phrase = fields[0]
        # pull out the phrase and replace the underscores with blanks
        phrase = phrase[phrase.rfind("|")+1:].replace("_", " ")
        #print "phrase: %s" % phrase
        #d_phr2score[phrase].append(float(doc_score))
        d_phr2score.setdefault(phrase,[]).append(float(doc_score))

    if unexpected_input > 0:
        print "WARNING: %d lines in the input were unexpected" % unexpected_input
        
    for key in d_phr2score.keys():
        sum = 0.0
        count = 0
        max = 0.0
        min = 1000000000000.0
        for score in d_phr2score.get(key):
            sum = sum + score
            count += 1
            if score > max:
                max = score
            if score < min:
                min = score
        average = sum / count
        #print "%s\t%f\t%i\t%f\t%f" % (key, average, count, min, max)

        if remove_backslash_p:
            key = key.replace("\/", "/")
        s_sum_scores_file.write("%s\t%f\t%i\t%f\t%f\n" % (key, average, count, min, max))
    s_doc_scores_file.close()
    s_sum_scores_file.close()

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sum_scores(input_file, output_file)
    print "[sum_scores.py]Unsorted score summary in %s" % output_file

