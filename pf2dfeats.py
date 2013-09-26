# convert a file of phrase occurrence features (phr_feats)
# to a file with one (summed) set of features per phrase (doc_feats)

import os
import codecs


def make_doc_feats(phr_feats, doc_feats, doc_id, year):
    """Take a file with phrase features and create a file with document
    features."""
    s_phr_feats = codecs.open(phr_feats)
    s_doc_feats = codecs.open(doc_feats, "w")
    d_p2f = generate_doc_feats(s_phr_feats, doc_id, year)
    for key in sorted(d_p2f.keys()):
        features = d_p2f[key]
        s_doc_feats.write("\t".join(features) + "\n")
    s_phr_feats.close()
    s_doc_feats.close()

def generate_doc_feats(s_phr_feats, doc_id, year):
    """Given a file handle to a file with phase features, generate and return a
    mapping from phrases to the document features for the phrase. The document
    features include the term as the first element and an identifier with year,
    document and term as the second element."""
    d_doc_feats = {}
    for line in s_phr_feats:
        l_feat = line.strip("\n").split("\t")
        # key is the chunk/phrase itself
        key, feats = l_feat[2], l_feat[3:]
        d_doc_feats.setdefault(key, set()).update(set(feats))
    for key, value in d_doc_feats.items():
        symbol_key = key.replace(" ", "_")
        uid = year + "|" + doc_id + "|" + symbol_key
        features = [key, uid]
        features.extend(sorted(list(value)))
        d_doc_feats[key] = features
    return d_doc_feats

def pf2dfeats_dir(phr_feats_year_dir, doc_feats_year_dir, year):
    for file in os.listdir(phr_feats_year_dir):
        input = os.path.join(phr_feats_year_dir, file)
        output = os.path.join(doc_feats_year_dir, file)
        (doc_id, extension) = file.split(".")
        make_doc_feats(input, output, doc_id, year)

# e.g. tag2chunk.patent_tag2chunk_dir("/home/j/anick/fuse/data/patents", "de")
def patent_pf2dfeats_dir(patent_path, language):
    lang_path = patent_path + "/" + language
    phr_feats_path = lang_path + "/phr_feats"
    doc_feats_path = lang_path + "/doc_feats"
    for year in os.listdir(phr_feats_path):
        phr_feats_year_dir = phr_feats_path + "/" + year
        doc_feats_year_dir = doc_feats_path + "/" + year
        print "[patent_pf2dfeats_dir]calling pf2dfeats for dir: %s" % phr_feats_year_dir
        pf2dfeats_dir(phr_feats_year_dir, doc_feats_year_dir, year)
    print "[patent_pf2dfeats_dir]finished creating doc_feats in: %s" % (doc_feats_path)

def pipeline_pf2dfeats_dir(root, language):
    phr_feats_path = root + "/phr_feats"
    doc_feats_path = root + "/doc_feats"
    # The only way to determine the year for a file is to look in file_list.txt
    file_list_file = os.path.join(root, "file_list.txt")
    s_list = open(file_list_file)
    for line in s_list:
        (identifier, year, path) = line.split(" ")
        file_name = identifier + ".xml"
        phr_feats_file = os.path.join(phr_feats_path, file_name)
        doc_feats_file = os.path.join(doc_feats_path, file_name)
        make_doc_feats(phr_feats_file, doc_feats_file, identifier, year)
    s_list.close()

# pf2dfeats.test_p2d()
def test_p2d():
    input_phr_feats = "/home/j/anick/fuse/data/patents/en_test/phr_feats/US20110052365A1.xml"
    output_doc_feats = "/home/j/anick/fuse/data/patents/en_test/doc_feats/US20110052365A1.xml"
    year = "1980"
    (doc_id, extension) = input_phr_feats.split(".")
    make_doc_feats(input_phr_feats, output_doc_feats, doc_id, year)


if __name__ == '__main__':
    import sys
    (phr_feats, doc_feats, doc_id, year) = sys.argv[1:]
    make_doc_feats(phr_feats, doc_feats, doc_id, year)


"""

python pf2dfeats.py data/patents/en/data/d3_phr_feats/01/files/home/j/corpuswork/fuse/fuse-patents/500-patents/DATA/Lexis-Nexis/US/Xml/2009/US20090032458A1.xml data/tmp.US20090032458A1.doc_feats.txt US20090032458A1.xml 2009

diff data/patents/en/data/d4_doc_feats/01/files/home/j/corpuswork/fuse/fuse-patents/500-patents/DATA/Lexis-Nexis/US/Xml/2009/US20090032458A1.xml data/tmp.US20090032458A1.doc_feats.txt

"""
