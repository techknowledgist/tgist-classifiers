"""

Utility code to explore the content of a file.

"""

import os
import sys

from path import open_input_file


class FileData(object):
    """An instance contains a terms dictionary with term information from the
    d3_feats file, amended with a context taken from the d2_tags file. Each term
    is an instance of Term and contains a list of TermInstances. Each
    TermInstance provides access to the features and the context of the
    instance.

    This is not used in processing modules but it is useful for data exploration.
    """

    def __init__(self, tag_file, feat_file, verbose=False):
        self.verbose = verbose
        self.tag_file = tag_file
        self.feat_file = feat_file
        self._init_collect_lines_from_tag_file()
        self._init_collect_term_info_from_phrfeats_file()
        self._init_amend_term_info()

    def __str__(self):
        return "<FileData\n   %s\n   %s>" % (self.tag_file, self.feat_file)

    def get_title(self):
        for section, line in self.tags:
            if section == 'FH_TITLE:':
                return ' '.join(line)
        return ''

    def get_abstract(self):
        abstract = []
        for section, line in self.tags:
            if section == 'FH_ABSTRACT:':
                abstract.append(' '.join(line))
        return ' '.join(abstract)

    def get_term(self, term):
        """Return the Term instance for term or None if term is not in the
        dictionary."""
        return self.terms.get(term)

    def get_terms(self):
        """Returns the list of terms (just the strings) of all terms in the
        dictionary."""
        return self.terms.keys()

    def get_term_instances_dictionary(self):
        """Returns a dictionary indexed on document offsets (sentence
        numbers). The values are lists of TermInstances."""
        terms = {}
        for t in self.get_terms():
            term = self.get_term(t)
            for inst in term.term_instances:
                terms.setdefault(inst.doc_loc, []).append(inst)
        return terms

    def _init_collect_lines_from_tag_file(self):
        self.tags = []
        with open_input_file(self.tag_file) as fh:
            section = None
            for line in fh:
                if line.startswith('FH_'):
                    section = line.strip()
                else:
                    tokens = line.rstrip().split(' ')
                    tokens = [t.rpartition('_')[0] for t in tokens]
                    self.tags.append([section, tokens])

    def _init_collect_term_info_from_phrfeats_file(self):
        self.terms = {}
        with open_input_file(self.feat_file) as fh:
            section = None
            for line in fh:
                (id, year, term, feats) = parse_feats_line(line)
                locfeats = dict((k, v) for (k, v) in feats.items()
                                if k.endswith('_loc'))
                self.terms.setdefault(term, []).append([id, year, feats, locfeats])

    def _init_amend_term_info(self):
        """Replaces the term_data lists in the self.terms dictionary with
        instances of the Term class. Also adds context information from the tag
        data."""
        for term in self.terms:
            t = Term(term)
            for term_data in self.terms[term]:
                term_instance = TermInstance(term, term_data)
                context = self.tags[term_instance.doc_loc]
                term_instance.add_context(context)
                t.add_instance(term_instance)
            self.terms[term] = t

    def print_terms(self, limit=5):
        print "\n%s\n" % self
        count = 0
        for term in self.terms:
            count += 1
            if count > limit:
                break
            self.terms[term].pp()
            print


class Term(object):
    """A Term is basically a container for a list of TermInstances. The
    instances are accessible in the instance variable term_instances."""

    def __init__(self, term):
        self.term = term
        self.term_instances = []

    def __str__(self):
        term_string = "<Term '%s'>" % self.term
        return term_string.encode("UTF-8")

    def add_instance(self, instance):
        self.term_instances.append(instance)

    def pp(self):
        print self
        for instance in self.term_instances:
            print "  %s" % instance


class TermInstance(object):
    """A TermInstance provides access to (i) all features for the term, (ii) the
    context of the term, and (iii) the position of the term in the document and
    the context (as a list of tokens)."""

    def __init__(self, term, term_data):
        self.term = term
        self.id = term_data[0]
        self.doc = term_data[0].rstrip('01234567890')[:-5]
        self.year = term_data[1]
        self.feats = term_data[2]
        doc_loc = self.feats.get('doc_loc')
        sent_loc = self.feats.get('sent_loc')
        if doc_loc.startswith('sent'):
            doc_loc = doc_loc[4:]
        tok1, tok2 = sent_loc.split('-')
        self.sec_loc = self.feats.get('section_loc')
        self.doc_loc = int(doc_loc)
        self.sent_loc = (int(tok1), int(tok2))
        self.tok1 = int(tok1)
        self.tok2 = int(tok2)

    def __str__(self):
        string = "<TermInstance %s %s %d-%d '%s'>" \
                 % (self.id, self.doc_loc, self.tok1, self.tok2, self.context_token())
        return string.encode("UTF-8")

    def __cmp__(self, other):
        comparison1 = cmp(self.doc_loc, other.doc_loc)
        if comparison1 != 0:
            return comparison1
        return cmp(self.tok1, other.tok1)

    def add_context(self, context):
        self.context = context

    def context_section(self):
        return self.context[0]

    def context_all(self):
        return "%s [%s] %s" % (self.context_left(), self.context_token(), self.context_right())

    def context_token(self):
        return ' '.join(self.context[1][self.tok1:self.tok2])

    def context_left(self):
        return ' '.join(self.context[1][:self.tok1])

    def context_right(self):
        return ' '.join(self.context[1][self.tok2:])

    def check_feature(self, feat, val):
        return self.feats.get(feat) == val

    def print_as_tabbed_line(self, fh):
        fh.write("\t%s\t%s\t%s\t%s\t%s\t%s\n"
                 % (self.year, self.id, self.feats.get('section_loc'),
                    self.context_left(), self.context_token(), self.context_right()))

    def print_as_html(self, fh):
        fh.write("<file>%s -- %s</file><br/>\n%s <np>%s</np> %s"
                 % (self.id, self.sec_loc,
                    self.context_left(), self.context_token(), self.context_right()))


def parse_feats_line(line):
    """Parse a line from a phr_feats file and return a tuple with id year,
    term and features."""
    # TODO: this is similar to the function parse_phr_feats_line() in
    # step6_index.py, with which it should be merged
    (id, year, term, feats) = line.strip().split("\t", 3)
    feats = feats.split("\t")
    feats = dict((k, v) for (k, v) in [f.split('=', 1) for f in feats])
    return (id, year, term, feats)


if __name__ == '__main__':

    # all you need to get the data is the corpus and the relative path
    corpus = '../data/corpora/sample-us'
    filename = '1980/US4192770A.xml'

    fd = FileData(os.path.join(corpus, 'data', 'd2_tag', '01', 'files', filename),
                  os.path.join(corpus, 'data', 'd3_feats', '01', 'files', filename))

    fd.print_terms(20)

    # now let's look into one of them
    term = fd.get_term('invention')
    print "\n", term
    for inst in term.term_instances:
        # only get those instances where we have prev_V=permitted
        # if inst.check_feature('prev_V', 'permitted'):
        #    inst.print_as_html(sys.stdout)
        #    print
        inst.print_as_tabbed_line(sys.stdout)
        print
