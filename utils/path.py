import os, errno, subprocess, gzip, codecs


def open_input_file(filename):
    """First checks whether there is a gzipped version of filename, if so, it
    returns a StreamReader instance. Otherwise, filename is a regular
    uncompressed file and a file object is returned."""
    # TODO: generalize this over reading and writing (or create two methods)
    if os.path.exists(filename + '.gz'):
        gzipfile = gzip.open(filename + '.gz', 'rb')
        reader = codecs.getreader('utf-8')
        return reader(gzipfile)
    elif os.path.exists(filename):
        # fallback case, possibly needed for older runs
        return codecs.open(filename, encoding='utf-8')
    else: 
        print "[file.py open_input_file] file does not exist: %s" % filename


def open_output_file(fname, compress=True):
    """Return a StreamWriter instance on the gzip file object if compress is
    True, otherwise return a file object."""
    if compress:
        if fname.endswith('.gz'):
            gzipfile = gzip.open(fname, 'wb')
        else:
            gzipfile = gzip.open(fname + '.gz', 'wb')
        writer = codecs.getwriter('utf-8')
        return writer(gzipfile)
    else:
        return codecs.open(fname, 'w', encoding='utf-8')


def ensure_path(path, verbose=False):
    """Make sure path exists."""
    try:
        os.makedirs(path)
        if verbose:
            print "[ensure_path] created %s" % path
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_file(filename, content=None):
    """Create a file with name filename and write content to it if any was given."""
    fh = open(filename, 'w')
    if content is not None:
        fh.write(content)
    fh.close()


def filename_generator(path, filelist):
    """Creates generator on the filelist, yielding the concatenation of the path
    and a path in filelist."""
    fh = open(filelist)
    for line in fh:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        fspec = FileSpec(line)
        yield os.path.join(path, 'files', fspec.target)
    fh.close()


def compress(*fnames):
    """Compress all filenames fname in *fnames using gzip. Checks first if the
    file was already compressed."""
    for fname in fnames:
        if fname.endswith(".gz"):
            continue
        if os.path.exists(fname + '.gz'):
            continue
        subprocess.call(['gzip', fname])


def get_year_and_docid(path):
    """Get the year and the document name from the file path. This is a tad
    dangerous since it relies on a particular directory structure, but it works
    with how the patent directories are set up, where each patent is directly
    inside a year directory. If there is no year directory, the year returned
    will be 9999."""
    year = os.path.basename(os.path.dirname(path))
    doc_id = os.path.basename(path)
    if not (len(year) == 4 and year.isdigit()):
        year = '9999'
    return year, doc_id


class FileSpec(object):

    """A FileSpec is created from a line from a file that specifies the
    sources. Such a file has two mandatory columns: year and source_file. These
    fill the year and source instance variables in the FileSpec. The target
    instance variable is by default the same as the source, but can be overruled
    if there is a third column in the file. Example input lines:

       1980    /data/patents/xml/us/1980/12.xml   1980/12.xml
       1980    /data/patents/xml/us/1980/13.xml   1980/13.xml
       1980    /data/patents/xml/us/1980/14.xml
       0000    /data/patents/xml/us/1980/15.xml

    FileSpec can also be created from a line with just one field, in that case
    the year and source are set to None and the target to the only field. This
    is typically used for files that simply list filenames for testing or
    training.
    """

    def __init__(self, line):
        fields = line.strip().split("\t")
        if len(fields) > 1:
            self.year = fields[0]
            self.source = fields[1]
            self.target = fields[2] if len(fields) > 2 else fields[1]
        else:
            self.year = None
            self.source = None
            self.target = fields[0]
        self._strip_slashes()

    def __str__(self):
        return "%s\n  %s\n  %s" % (self.year, self.source, self.target)

    def _strip_slashes(self):
        if self.target.startswith(os.sep):
            self.target = self.target[1:]
