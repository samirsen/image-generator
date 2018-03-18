"""Downloads the GloVe vectors and unzips them"""

import zipfile
import argparse
import os
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner

def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename.
    num_bytes=None disables the file size check."""
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print "Downloading file {}...".format(url + filename)
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix, filename), reporthook=reporthook(t))
        except AttributeError as e:
            print "An error occurred when downloading the file! Please get the dataset using a browser."
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print "File {} successfully loaded".format(filename)
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", required=True) # where to put the downloaded glove files
    return parser.parse_args()


def main():
    args = setup_args()
    glove_base_url = "http://nlp.stanford.edu/data/"
    glove_filename = "glove.6B.zip"

    print "\nDownloading wordvecs to {}".format(args.download_dir)

    if not os.path.exists(args.download_dir):
        os.makedirs(args.download_dir)

    maybe_download(glove_base_url, glove_filename, args.download_dir, 862182613L)
    glove_zip_ref = zipfile.ZipFile(os.path.join(args.download_dir, glove_filename), 'r')

    glove_zip_ref.extractall(args.download_dir)
    glove_zip_ref.close()


if __name__ == '__main__':
    main()
