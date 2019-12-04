import os
import tarfile
import sys
import urllib

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_gtzan(data_dir):
    flag_file = os.path.join(data_dir, 'genres_done.txt')
    if os.path.exists(flag_file):
        return

    zip_file_path = data_dir + '/genres.tar.gz'

    if not os.path.exists(zip_file_path):
        print('gz model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=DATA_URL, filename=zip_file_path,
                                   reporthook=reporthook)

    tar = tarfile.open(zip_file_path, "r:gz")
    tar.extractall(data_dir)
    tar.close()

    with open(flag_file, 'wt') as file:
        file.write('done')
