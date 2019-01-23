from builtins import str
from past.builtins import basestring
import os
import time
import shutil
# from os import makedirs, listdir, sep, unlink, rmdir, remove
# from os.path import dirname, exists, isfile, isdir, join
# from shutil import copy2


def touch(file_name):
    assure_path_exists(file_name)
    open(file_name, 'a').close()


def file_exists(file_name):
    return os.path.isfile(file_name)


def assure_path_exists(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def get_file_names(directory, suffix=None, prefix=None, pattern=None):
    if not os.path.isdir(directory):
        return []
    if directory[-1] == os.sep:
        directory = directory[:-1]

    result = os.listdir(directory)
    if prefix:
        if isinstance(prefix, basestring):
            result = [f for f in result if f.startswith(prefix)]
        else:
            result = [f for f in result for p in prefix if f.startswith(p)]
    if suffix:
        if isinstance(suffix, basestring):
            result = [f for f in result if f.endswith(suffix)]
        else:
            result = [f for f in result for s in suffix if f.endswith(s)]
    if pattern:
        if isinstance(pattern, basestring):
            result = [f for f in result if pattern in f]
        else:
            result = [f for f in result for p in pattern if p in f]
    return result


def copy_file(src, dst):
    shutil.copy2(src, dst)


def delete_file(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def delete_files(path, prefix):
    if os.path.isdir(path):
        for f in [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]:
            if f.startswith(prefix):
                os.remove(path + '/' + f)


def get_subdirectories(directory):
    if not os.path.isdir(directory):
        return []
    if directory[-1] == os.sep:
        directory = directory[:-1]
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def create_tmp_directory(directory):
    """
    Create a temporary host_directory in the given host_directory and return it
    """
    timestamp = str(int(time.time()))
    # Create a new temporary work host_directory. Append a '/' even if there is
    # already one. Append timestamp
    wd = os.path.join(directory, 'tmp' + timestamp)
    i = 0
    while os.path.isdir(wd):
        wd = directory + str(i)
        i += 1
    os.makedirs(wd)
    return wd


def remove_directory(directory):
    if not os.path.isdir(directory):
        return
    if directory[-1] == os.sep:
        directory = directory[:-1]
    files = os.listdir(directory)
    for filename in files:
        if filename == '.' or filename == '..':
            continue
        path = directory + os.sep + filename
        if os.path.isdir(path):
            remove_directory(path)
        else:
            os.unlink(path)
    os.rmdir(directory)


