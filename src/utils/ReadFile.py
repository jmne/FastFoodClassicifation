# read folder with images
import os


def read_folder():
    """Read folder with images

    Returns:
        array: array with folder names
    """
    folders = []
    for file in os.listdir("resources/"):
        if os.path.isdir(os.path.join("resources/", file)):
            folders.append(file)
    return folders
        