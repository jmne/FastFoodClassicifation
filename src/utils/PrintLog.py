import os


def print_to_file(_str):
    if not os.path.isdir('../logs'):
        os.mkdir('../logs')
    if not os.path.isdir('../logs/plots'):
        os.mkdir('../logs/plots')
    f = open("../logs/log.txt", 'a')
    print(_str + "\n")
    f.writelines(_str + "\n")
