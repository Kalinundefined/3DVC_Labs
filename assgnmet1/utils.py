import os

def path_(r):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base,'data', r))