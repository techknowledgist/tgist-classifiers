"""
Loading this module adjust sys.path by adding the root directory of the
patent-classifier code. It does not add this root directory if it is already in
the path.

"""

import os, sys

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
os.chdir('../..')
parent_dir = os.getcwd()
if not parent_dir in sys.path:
    sys.path.insert(0, os.getcwd())
os.chdir(script_dir)
