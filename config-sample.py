"""Configuration file

File with configuration settings, which is really just the location of the
Mallet toolkit. 

Copy this file into config.py and edit as needed. With the current setting the
expectation is that this repository has a tools section with mallet-2.0.7 in it
(or a soft or hard link to the real Mallet directory).

"""

import os

MALLET_DIR = os.path.join('tools', 'mallet-2.0.7', 'bin')
