
# import openslide
import sys
from ctypes import cdll
import platform

cdll.LoadLibrary('libopenslide-0.dll')
print('hello')
