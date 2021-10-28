from setuptools import find_packages
import subprocess
from glob import glob

from distutils.core import setup, Extension


def pkgconfig(package, kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = subprocess.getoutput(
        'pkg-config --cflags --libs {}'.format(package))
    if 'not found' in output:
        raise Exception()
    for token in output.strip().split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


sources = ['./libffcv/libffcv.cpp']

extension_kwargs = {
    'sources': sources,
    'include_dirs': []
}
extension_kwargs = pkgconfig('opencv4', extension_kwargs)
extension_kwargs = pkgconfig('libturbojpeg', extension_kwargs)

extension_kwargs['libraries'].append('pthread')


libffcv = Extension('ffcv._libffcv',
                        **extension_kwargs)

setup(name='ffcv',
      version='0.1',
      description='Training go BrrRr',
      author='Guillaume Leclerc',
      author_email='leclerc@mit.edu',
      url='https://github.com/MadryLab/fastercv',
      long_description='''
Load and train computer vision fast
       ''',
      packages=find_packages(),
      ext_modules=[libffcv]
      )
