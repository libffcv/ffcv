from setuptools import find_packages
import subprocess
from glob import glob

from distutils.core import setup, Extension

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


def pkgconfig(package, kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = subprocess.getoutput(
        'pkg-config --cflags --libs {}'.format(package))
    if 'not found' in output:
        raise Exception(f"Could not find required package: {package}.")
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
      version='0.0.3rc1',
      description=' FFCV: Fast Forward Computer Vision ',
      author='MadryLab',
      author_email='leclerc@mit.edu',
      url='https://github.com/libffcv/ffcv',
      license_files = ('LICENSE.txt',),
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      ext_modules=[libffcv],
      install_requires=[
          'terminaltables',
            'pytorch_pfn_extras',
            'fastargs',
            'matplotlib',
            'scikit-learn',
            'imgcat',
            'pandas',
            'assertpy',
            'tqdm',
            'psutil',
            'webdataset',
      ]
      )
