try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

NAME = 'girs'

VERSION = '0.0.1'

DESCRIPTION = 'gdal/ogr wrapper'

LONG_DESCRIPTION = 'gdal/ogr wrapper'

CLASSIFIERS = [  # https://pypi.python.org/pypi?:action=list_classifiers
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Cython',
    'Programming Language :: C++',
    'Topic :: Scientific/Engineering :: GIS'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url='https://github.com/JRoehrig/girs',
    url='http://warsa.de/girs/',
    author='Jackson Roehrig',
    author_email='Jackson.Roehrig@th-koeln.de',
	license='MIT',
	classifiers=CLASSIFIERS,
    install_requires=['numpy', 'scipy', 'pandas', 'gdal', 'matplotlib', 'pillow'],
    packages=['girs'],
    scripts=[]
)

