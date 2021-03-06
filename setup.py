from setuptools import setup, find_packages
NAME = 'girs'

VERSION = '0.1.1'

DESCRIPTION = 'gdal/ogr wrapper'

LONG_DESCRIPTION = 'gdal/ogr wrapper'

CLASSIFIERS = [  # https://pypi.python.org/pypi?:action=list_classifiers
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
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
    install_requires=['numpy', 'scipy', 'pandas', 'gdal', 'matplotlib'],
    packages=find_packages(),
    scripts=[]
)

