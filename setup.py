from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='SVMCBO',
    author='Martin Stoehr',
    author_email='martin.stoehr.research@gmail.com',
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.22.1',

    description='Utility suite for PYSEQM.',
    long_description=long_description,
    
    # The project's main homepage.
    url='https://github.com/martin-stoehr/SVMCBO',

    # Choose your license
#    license='',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Research',
        'Topic :: Optimization',

        # Pick your license as you wish (should match "license" above)
#        'License :: ',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9'
        'Programming Language :: Python :: 3.10'
    ],

    # What does your project relate to?
    keywords='Optimization',

    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=["numpy>=1.19.2", "pandas>=1.1.2", "Pillow>=7.2.0",
                      "scikit-learn>=0.23.2", "scikit-optimize>=0.8.1",
                      "scipy>=1.5.2"],
    include_package_data=True,
)
