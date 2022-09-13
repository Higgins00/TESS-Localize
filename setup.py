from setuptools import setup

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()
    

setup(name='TESS_Localize',
version='0.6.0',
description='Package for localizing variable stars in TESS Photometry',
url='https://github.com/Higgins00/TESS-Localize',
author='Michael Higgins',
author_email='michael.higgins@duke.edu',
license='GNU General Public License v3.0',
install_requires=requirements,
packages=['TESS_Localize'],
package_data={"": ["*.npz"]},
zip_safe=False)

