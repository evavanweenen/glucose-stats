from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='glucose-stats',
    version='0.0.4',
    author='Eva van Weenen',
    author_email='evanweenen@ethz.ch',
    description='Calculate glucose statistics from continuous glucose monitoring data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/evavanweenen/glucose-stats',
    license='MIT',
    packages=['glucosestats'],
    scripts=['bin/run.py'],
    zip_safe=False,
    install_requires=['numpy', 'pandas','torch','matplotlib','seaborn']
    )