from setuptools import setup, find_packages

# Read requirements.txt and transform it into a list
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='GraFT',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements
)
