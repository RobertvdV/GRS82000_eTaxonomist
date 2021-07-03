from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project aims at solving this problem by splitting the system into two agents that communicate using natural language. The first is a visual-language hybrid model that takes an image as input and generates descriptions of the image in natural language using the vocabulary of expert taxonomists, but is not aware of species names. The second is a pure language model that takes as input the description provided by the first and outputs the corresponding species name.',
    author='Robert van de Vlasakker',
    license='MIT',
)
