from setuptools import setup, find_packages

setup(
    name="aimon",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # dd with cudd is a dependency, needs to be installed via provided installer script
    ],
)

