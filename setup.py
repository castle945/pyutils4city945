from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pu4c",
    version="1.1.5",
    packages=find_packages(exclude=["test"]),
    author="city945",
    author_email="city945@njust.edu.cn",
    url="https://github.com/city945",
    description="A python utils package for city945",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'rpyc',
    ],
)