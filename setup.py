from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

with open('requirements/base.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="deeplearning_ai",
    version="0.1",
    packages=find_packages(),
    author='Vikas Pandey',
    install_requires=requirements,
    include_package_data=True,
    author_email='vikas.pandey@anarock.com',
    description='Implementations of ML and DL algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vik228/deeplearning_ai',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
