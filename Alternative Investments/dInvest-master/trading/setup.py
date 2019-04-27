#!/usr/bin/env python

from setuptools import setup, find_packages
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_requirements = parse_requirements('requirements.txt', session=False)

# requirements is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
requirements = [str(ir.req) for ir in install_requirements]

setup(
        name='dinvest',
        version='1.0.0',
        description='Shortest Path Transformation of Words from a Dictionary',
        author='Dominik Harz',
        author_email='dominik.harz@gmail.com',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python :: 2.7',
        ],
        install_requires=requirements,
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        packages=find_packages(exclude=['analysis', 'contrib', 'docs', 'tests', 'venv']),
        scripts=[
            'recommender/InvestHandler.py',
            'trader/TradeHandler.py',
        ],
        entry_points={
            'console_scripts': ['recommender=recommender.InvestHandler:main',
                                'trader=trader.TradeHandler:main'],

        },
      )
