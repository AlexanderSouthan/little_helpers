from setuptools import setup, find_packages

setup(
    name='little_helpers',
    version='0.0.2',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'scikit-learn', 'scipy', 'statsmodels']
)
