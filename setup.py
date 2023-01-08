from setuptools import setup, find_packages

setup(
    name='little_helpers',
    version='0.0.5',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'scikit-learn', 'scipy', 'statsmodels', 'matplotlib', 'sympy', 'python-docx', 'latex2mathml', 'lxml']
)
