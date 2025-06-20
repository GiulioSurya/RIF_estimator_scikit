import os

from setuptools import setup, find_packages

# Read the requirements.txt file with proper encoding handling
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name='rif_estimator',
    version='0.1.0',
    description='Scikit-learn compatible estimator for contextual anomaly detection based on residuals and Isolation Forest.',
    author='Giulio Surya Lo Verde',
    url='https://github.com/GiulioSurya/RIF_estimator_scikit',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Framework :: Scikit-Learn',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='anomaly-detection isolation-forest scikit-learn contextual-anomalies',
    long_description=open('README.md', 'r', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
)