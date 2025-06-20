from setuptools import setup, find_packages

# Legge il file requirements.txt
with open("requirements.txt", encoding="utf-16-le") as f:
    requirements = f.read().splitlines()


setup(
    name='rif_estimator',
    version='0.1.0',
    description='Scikit-learn compatible estimator for contextual anomaly detection based on residuals and Isolation Forest.',
    author='Giulio Surya Lo Verde',
    url='https://github.com/GiulioSurya/RIF_estimator_scikit',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Framework :: Scikit-Learn',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
