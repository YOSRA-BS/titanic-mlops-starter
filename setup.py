from setuptools import setup, find_packages

setup(
    name="titanic-mlops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[ #ce qu’il faut pour utiliser la librairie, pas pour la développer/tester
        "mlflow>=2.9.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.9.0",
        "flask>=3.0.0",
    ],
    python_requires=">=3.8",
)
#Transformer ton projet en package installable (pip install .)
#pour que le projet soit utilisé comme librairie dans une env (API , batch job ...)