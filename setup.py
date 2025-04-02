from setuptools import setup, find_packages

setup(
    name="mlchemy",  # Package name
    version="0.1.0",  # Initial version
    author="Dracarys",
    author_email="rohit.kaggle@gmail.com",
    description="🔮 MLchemy – The Magic Wand for Machine Learning Predictions 🪄✨",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rohitsingh02/MLchemy",  # Update with your GitHub repo
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",  # For boosting models
        "lightgbm", # Optional but useful
        "shap",     # Feature importance & interpretability
        "optuna",   # Hyperparameter tuning
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    license="MIT",
    entry_points={
        "console_scripts": [
            "mlchemy=mlchemy.cli.main:main",  # If you want a CLI interface
        ],
    },
)