from setuptools import setup, find_packages

setup(
    name="d100-d400-project",  # Replace with your package name
    version="0.1.0",
    author="Yingshan Qi",
    author_email="yingshanqiuk@outlook.com",
    description="A project for predicting used car prices",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/YingshQi/D100-D400_Project.git",  # Replace with your repository URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",  # Use >= to support later versions
    install_requires=[
        "numpy==1.26.4",
        "matplotlib>=3.8.0",
        "seaborn==0.13.2",
        "scikit-learn==1.6.0",
        "lightgbm==4.5.0",
        "pytest==8.3.4",
        "ipykernel==6.29.4",
        "jupyterlab==4.2.1",
        "joblib==1.4.2",
        "pathlib",  # No version needed; part of Python standard library
        "pyarrow",  # Add version if needed
        "notebook"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",  # Change if necessary
        "Operating System :: OS Independent",
    ],
)
