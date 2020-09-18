import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parampicker", # Replace with your own username
    version="0.0.1",
    author="Khurram Javed",
    author_email="khurramjaved1996@gmail.com",
    description="A package to select parameter settings using rank for grid searches in machine learning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khurramjaved96/experiment",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)