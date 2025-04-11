import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="teamFormationLibrary",
    version="1.0.0",
    description="Generate predictive teams based on a VAE",
    long_description=README,
    long_description_content_type="text/markdown",
    #url="https://github.com/realpython/reader",
    author="Aabid Mitha",
    author_email="aabid.mitha@ontariotechu.net",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["teamFormationLibrary"],
    include_package_data=True,
    install_requires=["tensorflow==1.15.0", "keras==2.0.0", "gensim", "nltk==3.5", "scikit-learn", "sklearn"],
)
