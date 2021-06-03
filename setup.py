from setuptools import find_packages, setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="odc",
    version="0.0.1",
    packages=find_packages(exclude=('tests',)),
    python_requires=">=3.6.0",
    description="Online Learning and Control of Complex Dynamical Systems from Sensory Input",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ], install_requires=['numpy', 'torchvision']
)
