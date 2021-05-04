import setuptools


setuptools.setup(
    name="rayps",
    version="0.0.1",
    description="Parameter Server based on Ray",
    url="https://github.com/vycezhong/ray-ps",
    packages=setuptools.find_packages(exclude=("tests")),
    classifiers=[
        "Programming Language :: Python ::3",
        "License :: OSI Approved :: MIT License",
        "Operation System :: POSIX :: Linux",
    ],
    python_requires='>=3.6'
)
