from setuptools import setup, find_packages

setup(
    name="alm_ir_hedging_prototype",
    version="0.1.0",
    author="Magomed Dubayev",
    author_email="dubayevmagomed@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)