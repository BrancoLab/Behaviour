from setuptools import setup, find_namespace_packages

requirements = [
    "numpy",
    "opencv-python",
    "nptdms",
    "configparser",
    "pandas",
    "tqdm",
    "seaborn",
    "matplotlib",
    "vtk",
    "statsmodels",
    "nptdms",
    "fancylog",
    "termcolor"

]


setup(
    name="behaviour",
<<<<<<< HEAD
    version="0.0.2.5",
=======
    version="0.0.3.1",
>>>>>>> f801512a2721bd41c9f07d05e0db80b47958e378
    author_email="federicoclaudi@protonmail.com",
    description="bunch of utility functions to analyse behaviour data",
    packages=find_namespace_packages(exclude=()),
    include_package_data=True,
    install_requires=requirements,
    url="https://github.com/BrancoLab/Behaviour",
    author="Federico Claudi",
    zip_safe=False,
)
