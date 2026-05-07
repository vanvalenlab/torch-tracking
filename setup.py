from setuptools import setup, find_packages

setup(
    name='torch-tracking',
    version='0.0.1',
    author='Sam Holtzen',
    author_email='sholtzen@caltech.edu',
    description='a cell tracking pipeline in pytorch',
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.1',
        'numpy>=1.19.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
