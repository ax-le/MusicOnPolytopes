import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polytopes",
    version="0.1.0",
    author="Marmoret Axel",
    author_email="axel.marmoret@irisa.fr",
    description="Package for polytopes applied on musical segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",    
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.7"
    ],
    license='BSD',
    install_requires=[
        'madmom',
        'matplotlib',
        'mir_eval',
        'numpy >= 1.18.0',
        'pandas',
        'scipy >= 0.13.0',
        'soundfile',
    ],
    python_requires='>=3.7',
)