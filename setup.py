import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplearning_ai_vik228", # Replace with your own username
    version="0.0.1",
    author="Vikas Pandey",
    author_email="vik.iiitmg@gmail.com",
    description="Basic wrapper over plotting libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vik228/deeplearning_ai.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)