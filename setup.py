from setuptools import find_packages, setup

setup(
    name="src",  # you should change "src" to your project name
    version="0.0.0",
    description="Fire Prediction Pytorch Lightning tmpl",
    author="",
    author_email="",
    url="https://github.com/iprapas/fire-prediction-pl",
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6", "torch==1.8.1", "torchvision==0.9.1",
                      "pytorch-lightning==1.5.8", "fastai==2.5.2"],
    packages=find_packages(),
)
