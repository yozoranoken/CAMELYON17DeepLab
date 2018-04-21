Breast Cancer Tumor Detection and pN Stage Classification
=============================

## System Requirements
1. [OpenSlide Tools](http://openslide.org/) `sudo apt-get install openslide-tools`
2. [Anaconda](https://www.anaconda.com/download/#linux)
3. Python 3.6 via an Anaconda environment `conda create -n tf-deeplab python=3.6`
4. [Automated Slide Analysis Platform (ASAP)](https://github.com/GeertLitjens/ASAP/releases) 
`sudo dpkg -i ASAP-X.X.X-Linux-python36.deb`
5. Python packages:
	1. Local `pip` installation for environment `conda install pip`
	1. `pip install openslide-python`
	1. `pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0rc0-cp36-cp36m-linux_x86_64.whl`
	1. `conda install -c conda-forge matplotlib opencv python-spams numpy pillow scipy scikit-image`

## Setup
- To import **multiresolutionimageinterface.py**, create a file ~/.anaconda3/envs/tf-deeplab/lib/python3.6/site-packages/asap.pth containing<br />
```
/opt/ASAP/bin
```
