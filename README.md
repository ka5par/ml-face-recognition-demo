# ml-face-recognition-demo with animal classifier
A face recognition demo for the machine learning group presentation on the institute's day. This is with an animal classifier extension. 

## Running on the GPU

In order to be able to run on the GPU you need the dlib library to be compiled with GPU capability.

1. Follow this for dlib https://gist.github.com/nguyenhoan1988/ed92d58054b985a1b45a521fcf8fa781

NB: `python setup.sh install` builds and installs python but you may need to remove previously install `dlib` via `pip uninstall dlib`. Make sure to check the dlib version before and after installation via `dlib.__version__` and check the gpu availability on dlib via `dlib.DLIB_USE_CUDA` within python.

NB: find nvcc path with 'which nvcc' and put it to '-DCUDAToolkit_ROOT=**********'

NB: Make sure cuda and cudnn versions in python env are same to the ones on your computer. 

2. Download [Animals-151 Dataset](https://www.kaggle.com/datasets/sharansmenon/animals141). Unzip it into `data/Animals-151` folder. Make so that all the subfolders in `data/Animals-151` are for a specific class (like ImageNet). 

Once the code runs on GPU, you should be able to see the process on gpu via `nvidia-smi` command.


_Disclaimer: This demo is based on the code on this [article](https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340), and only some whistles and bells were added for the purpose of the demo._
