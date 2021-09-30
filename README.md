# ml-face-recognition-demo
A face recognition demo for the machine learning group presentation on the institute's day

## Running on the GPU

In order to be able to run on the GPU you need the dlib library to be compiled with GPU capability.

1. Get dlib package from github
2. Try to compile (roughly using [this](https://stackoverflow.com/questions/51697468/how-to-check-if-dlib-is-using-gpu-or-not) instruction)
3. During build you'll notice missing packages. Among those libcudnn and cuda are the most important. [Here](https://gist.github.com/matheustguimaraes/43e0b65aa534db4df2918f835b9b361d)'s instructions for libcudnn but easiest is to follow the steps from Nvidia page itself and using the package manager e.g.`apt`.
4. You may need to set your gcc to a lower version like gcc-7 in order to get a successful build for gpu. The build process will warn you if that is the case.

NB: `python setup.sh install` builds and installs python but you may need to remove previously install `dlib` via `pip uninstall dlib`. Make sure to check the dlib version before and after installation via `dlib.__version__` and check the gpu availability on dlib via `dlib.DLIB_USE_CUDA` within python.

Once the code runs on GPU, you should be able to see the process on gpu via `nvidia-smi` command.
