# Running the RBC Model with ONNX Runtime

You will need at least Python 3.8! A newer version may also work, but you may need to regenerate the cython code required by Yolo.
Run the `setup.py` script in that case!

## Downloading the weights

Similar to the original RBC model, we need to first download the weights for the model:

- Weights: [```download```](https://1drv.ms/u/s!AlXVRhh1rUKThlxTievX0X1CpXd0?e=9cKxYb) the trained weights file for blood cell detection and put the ```weights``` folder in the working directory.

Execution of the following Python scripts will fail if the weights are not available.

The training data is available from GitHub: https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset.git. Check the Dockerfile.train for details on how to download it.

## Installing the runtime environment with Conda

The `environment.yml` contains the libraries that were installed using Conda. 
To recreate the environment on your system, please install Conda / MiniConda first.

Then run:

```
conda env create -f environment.yml
```

and

```
conda activate RBCs
```

to activate the environment. This should install all necessary dependencies independently of your local Python installation and packages.

Alternatively, you can create the environment from scratch:

```
conda install onnx onnxruntime onnxconverter-common numpy tensorflow tf2onnx tf_slim opencv-python-headless scipy cython tensorboard
```

If you want to use GPU accelaration for training, you may need to add the `tensorflow-gpu` library to the conda install command.

## Running the code

Make sure that the RBCs Conda environment is activated:

```
conda activate RBCs
```

Then run:

```
python3 predict-tf-and-create-onnx.py
```

This will first run the tensorflow model for RBCs and write out the model graph with weights.
In turn, the serialized model will be parsed and converted to ONNX format.

Then run:

```
python3 predict-onnx.py
```

This will load the ONNX model and will create and execute an inference session with the ONNX runtime.

Result output for both models on the test image01 is written to the output folder.

## Building the Docker containers

In order to train the original model within a Docker container which contains all necessary dependencies, run:

```
docker build -f . -t "rbc-tf-train"
```

When you run the docker container, make sure to mount the output folder to your local directory to be able to access the ONNX model and other files:

```
mkdir dockeroutput
docker run -it -v ./dockeroutput:/rbc/output --rm rbc-tf-train
```

In order to run the ONNX model and use it for inference, run:

```
docker build -f . -t "rbc-onnx-inference"
```

Run the inference container as follows, mounting the local directory that contains the images to classify to the /rbc/data path:

```
docker run -it -v ./dockeroutput:/rbc/output -v ./data:/rbc/data --rm rbc-onnx-inference
```
