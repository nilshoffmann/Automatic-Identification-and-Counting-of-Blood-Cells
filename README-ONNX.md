# Running the RBC Model with ONNX Runtime

## Downloading the weights

Similar to the original RBC model, we need to first download the weights for the model:

- Weights: [```download```](https://1drv.ms/u/s!AlXVRhh1rUKThlxTievX0X1CpXd0?e=9cKxYb) the trained weights file for blood cell detection and put the ```weights``` folder in the working directory.

Execution of the following Python scripts will fail if the weights are not available.

## Installing the runtime environment with Conda

The `environment.txt` contains the libraries that were installed using Conda. 
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

## Running the code

Make sure that the RBCs Conda environment is activated:

```
conda activate RBCs
```

Then run:

```
python3 detect-and-predict-onnx.py
```

This will first run the tensorflow model for RBCs and write out the model graph with weights.
In turn, the serialized model will be parsed and converted to ONNX format.
Thereafter, the ONNX model will be loaded and an inference session is created and executed with ONNX runtime.

Result output for both models on the test image01 are written to serialized numpy arrays as 'yolov2-predictions.npy' and 'onnx-predictions.npy' respectively.
