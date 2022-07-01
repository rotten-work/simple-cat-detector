# Simple Cat Detector
A simple cat detector using neural networks and deep learning based on VGG using Tensorflow Serving with Docker.

## Documentation

### Set up
* [Install Anaconda](https://www.anaconda.com/)
* [Install Tensorflow](https://www.tensorflow.org/install/pip#windows)
* [Enable the WSL 2 feature on Windows](https://docs.microsoft.com/en-us/windows/wsl/install)
* [Install Docker Desktop on Windows](https://docs.docker.com/desktop/windows/install/)
* [Install Tensorflow Serving using Docker](https://github.com/tensorflow/serving)

### Use
Start TensorFlow Serving container and open the REST API port using a WSL terminal:

```bash
# Change the '/mnt/d/GitHub/' to your own repository path
docker run --name vgg_test -t -p 8501:8501 --mount type=bind,source=/mnt/d,target=/mnt/d -e MODEL_NAME=vgg_serving -e MODEL_BASE_PATH=/mnt/d/GitHub/simple-cat-detector tensorflow/serving &
```

Query the model using the predict API in an Anaconda Powershell Prompt:

```bash
python request_vgg_serving.py
```



