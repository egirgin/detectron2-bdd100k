# detectron2-bdd100k
BDD100K Lane Segmentation using Detectron2 API


## Installation

Learn your cuda and torch version

```python
nvcc --version
pip list | grep torch
```

Install appropriate torch version
```bash
# https://pytorch.org/get-started/previous-versions/
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install pyyaml

```bash
# install dependencies: 
!pip install pyyaml==5.1
!gcc --version
# opencv is pre-installed on colab
```

Check torch again
```python
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
```
Install appropriate detectron2 version
```python
# install detectron2: (Colab has CUDA 10.1 + torch 1.7)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import torch
assert torch.__version__.startswith("1.7")   # please manually install torch 1.7 if Colab changes its default version
!python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime
```

## Running


```bash
source run.sh
```
# Docker

```bash
docker build -t detectron2-bdd100k .
```
add --no-cache flag if you do not want to use pre-downloaded images 


```bash
docker run --gpus all -d detectron2-bdd100k bash run.sh
```
-d detaches you from the container
add -it to have interactive pseudo-tty

check logs (stdout) of the progrom with the command
```bash
docker logs -f <containerId>
```
sample
```bash
docker logs -f d362659da5fc
```
Get the output file (checkpoints, tensorboard etc.)
```bash
docker cp <containerId>:/home/appuser/detectron2-bdd100k/output .
```
sample: 
``` bash
docker cp d362659da5fc:/home/appuser/detectron2-bdd100k/output .
```


