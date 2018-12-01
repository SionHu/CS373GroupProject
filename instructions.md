# **Installment Instructions**

***

### This is the Installment Requirements that are needed for this project.
### For those can be that installed via PIP, please refer to to _requirements.txt_ in the the same folder and type `pip install requirements.txt`, which will download the packages and their dependencies. 

### For training a model or validate on **Facenet** or __YOLOv3__, a gpu version of Tensorflow and CUDA is recommended. 

**If Using a GPU on Facenet, please add:**
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```
**at the import section of the python files taht are desired to run with GPU**

## 1. YOLOv3
**Darknet** [Repo](https://github.com/AlexeyAB/darknet/tree/master/build/darknet):

