# **Installment and Running Instructions**

***

### This is the Installment Requirements that are needed for this project.
### For those can be that installed via PIP, please refer to to _requirements.txt_ in the the same folder and type `pip install requirements.txt`, which will download the packages and their dependencies. 

### For training a model or validate on **Facenet** or __YOLOv3__, a gpu version of Tensorflow and CUDA is recommended. Please follow the download instructions of Tensorflow: https://www.tensorflow.org/install/ (GPU version requires CUDA and Tensorflow will also show how to install)

**If Using a GPU on Facenet, please add:**
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```
**at the import section of the python files that are desired to run with GPU**

## 1. YOLOv3
**Darknet** [Repo](https://github.com/pjreddie/darknet): `git clone https://github.com/pjreddie/darknet.git`

- To run Yolov3 with darkent, the Yolov3 pre-trained model can be downloaded [here](wget https://pjreddie.com/media/files/yolov3.weights)
- Since we have trained two models (Although soon found out the presicions on them were low due to the limiation of tools) in `Model/Set_A` and `Model/Set_B`
- Put one of the `.weights` model in the root folder of `darknet/` folder, and put `obj.names` and `obj.data` from `Model folder` in `darknet/data/`, and `yolo-obj.cfg` in `darknet/cfg/`.
- Choose `train*.txt` and `test*.txt` from `kfoldtxt/` folder and copy to `darknet/` repo, where **'*'** is the same letter from _'A'_ to _'E'_. These two files will tell darknet to find the images used for training and testing. 
- Copy all the images and txt files in `Images/001/` to `darknet/data/Bush`, note if there is no such folder simplely create one.

    **For information on how to train or detect, please refer to _readme.md_ from this [repo](https://github.com/AlexeyAB/darknet)** 

## 2. Facenet 
**Facenet** [repo](https://github.com/davidsandberg/facenet): ` git clone https://github.com/davidsandberg/facenet.git`

- Set environment variable PYTHONPATH that used in facenet: `export PYTHONPATH=[...]/facenet/src`, where `[...]` should be replaced with the directory where the cloned facenet repo resides.
- For training, put the `datasets` and `Model` folder in the same level of `facenet` repo so that facenet executable could be founded on . 
- Some of the commands can be founded in `commands.sh` command. (**Note**: the shell script is used only for reference, running directly might have some issues)

    **For more information on training and validating, please refer to [wiki](https://github.com/davidsandberg/facenet/wiki)

## 3. SVM 
Run the test.py that is located in `SVM_part/`, The pre-processed dataset images using are located in `SVM_part/data/`



