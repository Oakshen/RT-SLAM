[ENG](https://github.com/Oakshen/RT-SLAM/README.md) [中文](https://github.com/Oakshen/RT-SLAM/blob/main/README_CN.md)
# RT-SLAM
Fusion RT-DETR in ORB-SLAM3

## Dependencies
* OpenCV(>=4.2)
* glog (apt install libgoogle-glog-dev)
* TensorRT (8.6.1.6)

## How to use your own RT-DETR model

## Notes
There is no support for ROS releases at this time, but will be available in the future

## Test in Datasets
You can use [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) or [BONN](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) datasets to test this system

### TUM

```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml ~/dataset/rgbd_dataset_freiburg2_desk ~/dataset/rgbd_dataset_freiburg2_desk/associate.txt
```


## Docker image