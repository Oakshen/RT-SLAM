[ENG](https://github.com/Oakshen/RT-SLAM/README.md) [中文](https://github.com/Oakshen/RT-SLAM/README_CN.md)
# RT-SLAM
将 RT-DETR 目标检测模块融合进 ORB-SLAM3 中

# 相关依赖
* OpenCV(>=4.2)
* glog (apt install libgoogle-glog-dev)
* TensorRT (8.6.1.6)


## 如何使用你自己训练的模型

## 备注
暂时未为对 ROS 版本提供支持，未来将提供支持

## Test in Datasets
You can use [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) or [BONN](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) datasets to test this system

### TUM 数据集

```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml ~/dataset/rgbd_dataset_freiburg2_desk ~/dataset/rgbd_dataset_freiburg2_desk/associate.txt
```


## Docker 镜像