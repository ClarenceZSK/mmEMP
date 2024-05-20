# mmEMP
This is the repository of mmEMP, which has been published on ICRA 2024:

**Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision**

Cong Fan, Shengkai Zhang, Kezhong Liu, Shuai Wang, Zheng Yang, Wei Wang
## Prerequisites
* Install Python3.7 with Anaconda3
  ```
  conda create -n $ENV_NAME$ python=3.7
  source activate $ENV_NAME$
  ```
* Install [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive)
* Install [cuDNN8.0](https://developer.nvidia.com/cudnn)
* Install PyTorch1.7.0
  ```
  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
  ```
* Clone the repository to local
  ```
  https://github.com/ClarenceZSK/mmEMP.git
  ```
* Install PointNet++ library for basic point cloud operation
  ```
  cd mmEMP/radar_pose/lib
  python setup.py install
## Dataset
Download the [Dataset](https://pan.baidu.com/s/1KYOStoLnHUi-qyTsGuO3XQ?pwd=52jk). The dataset comprises raw data from millimeter-wave radar, vision, and IMU, along with the output results of PC-side automatic VI-SLAM execution. Please make sure you have prepared the dataset by organizing the directory as: `data/dataset/your_dataset`. In the end, the dataset should be oragnized like this:
  ```
  Dataset
    ├── radar
    │   │── xyz
    │   │── doppler_reshape_256_64_192
    ├── rgb
    │   │── infra1out
    ├── pose
    │   │── vinsout
    │   │── vinspointcloudout
  ```
## Getting started
### 1. Temporal alignment
  Please align the data based on the timestamps provided by ROS using the following command.
  ```
  cd mmEMP/data/dataset/your_dataset
  python temporal_alignment.py <rgb_folder_path> <radar_folder_path> <pose_folder_path>
  ```
  The aligned data will be saved in the `mmEMP/data/dataset/synced_data` path.
### 2. Dynamic visual-inertial 3D reconstruction
  ```
  python dynamic_points.py --image_dir /mmEMP/data/dataset/synced_data/rgb --radar_dir /mmEMP/data/dataset/synced_data/radar
  ```
  This is the implementation of dynamic 3D reconstruction algorithm, corresponding to our first design. Please using the synced data as input, and after running, the above command will generate label in `mmEMP/label` path, please place doppler_reshape_256_64_192 (in dataset) and the resulting label, respectively, under the paths `data/your_dataset/data` and `data/your_dataset/label`, e.g., `data/GTVINS2/data` and `data/GTVINS2/label`.
### 3. Point cloud generation
* Train the point cloud enhancement model by
  ```
  python main.py train
  ```
  In `config/base_confige.yml`, you might want to change the following settings: `data` root path of the dataset for training or testing, `batch_size` for traning
* Obtain densified point clouds
  ```
  python main.py eval
  ```
  The obtained dense point clouds will be saved in the `mmEMP/result` path.
### 4. Point cloud refinement
* Move the densified point cloud and synchronized rgb and pose data to the `mmEMP/radar_pose/preprocess` path, to serve as the new data for subsequent processing.
  ```
  cp mmEMP/result/* mmEMP/radar_pose/preprocess
  cp mmEMP/data/dataset/synced_data/{rgb,pose} mmEMP/radar_pose/preprocess/
  cd mmEMP/radar_pose
  python preprocess/preprocess_vod.py --root_dir /mmEMP/radar_pose/preprocess --save_dir $SAVE_DIR$
  ```
  The final scene flow samples will be saved under the `$SAVE_DIR$/flow_smp/`. 
* Train the point cloud pose estimation model by
  ```
  cd mmEMP/radar_pose
  python main.py --dataset_path $SAVE_DIR$ --exp_name $EXP_NAME$ --model cmflow
  ```
  where `$EXP_NAME$` can be defined according to personal preferences.
* Estimate the pose between adjacent radar frames by extracting point cloud features and constructing motion probability maps
  ```
  cd mmEMP/radar_pose
  python main.py --eval --dataset_path $SAVE_DIR$ --exp_name $EXP_NAME$ --model cmflow
  ```
* Eliminate ghost points
  ```
  python ghost_eliminate.py
  ```
  After running, the refined point cloud coordinates will be saved in the `mmEMP/enhanced_points.txt`.
## Citation
If you find our work useful in your research, please consider citing:
  ```
@InProceedings{Fan_2024_ICRA,
    author    = {Fan, Cong and Zhang, Shengkai and Liu, Kezhong and Wang, Shuai and Yang, Zheng and Wang, Wei},
    title     = {Enhancing mmWave Radar Point Cloud via Visual-inertial Supervision},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year      = {2024},
}
  ```
## Acknowledgments
The code is heavily borrowed from the following repos:
* [RPDNet](https://github.com/thucyw/RPDNet)
* [CMFlow](https://github.com/Toytiny/CMFlow)
