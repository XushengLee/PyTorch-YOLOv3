# 红外数据YOLO模型

对采集的红外数据使用YOLOV3模型进行训练测试和部署。

## video推断

项目目录下 `config.py` 为配置文件，可使用记事本或sublime等工具打开，修改其中的参数进行调试。

```
# 以下说明部分可供调节的参数，其他一些参数留在之后的接口供开发人员使用
self.MODEL_DEF # yolo的具体配置文件 已跟据项目实际需求进行了设置 默认为yolov3-custom.cfg
self.WEIGHTS_PATH # 参数文件
self.CONF_THRES  # 该参数调整object conf达到多少时才会被认为是有效目标
self.NMS_THRES # 在NMS中使用的阈值，小于该阈值的bbox会被舍弃
self.VIDEO_PATH # 如果想要测试默认测试视频以外的video，需要修改此参数
self.SAVE_PATH # 输出视频的保存地址，保存文件名为out-(时间戳).mp4
```

## 红外数据预处理

由于设备的原因，采集到的数据格式上有损坏，需要先使用ffmpeg将视频进行重新生成，该脚本文件scripts/batch_video_ffmpeg.py可以批量处理。注：需要有一定简单基础的同学对输入输出路径进行简单的调整。

然后可以使用scripts/batch_video2img.py进行视频到图片的转换。scripts/mk_dataset.py进行数据的合并，scripts/split_set.py进行训练集和测试集的划分并生成yolo需要的txt数据集文件。

## requirements.txt

```
numpy
torch>=1.0
torchvision
matplotlib
tensorflow
tensorboard
terminaltables
pillow
tqdm
filterpy==1.4.5
numba==0.49.0
scikit-image
lap==0.4.0
```



## 模型训练

```bash
python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74
```

训练后checkpoint会保存在checkpoints/目录下，注意不要把正在使用的模型覆盖掉。

## 项目目录

```
assert/  # 放置效果展示图片
checkpoints/  # 训练结果
config/  # 配置生成脚本和yolo配置文件
config.py  # inference脚本使用的简易配置参数文件，可供使用者修改使用
data/  # 放置训练测试数据
detect.py  # 对应coco数据集的inference代码
examples/  # 放置测试数据
legacy/  # 早期代码
logs/  # 记录
models.py  # 模型文件
modified_sort.py  # sort tracking 根据项目需要跟踪具有多标签的人体的需求进行了修改
output/  # output video directory
requirements.txt  # 环境要求说明
sort.py  # sort tracking
test.py  # coco test代码
track_video.py  # 自然视频的tracking inference，可用examples/PETS09-S2L2.mp4查看效果
track_video_modified_sort.py  # 项目红外视频并带track的inference代码
train.py  # 训练代码
utils/  # 辅助函数等
video_detect.py  # 不带track的inference
webcam_detect.py # 摄像头inference
```
