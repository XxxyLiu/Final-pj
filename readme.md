1、将VOC数据集解压到faster rcnn项目的根路径下
2、训练：  使用命令： python train.py 训练模型， 训练好的模型保存在了model_data文件夹里,loss曲线以及评价指标保存在了log文件夹里
3、测试：  将要测试的图片放在img文件夹里，使用命令：python predict.py  检测完的图片保存在img_out文件里


网络结构：resnet50， batch size：4，  learning rate： 0.0001， 优化器：Adam，   epoch： 80
loss function：faster rcnn损失函数（包括回归损失、分类损失）
评价指标：mAP,mIoU