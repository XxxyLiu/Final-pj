# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch
from torch.utils.tensorboard import SummaryWriter
from utils.utils_map import get_map

if __name__ == "__main__":
    writer = SummaryWriter('./log/')
    Cuda = True

    train_gpu = [0, ]

    fp16 = False

    classes_path = 'model_data/voc_classes.txt'

    # model_path = 'checkpoint/imagnet.pth'         # 加载imagnet预训练参数
    model_path = 'checkpoint/mask rcnn.pth'        # 加载mask rcnn预训练参数

    input_shape = [600, 600]

    backbone = "resnet50"

    pretrained = False

    anchors_size = [8, 16, 32]

    Init_Epoch = 0
    Epochs = 60
    batch_size = 4


    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0

    lr_decay_type = 'cos'

    save_period = 5

    save_dir = 'logs'

    num_workers = 4

    #   获得图片路径和标签
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    #   获取classes和anchor
    class_names, num_classes = get_classes(classes_path)

    #   设置用到的显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))

        #   根据预训练权重的Key和模型的Key进行加载
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    model_train = model.train()
    if Cuda:
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #   读取数据集对应的txt
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #   Adam优化器
        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),

        #   获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epochs)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)

        train_util = FasterRCNNTrainer(model_train, optimizer)

        #   开始模型训练
        for epoch in range(Init_Epoch, Epochs):

            #   判断当前batch_size，自适应调整学习率
            nbs = 16
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            #   获得学习率下降的公式
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epochs)

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            # 加载训练数据
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=frcnn_dataset_collate)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            train_loss, val_loss = fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                      Epochs, Cuda, fp16, None, save_period, save_dir)

            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('val loss', val_loss, epoch)

            # 测试，计算mAP指标、mIoU指标
            MINOVERLAP = 0.5
            map_out_path = 'map_out'

            if not os.path.exists(map_out_path):
                os.makedirs(map_out_path)
            if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
                os.makedirs(os.path.join(map_out_path, 'ground-truth'))
            if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
                os.makedirs(os.path.join(map_out_path, 'detection-results'))
            if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
                os.makedirs(os.path.join(map_out_path, 'images-optional'))

            mAP, mIoU = get_map(MINOVERLAP, True, path=map_out_path)

            # 画图
            writer.add_scalar('acc', mAP, epoch)
            writer.add_scalar('train loss', train_loss, epoch)
            writer.add_scalar('train loss', val_loss, epoch)

        loss_history.writer.close()
