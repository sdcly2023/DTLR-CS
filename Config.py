# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 2:44 下午
# @Author  : Haonan Wang
# @File    : Config.py
# @Software: PyCharm
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
use_cuda = torch.cuda.is_available()
seed = 999
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 384
real_size = 384
print_frequency = 1
save_frequency = 5000
vis_frequency = 100
early_stopping_patience = 60


# pretrain = False
task_name = 'BNS_enhance'  # GlaS MoNuSeg ISBI_pic BNS_enhance
# task_name = 'GlaS'
learning_rate = 1e-3
batch_size = 16

# model_name = 'UNet'
# model_name = 'UCTransNet'
model_name = 'UCTransNet_pretrain'

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')+'_'+str(img_size)+'_'+str(real_size)
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net 64
    config.n_classes = 1
    config.img_size = img_size
    config.real_size = real_size
    return config


# used in testing phase, copy the session name in training phase
test_session = "Test_session_01.07_00h31_448_1000_64"
