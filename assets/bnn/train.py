#6.2 本代码是将bunet++网络用于应变


###导入库

import os
from sys import stderr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from bunetPP import UNetPlusPlus
from glob import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.io import loadmat


###GPU使用

model = UNetPlusPlus()
#gpus = [3]
#gpus_str = ",".join([str(i) for i in gpus])
#默认CUDA设备是按照ID顺序来排序的，设置这个环境变量按照物理PCI总线上的顺序来选择GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
device = torch.device("cuda:{}".format(0))
model = model.to(device)
#if len(gpus) > 1:
    #model = nn.DataParallel(model, device_ids=gpus)

###超参数设置
IMG_CHANNELS = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 15
EPOCHS = 200
adam = optim.Adam(model.parameters(),lr=LEARNING_RATE)
#loss_fn = nn.L1Loss()
scheduler = ReduceLROnPlateau(adam, mode='min', factor=0.1, patience=4, min_lr=1e-9)


###数据地址
train_savepath = '/home/data_training'


###数据生成器

class DataGenerator(Dataset):
    def __init__(self, data_dir, path='CGAN', inputpath='train_input', labelpath='train_label',
                 inputname='SpecklePhase', labelname='Strain'):
        self.data_dir = data_dir
        self.dataname = inputname
        self.labelname = labelname

        # 获取文件名并排序
        self.input_path = self.data_dir+'/' + path + '/' + inputpath
        self.label_path = self.data_dir+'/' + path + '/' + labelpath
        self.input_filename = os.listdir(self.input_path)
        self.label_filename = os.listdir(self.label_path)
        self.input_filename.sort(key=lambda x: int(x.split('.')[0].split('pecklePhase')[1]))
        self.label_filename.sort(key=lambda x: int(x.split('.')[0].split('n')[1]))
        self.inputs_list = [self.input_path+'/'+filename for filename in self.input_filename]
        self.labels_list = [self.label_path+'/'+filename for filename in self.label_filename]

    def __len__(self):
        return len(self.inputs_list)

    def __getitem__(self, index):
        input_file = self.inputs_list[index]
        #print(input_file)
        label_file = self.labels_list[index]

        input_data = self.__data_generation_Input(input_file)
        label_data = self.__data_generation_Label(label_file)

        return input_data, label_data

    def __data_generation_Input(self, filename):

        data = loadmat(filename)
        mat = np.transpose(data['SpecklePhase'])
        return torch.unsqueeze(torch.from_numpy(mat), 0)
    
    def __data_generation_Label(self, filename):

        data = loadmat(filename)
        mat = np.transpose(data['Strain'])
        return torch.unsqueeze(torch.from_numpy(mat), 0)

# 使用 DataLoader 封装数据生成器
dataset = DataGenerator(data_dir=train_savepath)
train_generator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

plot_count = -2
def show_imgs(imgs, titles):
    global plot_count
    plot_count += 1
    num_imgs = len(imgs[0])
    num_titles = len(titles)
    for k in range(num_imgs):        
        plt.figure(figsize=(18, 6))
        for v in range(num_titles):
            plt.subplot(1, num_titles, v+1) 
            plt.title(titles[v])
            plt.imshow(imgs[v][k].squeeze(), cmap=plt.cm.jet)
            plt.axis('on')
            plt.colorbar(fraction=0.05)
        plt.savefig(f"test_img{plot_count}.png")
        #plt.show(block=False)
        plt.pause(1)  # 暂停30秒
        plt.close()  # 暂停结束后自动关闭图形

# 检查训练图像
train_generator1 = DataLoader(dataset, batch_size=1, shuffle=True)
# print(test_generator.__len__())
# 获取一个迭代器
train_iter = iter(train_generator1)
# 获取一个批次的数据
train_imgs = next(train_iter)
    
title = ['SpecklePhase', 'Strain']
show_imgs(train_imgs, title)

test_savedir = '/home/data_training/CGAN/'
test_savepath = ['train_input1', 'train_label1']
test_label = ['SpecklePhase', 'Strain']

def regress_loss(true,mean,log_var,metric):
    precision = torch.exp(-log_var)
    if metric == "mae":
        # s = torch.sum(torch.div(1,(2 * precision) ** 2)* torch.abs(true - mean) + log_var / 2, 0)
        # print(s.shape)
        return torch.mean(torch.sum((0.5 * precision) * ((true - mean)**2) + log_var / 2, 0))
    elif metric == "rmse" or not metric:  # default is rmse
        return torch.mean(torch.sum(precision * (true - mean) ** 2 + log_var,0))
    else:
        print("Metric has to be 'rmse' or 'mae'")



test_data = []
for i in range(2):
    exp_list = glob(test_savedir + test_savepath[i] + '/' + '*.mat')
    exp_list = [path.replace('\\', '/') for path in exp_list]
    print(exp_list)
    #exp_list.sort(key=lambda x: int(x.split('.')[0].split('pecklePhase')[1]))
    data = []
    mat1 = loadmat(exp_list[0])
    mat = np.transpose(mat1[test_label[i]])
    data.append(mat)
    # with h5py.File(exp_list[0], 'r') as mat:
    #     print(mat)

    #     mat = np.transpose(mat[test_label[i]])
    #     data.append(mat)
    data = np.array(data)
    data = np.expand_dims(data, 0)
    test_data.append(data)
test_imgs = np.array(test_data)

title = ['SpecklePhase', 'Strain']
show_imgs(test_imgs, title)

class PresentationCallback:
    def __init__(self, model, test_imgs):
        self.model = model
        self.test_imgs = test_imgs

    def on_epoch_end(self, epoch):
        #if (epoch + 1) % 20 == 0:
        #if epoch == 0:
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            predict_imgs,_ ,_ = self.model(torch.from_numpy(self.test_imgs[0]).float().to(device))
            predict_imgs = predict_imgs.cpu()
            predict_imgs = predict_imgs.numpy()  # 将张量转换为 NumPy 数组
        titles = ['Input', 'Predict', 'Ground Truth']
        imgs = np.stack((self.test_imgs[0], predict_imgs, self.test_imgs[1]))
        show_imgs(imgs, titles)

    def on_train_end(self):
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算
            predict_imgs = self.model(torch.from_numpy(self.test_imgs[0] ).float())
            predict_imgs = predict_imgs.cpu()
            predict_imgs = predict_imgs.numpy()  # 将张量转换为 NumPy 数组
        titles = ['Input', 'Predict', 'Ground Truth']
        imgs = np.stack((self.test_imgs[0], predict_imgs, self.test_imgs[1]))
        show_imgs(imgs, titles)


class ModelCheckpoint:
    def __init__(self, filepath, save_best_only=True, monitor='train_loss'):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value = float('inf') if 'loss' in monitor else float('-inf')
    
    def __call__(self, model, current_value):
        if (self.save_best_only and 
            ((self.monitor == 'train_loss' and current_value < self.best_value) or
            (self.monitor != 'train_loss' and current_value > self.best_value))):
            self.best_value = current_value
            torch.save(model.state_dict(), self.filepath)
            print(f"Model saved at {self.filepath}")

class EarlyStopping:
    def __init__(self, patience=10, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')

    def __call__(self, current_value):
        if self.mode == 'min':
            if current_value < self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_value > self.best_value:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def should_stop(self):
        return self.early_stop



early_stopping = EarlyStopping(patience=10, mode='min')

checkpoint = ModelCheckpoint(filepath='model.pth', save_best_only=True, monitor='train_loss')

cp_presentation = PresentationCallback(model,test_imgs)

iteration = 0

for epoch in range(EPOCHS):

    train_loss = 0
    
    for i, (X_batch, Y_batch) in enumerate(train_generator):

        X_batch, Y_batch = X_batch.to(device,dtype=torch.float32), Y_batch.to(device,dtype=torch.float32)

        mean, log_var, regulariztion = model(X_batch)
        loss = regress_loss(Y_batch, mean, log_var, metric="mae") + regulariztion
        adam.zero_grad()
        loss.backward()
        adam.step()
        train_loss += loss.item()

    
    cp_presentation.on_epoch_end(iteration)
    
    early_stopping(train_loss)
    iteration += 1


    if early_stopping.should_stop():
        print("Early stopping")
        break


    
    average_loss = train_loss/55000
    scheduler.step(average_loss)
    print("Epochs : {} \t Train loss : {:.4f} \t Average loss : {:.6f}".format(iteration, train_loss, average_loss))

    current_lr = adam.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr}")

    checkpoint(model, train_loss)

cp_presentation.on_train_end()
