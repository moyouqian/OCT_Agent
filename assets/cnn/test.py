import torch
import os
import sys
from scipy.io import loadmat
import numpy as np
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as Dataset
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import h5py
from Unet import Unet


def test_net(net,device,dir_path='Phase260'):

    

    image_path = "D:/项目/cgan/跑完的代码/wrapped_phase/"+dir_path
    image1 = os.listdir(image_path)
    image1 = sorted(image1)
    # image1.sort(key=lambda x: int(x.split('.')[0].split('e')[1]))
    # 测试模式
    net.eval()
    t=0
    uncertainty_data=[]
    #遍历所有的图片
    for i in range(len(image1)):
        t = t + 1
        means = []
        log_vars = []
        path = image_path + '/' + image1[i]
    
        # 加载 .mat 文件
        mat_data = loadmat(path)
        # 去掉 MATLAB 文件中的元数据字段（如 '__header__', '__version__', '__globals__'）
        keys = [key for key in mat_data.keys() if not key.startswith('__')]
        # 假设 .mat 文件中只有一个变量，获取这个变量名并提取数据
        if len(keys) == 1:
            wrapped_data = mat_data[keys[0]]
        else:
            raise Exception("MAT 文件中包含多个变量，无法确定要提取的变量。")

        image = np.transpose(wrapped_data)
        image = torch.from_numpy(image.reshape(1,1,image.shape[0],image.shape[1]))
        image = image.to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            strain = net(image)
            
        
        strain = strain.squeeze()
        #print(strain.shape)
        strain = strain.cpu().detach().numpy()
        strain = np.array(strain)
        strain = np.transpose(strain)

        

        if os.path.exists("./pred/"+dir_path) :
             results_dir      =  './pred/'  + dir_path + '/'
        else:
             os.makedirs("./pred/"+dir_path)
             results_dir      =  './pred/'  + dir_path + '/'

        
        # save_res_path1 =  results_dir + "strain" + (str)(t) + '.mat'
        save_res_path1 =  results_dir + image1[i]
        savemat(save_res_path1, {'strain':strain})


        print("图" + (str)(t) + "预测完成")
    



if __name__ == '__main__':
    args = sys.argv[1:]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. 实例化模型结构
    net = Unet()  # 替换为你的模型类
    
    # 2. 加载参数（自动映射到当前设备）
    net.load_state_dict(
        torch.load('model.pth', map_location=device)
    )
    
    # 3. 将模型转移到设备
    net.to(device)
    
    test_net(net, device, args[0])

