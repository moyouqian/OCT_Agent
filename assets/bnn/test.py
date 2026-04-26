import torch
import os
from scipy.io import loadmat
import numpy as np
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as Dataset
from bunetPP import UNetPlusPlus
from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import h5py
import sys


def crop_to_divisible_by_32(Un_Phase):
    """
    将输入数组裁剪成行和列都能被32整除的最大尺寸。
    
    参数:
        Un_Phase (numpy.ndarray): 输入的二维矩阵
        save_path (str): 保存路径
        name (str): 文件名
    
    返回:
        cropped_phase (numpy.ndarray): 裁剪后的二维矩阵，行和列均能被32整除
    """
    # 计算目标尺寸，使得Un_Phase的行和列都能被32整除
    target_rows = Un_Phase.shape[0] - (Un_Phase.shape[0] % 32)
    target_cols = Un_Phase.shape[1] - (Un_Phase.shape[1] % 32)
    
    # 计算需要裁剪的行和列数
    rows_to_crop = Un_Phase.shape[0] - target_rows
    cols_to_crop = Un_Phase.shape[1] - target_cols
    
    # 计算每一边需要裁剪的行和列数
    top_crop = rows_to_crop // 2
    bottom_crop = rows_to_crop - top_crop
    left_crop = cols_to_crop // 2
    right_crop = cols_to_crop - left_crop
    
    # 裁剪矩阵，确保索引不超出范围
    if bottom_crop == 0:
        cropped_phase = Un_Phase[top_crop:, :]
    else:
        cropped_phase = Un_Phase[top_crop:-bottom_crop, :]
    
    if right_crop == 0:
        cropped_phase = cropped_phase[:, left_crop:]
    else:
        cropped_phase = cropped_phase[:, left_crop:-right_crop]
    
    # # 确保保存路径存在
    # os.makedirs(save_path, exist_ok=True)
    
    # # 保存文件
    # filename = os.path.join(save_path, name)
    # savemat(filename, {'cropped_phase': cropped_phase})
    # print(f'Image saved to {filename}')
    
    return cropped_phase

def test_net(net,device,dir_path='Phase260',MC_test=50):

    image_path = "D:/项目/cgan/跑完的代码/wrapped_phase/"+dir_path
    image1 = os.listdir(image_path)
    image1 = sorted(image1)
    # 测试模式
    net.eval()
    t=0
    uncertainty_data=[]
    #遍历所有的图片
    for m in range(len(image1)):
    
        t = t + 1
        means = []
        log_vars = []
        
        path = image_path + '/' + image1[m]
        
        mat_data = loadmat(path)
        # 去掉 MATLAB 文件中的元数据字段（如 '__header__', '__version__', '__globals__'）
        keys = [key for key in mat_data.keys() if not key.startswith('__')]
        # 假设 .mat 文件中只有一个变量，获取这个变量名并提取数据
        if len(keys) == 1:
            wrapped_data = mat_data[keys[0]]
        else:
            raise Exception("MAT 文件中包含多个变量，无法确定要提取的变量。")

        wrapped_data = crop_to_divisible_by_32(wrapped_data)

        image = np.transpose(wrapped_data)
        image = torch.from_numpy(image.reshape(1,1,image.shape[0],image.shape[1]))
        image = image.to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(MC_test):
                mean, log_var, _ = net(image)
                means.append(mean)
                log_vars.append(log_var)
        means = torch.stack([m for m in means])
        log_vars = torch.stack([t for t in log_vars])
        # 预测
        
        predicts = torch.mean(means, 0).squeeze()
        predicts = predicts.cpu().detach().numpy()
        predicts = np.array(predicts)
        strain = np.transpose(predicts)


        
        if os.path.exists("./pred/"+dir_path) :
             results_dir      =  './pred/'  + dir_path + '/'
        else:
             os.makedirs("./pred/"+dir_path)
             results_dir      =  './pred/'  + dir_path + '/'

        
        # save_res_path1 =  results_dir + "strain" + (str)(t) + '.mat'
        save_res_path1 =  results_dir + image1[m]
        savemat(save_res_path1, {'strain':strain})

        # epistemic_uncertainty = torch.var(means,0).squeeze()
        # epistemic_uncertainty = epistemic_uncertainty.cpu().detach().numpy()** 0.5
        # epistemic_uncertainty = np.array(epistemic_uncertainty)
        # epistemic_uncertainty = np.transpose(epistemic_uncertainty)
        
        # save_res_path2 =  results_dir + "epistemic_uncertainty" + (str)(t) + '.mat'
        # savemat(save_res_path2, {'epistemic_uncertainty':epistemic_uncertainty})
        
        #数据不确定性
        
        # aleatoric_uncertainty = torch.mean(stds**2, 0).squeeze()
        # aleatoric_uncertainty = aleatoric_uncertainty.cpu().detach().numpy()** 0.5
        # aleatoric_uncertainty = np.array(aleatoric_uncertainty)
        
        # save_res_path1 = "./pred{}/".format(num) + "predict" + (str)(t) + '.mat'
        # save_res_path2 = "./pred{}/".format(num) + "model uncertainty" + (str)(t) + '.mat'
        # #save_res_path3 = "./pred{}/".format(num) + "data uncertainty" + (str)(t) + '.mat'
        # savemat(save_res_path1, {'predicts':predicts})
        # savemat(save_res_path2, {'epistemic_uncertainty':epistemic_uncertainty})
        #savemat(save_res_path3, {'aleatoric_uncertainty':aleatoric_uncertainty})
    #     uncertainty_data.append(epistemic_uncertainty)
    # uncertainty_data=np.array(uncertainty_data)
    # savemat("/home/l/BBBunet3/dataset/uncertainty.mat", {'data': uncertainty_data})
        # out1 = plt.imshow(predicts, cmap='jet', interpolation='nearest')
        # plt.gcf().set_size_inches(256 / 100, 256 / 100)
        # plt.axis('off')
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.colorbar(out1)
        # savefig(save_res_path1)
        # out2 = plt.imshow(epistemic_uncertainty, cmap='jet', interpolation='nearest')
        # savefig(save_res_path2)
        # out2 = plt.imshow(aleatoric_uncertainty, cmap='jet', interpolation='nearest')
        # savefig(save_res_path3)
        print("图" + (str)(t) + "预测完成")
    #     for i in range(batch_size):
    #         t = t + 1
    #         # 保存结果地址
    #         save_res_path1 = "/home/l/bnn1/dataset/pred/" + "predict" + (str)(t) + '.png'
    #         save_res_path2 = "/home/l/bnn1/dataset/pred1/" + "model uncertainty" + (str)(t) + '.png'
    #         save_res_path3 = "/home/l/bnn1/dataset/pred2/" + "data uncertainty" + (str)(t) + '.png'
    #         predicts[i] = predicts[i].cpu().detach().numpy()
    #         epistemic_uncertainty[i] = epistemic_uncertainty[i].cpu().detach().numpy()
    #         aleatoric_uncertainty[i] = aleatoric_uncertainty[i].cpu().detach().numpy()
    #         out1 = plt.imshow(predicts[i].squeeze(), cmap='jet', interpolation='nearest')
    #         savefig(save_res_path1)
    #         out2 = plt.imshow(epistemic_uncertainty[i].squeeze(), cmap='jet', interpolation='nearest')
    #         savefig(save_res_path2)
    #         out2 = plt.imshow(aleatoric_uncertainty[i].squeeze(), cmap='jet', interpolation='nearest')
    #         savefig(save_res_path3)
    #         print("图" + (str)(t) + "预测完成")



if __name__ == '__main__':

    args = sys.argv[1:]


    #num = 112
    # 选择设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = UNetPlusPlus()
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('model.pth', map_location=device))
    # net.load_state_dict(torch.load('/home/l/BBBunet3/data/dataset/best_model.pth', map_location=device))
    # data_path = "/home/l/bnn1/dataset/test/image/"
    test_net(net,device,args[0])


