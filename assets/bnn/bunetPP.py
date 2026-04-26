
import torch
from torch import nn
import torch.nn.functional as F


class ConcreteDropout(nn.Module):
    def __init__(self, dropout=True, concrete=True, p_fix=0.01, weight_regularizer=1e-7,
                 dropout_regularizer=1e-6, conv="lin", Bayes=True):
        """

        :param dropout:在确定性模型的情况下，如果“真”，则应用 dropout，否则没有 dropout
        :param concrete:当“False”时，dropout参数是固定的。 如果“真”，则concrete dropout
        :param p_fix:在 not self.concrete 的情况下使用的 dropout 参数
        :param weight_regularizer ELBO 中权重正则化的参数
        :param dropout_regularizer: ELBO 中的 dropout 正则化参数
        :param conv:"lin" for dense layers, "1D" or "2D" for 1D or 2D convolutional layers
        :param Bayes:BNN 如果“真”，确定性模型如果“假”
        """
        super(ConcreteDropout, self).__init__()
        self.dropout = dropout
        self.concrete = concrete
        self.p_fix = p_fix
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.conv = conv
        self.Bayes = Bayes

        self.p_logit = nn.Parameter(torch.FloatTensor([0]))

    def forward(self, x, layer, stop_dropout=False):
        """

        :param x:dropout层输入
        :param layer:调用层
        :param stop_dropout:if "True" 防止在确定性模型的推理过程中丢失
        :return:
        out:输出
        regularization：对应的 KL 项
        """
        x=x.cuda()
        if self.concrete:
            p = torch.sigmoid(self.p_logit)
        else:
            p = torch.tensor(self.p_fix).cuda()
        if (self.dropout and not stop_dropout) or self.Bayes:
            out = layer(self._concrete_dropout(x, p, self.concrete))
        else:
            out = layer(x)

        sum_of_square = 0
        #求出长度
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        regularization, weights_regularizer, dropout_regularizer = 0, 0, 0
        if self.Bayes:
            weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
            if self.concrete:
                dropout_regularizer = p * torch.log(p)
                dropout_regularizer += (1. - p) * torch.log(1. - p)
                if self.conv == "lin":
                    input_dimensionality = x[0].numel()
                elif self.conv == "1D":
                    input_dimensionality = list(x.size())[1]
                else:
                    input_dimensionality = list(x.size())[1]
                dropout_regularizer *= self.dropout_regularizer * input_dimensionality
            regularization = weights_regularizer + dropout_regularizer  # KL(q(W)|p(W))) eq. 3 in concrete dropout paper

        return out, regularization

    def _concrete_dropout(self, x, p, concrete):
        """

        :param x: 输入
        :param p: dropout参数
        :param concrete:当“False”时，dropout 参数是固定的。 如果“真”，则concrete dropout
        :return:应用dropout的输入
        """
        if not concrete:
            if self.conv == "lin":
                drop_prob = torch.bernoulli(torch.ones(x.shape).cuda() * p)
            elif self.conv == "1D":
                drop_prob = torch.bernoulli(torch.ones(list(x.size())[0], list(x.size())[1], 1).cuda() * p)
                drop_prob = drop_prob.repeat(1, 1, list(x.size())[2])
            else:
                drop_prob = torch.bernoulli(torch.ones(list(x.size())[0], list(x.size())[1], 1, 1).cuda() * p)
                drop_prob = drop_prob.repeat(1, 1, list(x.size())[2], list(x.size())[3])

        else:
            eps = 1e-7  # to avoid torch.log(0)
            temp = 0.4  # temperature

            if self.conv == "lin":
                unif_noise = torch.rand_like(x)
            elif self.conv == "1D":
                unif_noise = torch.rand(list(x.size())[0], list(x.size())[1], 1).cuda()
                unif_noise = unif_noise.repeat(1, 1, list(x.size())[2])
            else:
                unif_noise = torch.rand(list(x.size())[0], list(x.size())[1], 1, 1).cuda()
                unif_noise = unif_noise.repeat(1, 1, list(x.size())[2], list(x.size())[3])

            drop_prob = (torch.log(p + eps)- torch.log(1 - p + eps)+ torch.log(unif_noise + eps)-torch.log(1 - unif_noise + eps))

            drop_prob = torch.sigmoid(drop_prob / temp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        # print(x.is_cuda, random_tensor.is_cuda)
        x = torch.mul(x, random_tensor)

        x /= retain_prob

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, weight_regularizer=1e-7,dropout_regularizer=1e-6,
                 dropout=True,concrete=True,p_fix=0.01,Bayes=True):
        super(ConvBlock, self).__init__()
        self.conv = "2D"
        self.double_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True))
        self.double_conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True))
        self.double_conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                            nn.BatchNorm2d(out_channels)
                            )
        self.relu = nn.ReLU(inplace=True)
        self.resconv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.conc_drop1 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes)
        self.conc_drop2 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes)
        self.conc_drop3 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes)


    def forward(self, x):
        residual = self.resconv(x)
        regularization = torch.empty(3, device=x.device)
        x1, regularization[0] = self.conc_drop1(x, self.double_conv1)
        x1, regularization[1] = self.conc_drop2(x1, self.double_conv2)
        x1, regularization[2] = self.conc_drop3(x1, self.double_conv3)
        x1 += residual
        x1 = self.relu(x1)

        return x1, regularization.sum()
     

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1,weight_regularizer=1e-7,dropout_regularizer=1e-6,
                 dropout=True,concrete=True,p_fix=0.01,Bayes=True):
        super(UNetPlusPlus, self).__init__()
        self.conv = "2D"
        self.filters = [16, 32, 64, 128, 256, 512]

        self.c41 = ConvBlock(self.filters[4]*2,self.filters[4])

        self.c31 = ConvBlock(self.filters[3]*2,self.filters[3])
        self.c32 = ConvBlock(self.filters[3]*3,self.filters[3])

        self.c21 = ConvBlock(self.filters[2]*2,self.filters[2])
        self.c22 = ConvBlock(self.filters[2]*3,self.filters[2])
        self.c23 = ConvBlock(self.filters[2]*4,self.filters[2])

        self.c11 = ConvBlock(self.filters[1]*2,self.filters[1])
        self.c12 = ConvBlock(self.filters[1]*3,self.filters[1])
        self.c13 = ConvBlock(self.filters[1]*4,self.filters[1])
        self.c14 = ConvBlock(self.filters[1]*5,self.filters[1])

        self.c01 = ConvBlock(self.filters[0]*2,self.filters[0])
        self.c02 = ConvBlock(self.filters[0]*3,self.filters[0])
        self.c03 = ConvBlock(self.filters[0]*4,self.filters[0])
        self.c04 = ConvBlock(self.filters[0]*5,self.filters[0])
        self.c05 = ConvBlock(self.filters[0]*6,self.filters[0])

        self.c00 = ConvBlock(in_channels,self.filters[0])
        self.c10 = ConvBlock(self.filters[0],self.filters[1])
        self.c20 = ConvBlock(self.filters[1],self.filters[2])
        self.c30 = ConvBlock(self.filters[2],self.filters[3])
        self.c40 = ConvBlock(self.filters[3],self.filters[4])
        self.c50 = ConvBlock(self.filters[4],self.filters[5])

        self.pool = nn.MaxPool2d(2)

        self.up41 = nn.ConvTranspose2d(self.filters[5], self.filters[4], kernel_size=2, stride=2)

        self.up31 = nn.ConvTranspose2d(self.filters[4], self.filters[3], kernel_size=2, stride=2)
        self.up32 = nn.ConvTranspose2d(self.filters[4], self.filters[3], kernel_size=2, stride=2)

        self.up21 = nn.ConvTranspose2d(self.filters[3], self.filters[2], kernel_size=2, stride=2)
        self.up22 = nn.ConvTranspose2d(self.filters[3], self.filters[2], kernel_size=2, stride=2)
        self.up23 = nn.ConvTranspose2d(self.filters[3], self.filters[2], kernel_size=2, stride=2)

        self.up11 = nn.ConvTranspose2d(self.filters[2], self.filters[1], kernel_size=2, stride=2)
        self.up12 = nn.ConvTranspose2d(self.filters[2], self.filters[1], kernel_size=2, stride=2)
        self.up13 = nn.ConvTranspose2d(self.filters[2], self.filters[1], kernel_size=2, stride=2)
        self.up14 = nn.ConvTranspose2d(self.filters[2], self.filters[1], kernel_size=2, stride=2)

        self.up01 = nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2)
        self.up02 = nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2)
        self.up03 = nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2)
        self.up04 = nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2)
        self.up05 = nn.ConvTranspose2d(self.filters[1], self.filters[0], kernel_size=2, stride=2)


        # output
        self.conv_last_mean = nn.Conv2d(self.filters[0], 1, kernel_size=1)
        self.conv_last_logvar = nn.Conv2d(self.filters[0], 1, kernel_size=1)

        self.conv_last_drop1 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes)
        self.conv_last_drop2 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                               weight_regularizer=weight_regularizer,
                                               dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.Softplus()
        

    def forward(self, x):

        regularization = torch.empty(23, device=x.device)
        x00,regularization[0] = self.c00(x)
        x10,regularization[1] = self.c10(self.pool(x00))
        x20,regularization[2] = self.c20(self.pool(x10))
        x30,regularization[3] = self.c30(self.pool(x20))
        x40,regularization[4] = self.c40(self.pool(x30))
        x50,regularization[5] = self.c50(self.pool(x40))

        x01 = self.up01(x10)
        # x01 = padding(x01,x00)
        x01 = torch.cat([x01, x00], dim=1)
        x01,regularization[6] = self.c01(x01)

        x11 = self.up11(x20)
        # x11 = padding(x11,x10)
        x11 = torch.cat([x11, x10], dim=1)
        x11,regularization[7] = self.c11(x11)

        x21 = self.up21(x30)
        # x21 = padding(x21,x20)
        x21 = torch.cat([x21, x20], dim=1)
        x21,regularization[8] = self.c21(x21)

        x31 = self.up31(x40)


        # diffy = x30.size()[2] - x31.size()[2]
        # diffx = x30.size()[3] - x31.size()[3]
        # # 利用差值，进行补充
        # x31 = F.pad(x31, [diffx // 2, diffx - diffx // 2,
        #                 diffy // 2, diffy - diffy // 2])
        x31 = torch.cat([x31, x30], dim=1)

        x31,regularization[9] = self.c31(x31)

        x41 = self.up41(x50)
        # diffy = x40.size()[2] - x41.size()[2]
        # diffx = x40.size()[3] - x41.size()[3]
        # # 利用差值，进行补充
        # x41 = F.pad(x41, [diffx // 2, diffx - diffx // 2,
        #                 diffy // 2, diffy - diffy // 2])
        x41 = torch.cat([x41, x40], dim=1)
        x41,regularization[10] = self.c41(x41)

        x32 = self.up32(x41)
        # x32 = padding(x32,x30)
        x32 = torch.cat([x32, x30, x31], dim=1)
        x32,regularization[11] = self.c32(x32)

        x22 = self.up22(x31)
        # x22 = padding(x22,x20)
        x22 = torch.cat([x22, x20, x21], dim=1)
        x22,regularization[12] = self.c22(x22)

        x12 = self.up12(x21)
        # x12 = padding(x12,x10)
        x12 = torch.cat([x12, x10, x11], dim=1)
        x12,regularization[13] = self.c12(x12)

        x02 = self.up02(x11)
        # x02 = padding(x02,x00)
        x02 = torch.cat([x02, x00, x01], dim=1)
        x02,regularization[14] = self.c02(x02)

        x03 = self.up03(x12)
        # x03 = padding(x03,x00)
        x03 = torch.cat([x03, x00, x01, x02], dim=1)
        x03,regularization[15] = self.c03(x03)

        x13 = self.up13(x22)
        # x13 = padding(x13,x10)
        x13 = torch.cat([x13, x10, x11, x12], dim=1)
        x13,regularization[16] = self.c13(x13)

        x23 = self.up23(x32)
        # print(x23.shape)
        # print(x20.shape)
        # print(x21.shape)
        # print(x22.shape)
        # x23 = padding(x23,x20)
        x23 = torch.cat([x23, x20, x21, x22], dim=1)
        x23,regularization[17] = self.c23(x23)

        x14 = self.up14(x23)
        # x14 = padding(x14,x10)
        x14 = torch.cat([x14, x10, x11, x12, x13], dim=1)
        x14,regularization[18] = self.c14(x14)

        x04 = self.up04(x13)
        # x04 = padding(x04,x00)
        x04 = torch.cat([x04, x00, x01, x02, x03], dim=1)
        x04,regularization[19] = self.c04(x04)

        x05 = self.up05(x14)
        # x05 = padding(x05,x00)
        x05 = torch.cat([x05, x00, x01, x02, x03, x04], dim=1)
        x05,regularization[20] = self.c05(x05)

        mean,regularization[21] = self.conv_last_drop1(x05,nn.Sequential(self.conv_last_mean,self.act1))
        log_var,regularization[22] = self.conv_last_drop2(x05,nn.Sequential(self.conv_last_logvar,self.act2))
        
        return mean,log_var,regularization.sum()

# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model =  UNetPlusPlus(1,1)
#     model.to(device)
#     train = torch.randn((1, 1, 236, 236)).to(device)
#     train.to(device)
#     out,std ,regularization= model(train)
#     precision = torch.exp(-std)
#     a = torch.sum((0.5 * precision) * ((out - std)**2) + std / 2, 0)
#     b= torch.mean(a)
#     print(out.shape,std.shape,regularization.shape,a.shape,b.shape)