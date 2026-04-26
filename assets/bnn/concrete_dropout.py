import torch
from torch import nn


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
