import torch
import torch.nn as nn
import torch.nn.functional as F


# 针对二分类任务的 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):

        device = target.device
        self.alpha = self.alpha.to(device)

        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).to(device)
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).to(device)
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
 
        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


#密度的导数作为权重

class GHMC(nn.Module):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def forward(self, pred, target):
        device = target.device
        pred = nn.Sigmoid()(pred).flatten()
        edges = torch.arange(self.bins + 1).float().to(device) / self.bins  #定义区间边界
        g = torch.abs(pred.detach() - target)  #输出预测难度
        tot = pred.size(0)        # 计算所有有效样本总数
        n = 0  # n valid bins
        # 通过循环计算落入10个bins的梯度模长数量
        weights = torch.zeros_like(pred).to(device)
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                # 重点，所谓的梯度密度就是1/num_in_bin
                weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        criterion = nn.BCELoss(weight=weights)
        loss = criterion(pred, target.float())
        return loss

# if __name__ == "__main__":
#     criterion = GHMC()
#     pred = torch.rand(64)
#     labels = torch.ones(64)
#     loss = criterion(pred, labels)
#     print(loss)