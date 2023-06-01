import torch
from torch.utils.data import DataLoader, TensorDataset
from Dataset import features, labels

def load_dataset():
    # 使用预训练数据集（如果有的话）训练全局模型
    pretrain_data = features  # 加载预训练数据集,详情见Dataset.py
    pretrain_targets = labels.values  # 加载预训练目标（标签）

    # 处理数据以适应您的模型
    #这边可能出现问题
    pretrain_dataset = TensorDataset(torch.tensor(pretrain_data.values.reshape(-1, 54, 1), dtype=torch.float32),
                                     torch.tensor(pretrain_targets, dtype=torch.float32))
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)

    return pretrain_loader


def load_local_dataset():
    local_data = features  # 加载预训练数据集,详情见Dataset.py
    local_targets = labels.values  # 加载预训练目标（标签）
    local_dataset = TensorDataset(torch.tensor(local_data.values.reshape(-1, 54, 1), dtype=torch.float32),
                                     torch.tensor(local_targets, dtype=torch.float32))
    local_loader = DataLoader(local_dataset, batch_size=64, shuffle=True)
    '''
pretrain_dataset = TensorDataset(torch.tensor(pretrain_data.reshape(-1, 54, 1), dtype=torch.float32),
                                     torch.tensor(pretrain_targets, dtype=torch.float32))
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True)
    '''

    return local_loader