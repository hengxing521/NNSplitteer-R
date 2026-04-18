# datasets/stl10.py
import torch
import torchvision
import torchvision.transforms as transforms

def get_stl10_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """加载 STL-10 数据集 (10分类, 96x96大分辨率)"""
    
    # STL-10 数据集的全局 RGB 均值和标准差
    stl10_mean = (0.4467, 0.4398, 0.4066)
    stl10_std = (0.2603, 0.2565, 0.2712)
    
    transform_train = transforms.Compose([
        # 对应 96x96 的图像，使用 12 像素的 padding 以保持平移扰动比例
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std),
    ])

    # 注意: STL-10 调用 torchvision 接口时使用 'split' 参数而不是 'train'
    trainset = torchvision.datasets.STL10(
        root=data_dir, split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.STL10(
        root=data_dir, split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader