# datasets/cifar.py
import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """加载 CIFAR-10 数据集 (10分类, 32x32分辨率)"""
    # 使用 CIFAR-10 官方统计的全局像素均值和方差
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader


def get_cifar100_dataloaders(batch_size=128, num_workers=4, data_dir='./data'):
    """加载 CIFAR-100 数据集 (100分类, 32x32分辨率)"""
    # 使用 CIFAR-100 官方统计的全局像素均值和方差
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader