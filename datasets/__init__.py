# datasets/__init__.py
from .cifar import get_cifar10_dataloaders, get_cifar100_dataloaders
from .stl10 import get_stl10_dataloaders

def get_dataloader(dataset_name, batch_size=128, num_workers=4, data_dir='./data'):
    """
    统一的数据集分发路由 (Factory Pattern)
    根据传入的数据集名称，动态返回对应的 trainloader, testloader 和 类别数
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == 'cifar10':
        trainloader, testloader = get_cifar10_dataloaders(batch_size, num_workers, data_dir)
        num_classes = 10
    elif dataset_name == 'cifar100':
        trainloader, testloader = get_cifar100_dataloaders(batch_size, num_workers, data_dir)
        num_classes = 100
    elif dataset_name == 'stl10':
        trainloader, testloader = get_stl10_dataloaders(batch_size, num_workers, data_dir)
        num_classes = 10
    else:
        raise ValueError(f"[错误] 不支持的数据集: {dataset_name}。目前仅支持 cifar10, cifar100, stl10。")
        
    return trainloader, testloader, num_classes