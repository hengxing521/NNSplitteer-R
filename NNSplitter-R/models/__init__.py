# models/__init__.py
from .alexnet import alexnet
from .resnet import resnet18, resnet50
from .vgg import vgg16_bn

def get_model(model_name, num_classes=10):
    """
    统一模型分发工厂 (Factory Pattern)
    根据传入的模型名称字符串，动态实例化并返回网络拓扑
    """
    model_name = model_name.lower().strip()
    
    if model_name == 'alexnet':
        model = alexnet(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif model_name == 'vgg16' or model_name == 'vgg16_bn':
        model = vgg16_bn(num_classes=num_classes)
    else:
        raise ValueError(f"[错误] 不支持的模型架构: {model_name}。目前仅支持 alexnet, resnet18, resnet50, vgg16_bn。")
        
    return model