import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys 
import argparse
from tiny_imagenet_torch import TinyImageNet

from tqdm import tqdm
sys.path.append("./build/Release/")
import mytorch
mytorch.set_default_device(mytorch.Cuda)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48024562, 0.4480722, 0.3975478), (0.27201378, 0.26554194, 0.27431726)),
])

batch_size = 32

testset = TinyImageNet(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=batch_size , shuffle=False  )




def mytensor_from_tensor(tensor):
    return mytorch.from_dlpack_deepcopy(torch.utils.dlpack.to_dlpack(tensor))

def test(model):
    model.eval()
    total_correct = 0
    for inputs, labels in tqdm(testloader):
        inputs = mytensor_from_tensor(inputs)
        labels = labels.numpy()
        outputs = model(inputs)
        predictions = np.argmax(outputs.numpy(), 1)
        total_correct += np.sum(predictions == labels)
    accuracy = total_correct / len(testset)
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test Models on TinyImageNet")
    parser.add_argument("--model" , type=str , default="resnet18" , help="Model name")
    args = parser.parse_args()

    modelname = args.model
    path = f"./models/{modelname}_tinyimagenet.pth"
    if modelname == "resnet18":
        model = mytorch.nn.ResNet18(num_classes=200 , h=64 , w=64)
    elif modelname == "resnet34":
        model = mytorch.nn.ResNet34(num_classes=200 , h=64 , w=64)
    elif modelname == "resnext18":
        model = mytorch.nn.ResNeXt18(num_classes=200 , h=64 , w=64)
    elif modelname == "resnext34":
        model = mytorch.nn.ResNeXt34(num_classes=200 , h=64 , w=64)
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    model.load(path)

    test(model)

    



