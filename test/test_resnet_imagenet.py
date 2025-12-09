import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys 
from tqdm import tqdm
sys.path.append("../build/Release/")
import mytorch
import time
from tiny_imagenet_torch import TinyImageNet

writer = SummaryWriter()



transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.48024562, 0.4480722, 0.3975478), (0.27201378, 0.26554194, 0.27431726)),


])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48024562, 0.4480722, 0.3975478), (0.27201378, 0.26554194, 0.27431726)),
])

batch_size = 64
trainset = TinyImageNet(root="./data" , train=True , download=True , transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , 
                                          num_workers=4 , pin_memory=True , persistent_workers=True , prefetch_factor=4)
testset = TinyImageNet(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=batch_size , shuffle=False , 
                                          num_workers=4 , pin_memory=True , persistent_workers=True , prefetch_factor=4)


def mytensor_from_tensor(tensor):
    return mytorch.from_dlpack_deepcopy(torch.utils.dlpack.to_dlpack(tensor))


if __name__ == '__main__':
    mytorch.set_default_device(mytorch.Cuda)
    test_images = []
    test_labels = []
    for images, labels in testloader:
        images = mytensor_from_tensor(images)
        labels = labels.numpy()
        test_images.append(images)
        test_labels.append(labels)

    resnet = mytorch.nn.ResNet18(num_classes=200 , h=64 , w=64)

    criterion = mytorch.nn.CrossEntropy()
    optimizer = mytorch.optim.SGD(resnet.parameters() , lr = 0.1  , weight_decay=5e-4 , momentum=0.9)
    epochs = 100
    scheduler = mytorch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs , eta_min=1e-3)




    try:
        for epoch in range(epochs):
            running_loss = 0.0
            resnet.train()

            for inputs, labels in tqdm(trainloader,
                                total=len(trainloader),
                                desc=f"Epoch {epoch+1} - Training"):
                inputs = mytensor_from_tensor(inputs)
                labels = mytensor_from_tensor(labels.float())
                

                optimizer.zero_grad()
                inputs.set_requires_grad(True)
                outputs = resnet(inputs)
                loss = criterion(outputs , labels)
                loss.backward()
                optimizer.step()
                running_loss+= loss.item()
            scheduler.step()
            avg_loss = running_loss / len(trainloader)
            print(f'[{epoch + 1}] loss: {avg_loss:.3f}')
            writer.add_scalar("Loss/train of ResNet18 with SGD" , avg_loss , epoch)

            resnet.eval()
            total_correct = 0
            for inputs, labels in zip(test_images, test_labels):
                outputs = resnet(inputs)
                predictions = np.argmax(outputs.numpy(), 1)
                total_correct += np.sum(predictions == labels)
            accuracy = total_correct / len(testset)
            print(f'Accuracy: {accuracy:.4f}')
            writer.add_scalar("Accuracy/test of ResNet18 with SGD" , accuracy , epoch)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


    print("Finished Training")

    
