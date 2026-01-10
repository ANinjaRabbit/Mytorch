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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
mytorch.set_default_device(mytorch.Cuda)






transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48024562, 0.4480722, 0.3975478), (0.27201378, 0.26554194, 0.27431726)),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize((0.48024562, 0.4480722, 0.3975478), (0.27201378, 0.26554194, 0.27431726)),

])




testset = TinyImageNet(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=32 , shuffle=False , 
                                        )




def mytensor_from_tensor(tensor):
    return mytorch.from_dlpack_deepcopy(torch.utils.dlpack.to_dlpack(tensor))

# preload testdatas
test_images = []
test_labels = []
for images, labels in testloader:
    images = mytensor_from_tensor(images)
    labels = labels.numpy()
    test_images.append(images)
    test_labels.append(labels)

def train(modelname , model , trainloader , criterion , optimizer , scheduler , epochs , batch_size):
    try:
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()

            for inputs, labels in tqdm(trainloader,
                                total=len(trainloader),
                                desc=f"Epoch {epoch+1} - Training"):
                inputs = mytensor_from_tensor(inputs)
                labels = mytensor_from_tensor(labels.float())
                

                optimizer.zero_grad()
                inputs.set_requires_grad(True)
                outputs = model(inputs)
                loss = criterion(outputs , labels)
                loss.backward()
                optimizer.step()
                running_loss+= loss.item()
            scheduler.step()
            avg_loss = running_loss / len(trainloader)
            print(f'[{epoch + 1}] loss: {avg_loss:.3f}')
            writer.add_scalar(f"Loss/train of {modelname} on TinyImageNet" , avg_loss , epoch)

            model.eval()
            total_correct = 0
            for inputs, labels in zip(test_images, test_labels):
                outputs = model(inputs)
                predictions = np.argmax(outputs.numpy(), 1)
                total_correct += np.sum(predictions == labels)
            accuracy = total_correct / len(testset)
            print(f'Accuracy: {accuracy:.4f}')
            writer.add_scalar(f"Accuracy/test of {modelname} on TinyImageNet" , accuracy , epoch)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    print("Finished Training")
    
    model.save(f"models/{modelname}_tinyimagenet.pth")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train Models on TinyImageNet")
    parser.add_argument("--model" , type=str , default="resnet18" , help="Model name")
    parser.add_argument("--epochs" , type=int , default=100 , help="Number of epochs")
    args = parser.parse_args()
    modelname = args.model


    if modelname == "resnet18":
        model = mytorch.nn.ResNet18(num_classes=200 , h=64 , w=64)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr =0.001 , weight_decay=5e-4)
        batch_size = 40
    elif modelname == "resnet34":
        model = mytorch.nn.ResNet34(num_classes=200 , h=64 , w=64)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr =0.001 , weight_decay=5e-4)
        batch_size = 40
    elif modelname == "resnext18":
        model = mytorch.nn.ResNeXt18(num_classes=200 , h=64 , w=64)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr=0.001  , weight_decay=5e-4)
        batch_size = 40
    elif modelname == "resnext34":
        model = mytorch.nn.ResNeXt34(num_classes=200 , h=64 , w=64)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr =0.001 , weight_decay=5e-4)
        batch_size = 40
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    trainset = TinyImageNet(root="./data" , train=True , download=True , transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , 
                                            num_workers=4 , pin_memory=True , persistent_workers=True , prefetch_factor=4)


    criterion = mytorch.nn.CrossEntropy()

    epochs = args.epochs
    scheduler = mytorch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs , eta_min=1e-5)

    trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , 
                                            num_workers=4 ,  persistent_workers=True , prefetch_factor=4)

    train(modelname , model , trainloader , criterion , optimizer , scheduler , epochs , batch_size)
    




