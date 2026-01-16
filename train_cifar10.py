import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys 
import argparse

from tqdm import tqdm
sys.path.append("./build/Release/")
import mytorch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
mytorch.set_default_device(mytorch.Cuda)






transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
])


testset = torchvision.datasets.CIFAR10(root="./data" , train=False , download=True , transform=transform)
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
            writer.add_scalar(f"Loss/train of {modelname} on CIFAR10" , avg_loss , epoch)

            model.eval()
            total_correct = 0
            for inputs, labels in zip(test_images, test_labels):
                outputs = model(inputs)
                predictions = np.argmax(outputs.numpy(), 1)
                total_correct += np.sum(predictions == labels)
            accuracy = total_correct / len(testset)
            print(f'Accuracy: {accuracy:.4f}')
            writer.add_scalar(f"Accuracy/test of {modelname} on CIFAR10" , accuracy , epoch)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    print("Finished Training")
    
    model.save(f"models/{modelname}_cifar10.pth")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train Models on CIFAR10")
    parser.add_argument("--model" , type=str , default="resnet18" , help="Model name")
    parser.add_argument("--epochs" , type=int , default=100 , help="Number of epochs")
    args = parser.parse_args()

    modelname = args.model
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    if modelname == "resnet18":
        model = mytorch.nn.ResNet18(num_classes=10 , h=32 , w=32)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr =0.001 , weight_decay=1e-4)
        batch_size = 200
    elif modelname == "resnet34":
        model = mytorch.nn.ResNet34(num_classes=10 , h=32 , w=32)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr =0.001 , weight_decay=1e-4)
        batch_size = 40
    elif modelname == "resnext18":
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        model = mytorch.nn.ResNeXt18(num_classes=10 , h=32 , w=32)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr=0.001  , weight_decay=1e-4)
        batch_size = 200
    elif modelname == "resnext34":
        model = mytorch.nn.ResNeXt34(num_classes=10 , h=32 , w=32)
        optimizer = mytorch.optim.AdamW(model.parameters() , lr =0.001 , weight_decay=1e-4)
        batch_size = 40
    else:
        raise ValueError(f"Unknown model name: {modelname}")

    trainset = torchvision.datasets.CIFAR10(root="./data" , train=True , download=True , transform=transform_train)


    criterion = mytorch.nn.CrossEntropy()

    epochs = args.epochs
    scheduler = mytorch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs , eta_min=1e-6)

    trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , 
                                            num_workers=4 ,  persistent_workers=True , prefetch_factor=4)

    train(modelname , model , trainloader , criterion , optimizer , scheduler , epochs , batch_size)
    




