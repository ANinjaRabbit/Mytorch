import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys 
from tqdm import tqdm
sys.path.append("../build/Release/")
import mytorch
import time



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root="./data" , train=True , download=True , transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , 
                                          num_workers=4 , pin_memory=True , persistent_workers=True , prefetch_factor=4)
testset = torchvision.datasets.CIFAR10(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=batch_size , shuffle=True , 
                                          num_workers=4 , pin_memory=True , persistent_workers=True , prefetch_factor=4)

# preload testdatas
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

    resnet = mytorch.nn.ResNet18Cifar(num_classes=10)

    criterion = mytorch.nn.CrossEntropy()
    optimizer = mytorch.optim.Adam(resnet.parameters() , lr = 1e-3  , weight_decay=0.00001)

    epochs = 100
    scheduler = mytorch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs , eta_min=3e-5)




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
            resnet.eval()
            total_correct = 0
            for inputs, labels in zip(test_images, test_labels):
                outputs = resnet(inputs)
                predictions = np.argmax(outputs.numpy(), 1)
                total_correct += np.sum(predictions == labels)
            accuracy = total_correct / len(testset)
            print(f'Accuracy: {accuracy:.4f}')
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


    print("Finished Training")

    
    resnet.eval()

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = mytorch.tensor_from_numpy(images)
            labels = labels.numpy()
            outputs = resnet(images)
            outputs = outputs.numpy()
            predictions = np.argmax(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    total = 0
    totalcorrect = 0

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        total += total_pred[classname]
        totalcorrect += correct_count
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print(f'Total Accuracy: {100 * totalcorrect / total} %')