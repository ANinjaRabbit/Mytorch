import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys 
sys.path.append("../build/Release/")
import mytorch



transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 32
trainset = torchvision.datasets.CIFAR10(root="./data" , train=True , download=True , transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=batch_size , shuffle=True , num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



if __name__ == '__main__':
    mytorch.set_default_device(mytorch.Cuda)

    lenet = mytorch.nn.Sequential([
        mytorch.nn.Conv2d(3, 6, 5),
        mytorch.nn.BatchNorm2d(6),
        mytorch.nn.ReLU(),
        mytorch.nn.MaxPool2d(2),
        mytorch.nn.Conv2d(6, 16, 5),
        mytorch.nn.BatchNorm2d(16),
        mytorch.nn.ReLU(),
        mytorch.nn.MaxPool2d(2),
        mytorch.nn.Flatten(start_dim = 1),
        mytorch.nn.Linear(5 * 5 * 16, 120),
        mytorch.nn.ReLU(),
        mytorch.nn.Linear(120, 84),
        mytorch.nn.ReLU(),
        mytorch.nn.Linear(84, 10),
    ])


    criterion = mytorch.nn.CrossEntropy()
    optimizer = mytorch.optim.AdamW(lenet.parameters() , lr = 0.001  , weight_decay=0.00001)

    epochs = 100
    scheduler = mytorch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs , eta_min=0.0001)





    for epoch in range(epochs):
        running_loss = 0.0
        tensor_batches = []
        label_batches = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = mytorch.tensor_from_numpy(inputs)
            labels = mytorch.tensor_from_numpy(labels)
            tensor_batches.append(inputs)
            label_batches.append(labels)

        for i,data in enumerate(zip(tensor_batches , label_batches) , 0):
            inputs , labels = data
            optimizer.zero_grad()
            inputs.set_requires_grad(True)
            outputs = lenet(inputs)
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()

        scheduler.step()
        avg_loss = running_loss / len(tensor_batches)
        print(f'[{epoch + 1}] loss: {avg_loss:.3f}')
        if(avg_loss < 0.5):
            break

    
    print("Finished Training")

    lenet.eval()


    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = mytorch.tensor_from_numpy(images)
            labels = labels.numpy()
            outputs = lenet(images)
            outputs = mytorch.numpy_from_tensor(outputs)
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