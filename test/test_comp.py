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




transform = transforms.Compose( [transforms.ToTensor() , transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])
batch_size = 32
trainset = torchvision.datasets.CIFAR10(root="./data" , train=True , download=True , transform=transform)
trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=batch_size , shuffle=True , num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
#        self.conv1 = nn.Conv2d( 3 , 6,5 )
#        self.pool1 = nn.MaxPool2d(2,2)
#        self.batchnorm1 = nn.BatchNorm2d(6 , momentum=0.1)
#        self.conv2 = nn.Conv2d(6 , 16 ,5)
#        self.pool2 = nn.MaxPool2d(2,2)
#        self.batchnorm2 = nn.BatchNorm2d(16 , momentum=0.1)
#        self.fc1 = nn.Linear(5 * 5 * 16 , 120)
#        self.fc2 = nn.Linear(120 , 84)
#        self.fc3 = nn.Linear(84 , 10)
        self.conv = nn.Conv2d(3 , 6 , 5)
        self.bn = nn.BatchNorm2d(6 , momentum=0.1)
        self.fc = nn.Linear(28 * 28 * 6 , 10)

    def forward(self , x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.flatten(x , 1)
        x = self.fc(x)
        return x
#        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
#        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
#        x = torch.flatten(x , 1)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x

if __name__ == '__main__':
    mytorch.set_default_device(mytorch.Cuda)
    
    lenet = mytorch.nn.Sequential([
        mytorch.nn.Conv2d(3 , 6 , 5)
        , mytorch.nn.BatchNorm2d(6 , momentum=0.1)
        , mytorch.nn.Flatten(start_dim = 1),
        mytorch.nn.Linear(28 * 28 * 6 , 10)
        ]
    )
    criterion = mytorch.nn.CrossEntropy()
    optimizer = mytorch.optim.Adam(lenet.parameters() , lr = 0.001 )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lenet_torch = LeNet().to(device)
    lenet_torch.conv.weight = torch.nn.Parameter(torch.from_numpy(mytorch.numpy_from_tensor(lenet.parameters()[0])))
    lenet_torch.conv.bias = torch.nn.Parameter(torch.from_numpy(mytorch.numpy_from_tensor(lenet.parameters()[1])))
    lenet_torch.bn.weight = torch.nn.Parameter(torch.from_numpy(mytorch.numpy_from_tensor(lenet.parameters()[2])))
    lenet_torch.bn.bias = torch.nn.Parameter(torch.from_numpy(mytorch.numpy_from_tensor(lenet.parameters()[3])))
    lenet_torch.fc.weight = torch.nn.Parameter(torch.from_numpy(mytorch.numpy_from_tensor(lenet.parameters()[4])))
    lenet_torch.fc.bias = torch.nn.Parameter(torch.from_numpy(mytorch.numpy_from_tensor(lenet.parameters()[5])))
    
    criterion_torch = nn.CrossEntropyLoss()
    optimizer_torch = optim.Adam(lenet_torch.parameters(), lr=0.001 )


    for epoch in range(1):
        running_loss = 0.0
        weight_avg_grad = None
        for i,data in enumerate(trainloader , 0):
            inputs , labels = data
            mytorch_inputs = mytorch.tensor_from_numpy(inputs.numpy())
            mytorch_inputs.set_requires_grad(True)
            mytorch_labels = mytorch.tensor_from_numpy(labels.numpy())
            torch_inputs = inputs.to(device)
            torch_labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = lenet(mytorch_inputs)
            loss = criterion(outputs , mytorch_labels)
            loss.backward()
            optimizer.step()

            optimizer_torch.zero_grad()
            outputs_torch = lenet_torch(torch_inputs)
            outputs_torch.requires_grad_(True)
            loss_torch = criterion_torch(outputs_torch, torch_labels)
            loss_torch.backward()
            optimizer_torch.step()

            print(f"loss mytorch {loss.item():.3f}")
            print(f"loss torch {loss_torch.item():.3f}")


        




        weight_avg_grad /= len(trainloader)
        print(weight_avg_grad.mean() , weight_avg_grad.std() , weight_avg_grad.max() , weight_avg_grad.min())
        avg_loss = running_loss / len(trainloader)
        print(f'[{epoch + 1}] loss: {avg_loss:.3f}')

    
    print("Finished Training")
    for param in lenet.parameters():
        a = param.get_grad_tensor()
        g = mytorch.numpy_from_tensor(a)
        print(
          "grad mean", float(g.mean()),
          "grad std", float(g.std()),
          "grad max", float(g.max()),
          "grad min", float(g.min()))


    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    lenet.eval()

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = mytorch.tensor_from_numpy(images.numpy())
            labels = mytorch.tensor_from_numpy(labels.numpy())
            outputs = lenet(images)
            outputs = mytorch.numpy_from_tensor(outputs)
            outputs = torch.from_numpy(outputs)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    print(correct_pred)
    print(total_pred)
    total = 0
    totalcorrect = 0

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        total += total_pred[classname]
        totalcorrect += correct_count
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print(f'Total Accuracy: {100 * totalcorrect / total} %')
