import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



transform = transforms.Compose( [transforms.ToTensor() , transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))])
batch_size = 32
trainset = torchvision.datasets.CIFAR10(root="./data" , train=True , download=True , transform=transform)
trainloader = torch.utils.data.DataLoader(trainset , batch_size=batch_size , shuffle=True , num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./data" , train=False , download=True , transform=transform)
testloader = torch.utils.data.DataLoader(testset , batch_size=batch_size , shuffle=True , num_workers=2)
writer = SummaryWriter()
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

        self.fc = nn.Linear(32 * 32 * 3 , 10)
    def forward(self , x):
        x = self.fc(torch.flatten(x , 1))
        return x
#        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
#        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
#        x = torch.flatten(x , 1)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x


if __name__ == '__main__':
    momentum = 0.9
    print(f"test on momentum {momentum:.2f}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lenet = LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet.parameters() , lr = 0.01 )


    for epoch in range(1):
        running_loss = 0.0
        weight_avg_grad = None
        for i,data in enumerate(trainloader , 0):
            inputs , labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = lenet(inputs)
            loss = criterion(outputs , labels)
            loss.backward()
            if(weight_avg_grad is None):
                weight_avg_grad = lenet.fc.weight.grad.cpu().numpy()
            else:
                weight_avg_grad += lenet.fc.weight.grad.cpu().numpy()
            optimizer.step()
            running_loss+= loss.item()
            print(f"loss {loss.item():.3f}")
        weight_avg_grad /= len(trainloader)
        print(weight_avg_grad.mean() , weight_avg_grad.std() , weight_avg_grad.max() , weight_avg_grad.min())
        avg_loss = running_loss / len(trainloader)
        writer.add_scalar(f"Loss/train momentum {momentum}",avg_loss,epoch)
        print(f'[{epoch + 1}] loss: {avg_loss:.3f}')

    
    print("Finished Training")

    for param in lenet.parameters():
        print(param.grad.mean() , param.grad.std() , param.grad.max() , param.grad.min())


    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = lenet(images)
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
    writer.close()
