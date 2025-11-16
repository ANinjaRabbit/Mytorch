import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class MyModule(mytorch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = mytorch.randn((3 , 3))
        self.b1 = mytorch.randn((3 ,))
        self.w2 = mytorch.randn((1 , 3))
        self.b2 = mytorch.randn((1 ,))
        self.relu = mytorch.nn.ReLU()
        self.fc1 = mytorch.nn.Linear(self.w1 , self.b1)
        self.fc2 = mytorch.nn.Linear(self.w2 , self.b2)
    def forward(self , input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    def parameters(self):
        return [self.w1 , self.b1 , self.w2 , self.b2]
    def __call__(self , input):
        return self.forward(input)


if __name__ == '__main__':
    m = MyModule()
    input = mytorch.randn((1 , 3))
    input.set_requires_grad(True)
    output = m(input)
    output.print()
    output.backward()
