import sys 
sys.path.append("../build/Release/")
import mytorch


if __name__ == "__main__":
    mytorch.set_default_device(mytorch.Cuda)
    lenet = mytorch.nn.Sequential([
        mytorch.nn.Conv([6 , 1 , 5 , 5] , mytorch.nn.NoPadding) ,
        mytorch.nn.ReLU() ,
        mytorch.nn.Pool2d((2 , 2)) ,
        mytorch.nn.Conv([16 , 6 , 5 , 5] , mytorch.nn.NoPadding) ,
        mytorch.nn.ReLU() ,
        mytorch.nn.Pool2d((2 , 2)) ,
        mytorch.nn.Flatten(start_dim = 1) ,
        mytorch.nn.Linear(16 * 4 * 4 , 120) ,
        mytorch.nn.ReLU() ,
        mytorch.nn.Linear(120 , 84) ,
        mytorch.nn.ReLU() ,
        mytorch.nn.Linear(84 , 10)]
    )

    x = mytorch.ones((1 , 1 , 28 , 28))
    x.set_requires_grad(True)
    print(lenet.parameters())
    label = mytorch.ones((1 , 10))
    optim = mytorch.optim.SGD(lenet.parameters() , 0.001)
    
    for i in range(10):
        y = lenet(x)
        loss = y - label
        loss = loss * loss
        loss.print()
        loss.zero_grad()
        loss.backward()
        optim.step()



