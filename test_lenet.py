import sys 
sys.path.append("/build/Release/")
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

    x = mytorch.randn((10 , 1 , 28 , 28))
    x.set_requires_grad(True)
    print(lenet.parameters())
    label = mytorch.randn((10 , 10))
    optim = mytorch.optim.Adam(lenet.parameters() , 0.001)
    
    for i in range(100):
        y = lenet(x)
        loss = y - label
        loss = loss * loss
        loss = loss.sum(1)
        loss = loss.sum(0)
        print(loss.item())
        loss.zero_grad()
        loss.backward()
        optim.step()



