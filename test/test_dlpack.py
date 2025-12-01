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


mytorch.set_default_device(mytorch.Cuda)
x = torch.ones(1 ,1,4 , 4).to("cuda")
myx = mytorch.from_dlpack(torch.utils.dlpack.to_dlpack(x))
#myx = mytorch.from_dlpack_deepcopy(torch.utils.dlpack.to_dlpack(x))
conv = mytorch.nn.Conv2d(1 , 1 , 3)
out = conv(myx)
out.print()