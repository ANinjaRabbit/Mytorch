import torch
import time




conv = torch.nn.Conv2d(3 , 3 , 3 )

tot_time = 0
for i in range(100):
    a = torch.randn(1 , 3 , 400 , 400)
    start = time.time()
    b = conv(a)
    end = time.time()
    tot_time += end - start

print("Average Time: %f ms" % (tot_time / 100 * 1000))
