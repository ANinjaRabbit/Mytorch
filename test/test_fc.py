import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestFC(unittest.TestCase):
    def test_fc_cpu(self):
        w = np.random.rand(3, 4)
        b = np.random.rand(3)
        a = np.random.rand(3 ,3, 4)
        mytorchw = mytorch.tensor_from_numpy(w)
        mytorchb = mytorch.tensor_from_numpy(b)
        mytora = mytorch.tensor_from_numpy(a)



        fc = mytorch.nn.Linear(mytorchw , mytorchb) 
        out = fc(mytora)
        out_np = mytorch.numpy_from_tensor(out)
        ans_tensor = torch.nn.functional.linear(torch.from_numpy(a), torch.from_numpy(w), torch.from_numpy(b))

        #transpose because of different definition
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))
    
    def test_fc_cuda(self):
        w = np.random.rand(3, 4)
        b = np.random.rand(3)
        a = np.random.rand(3 ,3, 4)
        mytorchw = mytorch.tensor_from_numpy(w)
        mytorchb = mytorch.tensor_from_numpy(b)
        mytora = mytorch.tensor_from_numpy(a , mytorch.Cuda)


        fc = mytorch.nn.Linear(mytorchw , mytorchb) 
        out = fc(mytora)
        out_np = mytorch.numpy_from_tensor(out)
        ans_tensor = torch.nn.functional.linear(torch.from_numpy(a), torch.from_numpy(w), torch.from_numpy(b))

        #transpose because of different definition
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))


if __name__ == '__main__':
    unittest.main()
