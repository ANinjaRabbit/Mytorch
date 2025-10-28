import sys 
sys.path.append("../build/Release/")
import mytorch
import torch, numpy as np, unittest

class TestConv(unittest.TestCase):
    def test_relu_cpu(self):
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)



        relu = mytorch.nn.ReLU()
        out = relu(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)

        ans_tensor = torch.nn.functional.relu(torch.from_numpy(a))
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

    def test_sigmoid_cpu(self):
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)



        sigmoid = mytorch.nn.Sigmoid()
        out = sigmoid(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)

        ans_tensor = torch.nn.functional.sigmoid(torch.from_numpy(a))
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))
    
    def test_relu_cuda(self):
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.to(mytorch.Cuda)



        relu = mytorch.nn.ReLU()
        out = relu(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)

        ans_tensor = torch.nn.functional.relu(torch.from_numpy(a))
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))

    def test_sigmoid_cuda(self):
        a = np.random.rand(1 , 2 , 4 ,4)
        mytorch_a = mytorch.tensor_from_numpy(a)
        mytorch_a.to(mytorch.Cuda)



        sigmoid = mytorch.nn.Sigmoid()
        out = sigmoid(mytorch_a)
        out_np = mytorch.numpy_from_tensor(out)

        ans_tensor = torch.nn.functional.sigmoid(torch.from_numpy(a))
        self.assertTrue(np.allclose(out_np, ans_tensor.numpy()))


if __name__ == '__main__':
    unittest.main()
