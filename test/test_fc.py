import sys 
sys.path.append("../build/Release/")
import mytorch

a = mytorch.rand([3 , 3])
a.set_requires_grad(True)
w = mytorch.rand([3 , 3])
b = mytorch.rand([3])

fc = mytorch.nn.Linear(w , b)
out = fc(a)
out.print()
out.get_grad_fn().backward(mytorch.ones(out.shape()))[0].print()


