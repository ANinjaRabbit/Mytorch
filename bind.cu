#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "src/tensor.cuh"
#include "src/nn.cuh"

namespace py = pybind11;
using namespace mytorch;

template <typename T>
void bind_tensor(py::module &m) {
    py::class_<Tensor<T>, std::shared_ptr<Tensor<T>>>(m, "Tensor",
        R"doc(
        Tensor represents a multi-dimensional array that supports automatic differentiation.
        It holds data, gradients, and links to gradient functions in the computation graph.
        )doc")
        .def(py::init<>(), "Create an empty Tensor.")
        .def(py::init<const std::vector<size_t>&, const Device>(),
             py::arg("shape"), py::arg("device") = Cpu,
             "Create a Tensor with the given shape and device (CPU/CUDA).")
        .def(py::init<T, const std::vector<size_t>&, const Device>(),
             py::arg("value"), py::arg("shape") = std::vector<size_t>{1}, py::arg("device") = Cpu,
             "Create a Tensor filled with a constant value.")

        .def("size", &Tensor<T>::size, "Return the total number of elements.")
        .def("shape", &Tensor<T>::shape, py::return_value_policy::reference_internal, "Return the shape of the Tensor.")
        .def("device", &Tensor<T>::device, "Return the current device (CPU or CUDA).")
        .def("requires_grad", &Tensor<T>::requires_grad, "Check whether gradient computation is enabled.")
        .def("set_requires_grad", &Tensor<T>::set_requires_grad, "Enable or disable gradient computation.")
        .def("to", &Tensor<T>::to, "Move the Tensor to another device.")

        .def("set_grad_fn", &Tensor<T>::set_grad_fn, "Attach a gradient function to this Tensor.")
        .def("get_grad_fn", &Tensor<T>::get_grad_fn, "Return the gradient function attached to this Tensor.")
        .def("set_grad",  &Tensor<T>::set_grad,
             "Set the gradient Tensor for this Tensor.")
        .def("get_grad_tensor", &Tensor<T>::get_grad_tensor, "Return the gradient Tensor of this Tensor.")

        .def("__add__", &Tensor<T>::operator+, "Element-wise addition.")
        .def("__sub__", (Tensor<T> (Tensor<T>::*)(const Tensor<T>&) const) &Tensor<T>::operator-, "Element-wise subtraction.")
        .def("__mul__", &Tensor<T>::operator*, "Element-wise multiplication.")
        .def("__truediv__", &Tensor<T>::operator/, "Element-wise division.")
        .def("__neg__", (Tensor<T> (Tensor<T>::*)() const) &Tensor<T>::operator-, "Negation of the Tensor.")

        .def("relu", &Tensor<T>::relu, "Apply ReLU activation.")
        .def("sigmoid", &Tensor<T>::sigmoid, "Apply Sigmoid activation.")
        .def("reshape", &Tensor<T>::reshape, "Reshape the Tensor to a new shape.")
        .def("transpose", &Tensor<T>::transpose, "Transpose Tensor dimensions.")
        .def("matmul", &Tensor<T>::matmul, "Matrix multiplication.")
        .def("pool2d", &Tensor<T>::pool2d, "2D pooling operation.")
        .def("expand", &Tensor<T>::expand, "Expand Tensor to a new shape (broadcasting).")
        .def("print", &Tensor<T>::print, "Print the Tensor contents.")
        .def("deepcopy", &Tensor<T>::deepcopy, "Return a deep copy of the Tensor.")
        ;
}

// ========================== Function Binding ==========================
template <typename T>
void bind_function(py::module &m_func) {
    using namespace nn::Functional;

    py::class_<Function<T>, std::shared_ptr<Function<T>>>(m_func, "Function",
        R"doc(
        Base class for differentiable operations in the computation graph.
        Each Function defines forward() and backward() for autograd.
        )doc")
        .def("forward", &Function<T>::forward, "Perform the forward pass.")
        .def("backward", &Function<T>::backward, "Compute gradients in the backward pass.")
        .def("get_inputs", &Function<T>::get_inputs, "Return input Tensors of this Function.")
        .def("__call__" , (Tensor<T> (Function<T>::*)(const Tensor<T>&)) &Function<T>::operator() , "Forward pass through the function for only one input.");

    // Macro-like helper for subclasses
    #define BIND_FUNC(ClassName) \
        py::class_<ClassName<T>, Function<T>, std::shared_ptr<ClassName<T>>>(m_func, #ClassName, "Autograd function: " #ClassName)

    BIND_FUNC(NegFunc);
    BIND_FUNC(AddFunc);
    BIND_FUNC(SubFunc);
    BIND_FUNC(MulFunc);
    BIND_FUNC(DivFunc);
    BIND_FUNC(ReLUFunc);
    BIND_FUNC(SigmoidFunc);
    BIND_FUNC(TransposeFunc);
    BIND_FUNC(Pool2dFunc);
    BIND_FUNC(ReshapeFunc);
    BIND_FUNC(MatmulFunc);
    BIND_FUNC(ModuleFunctionWrapper);

    #undef BIND_FUNC
}

// ========================== Module Binding ==========================
template <typename T>
void bind_module(py::module &m_mod) {
    using namespace nn;

    py::class_<Module<T>, std::shared_ptr<Module<T>>>(m_mod, "Module",
        R"doc(
        Base class for neural network layers (modules).
        Each Module defines forward() and parameters(), and may override backward().
        )doc")
        .def("forward", &Module<T>::forward, "Forward pass through the module.")
        .def("_internal_backward", &Module<T>::_internal_backward, "Backward pass (computes parameter gradients).")
        .def("parameters", &Module<T>::parameters, "Return a list of learnable parameters.")
        .def("__call__" , (Tensor<T>(Module<T>::*)(const Tensor<T>&)) &Module<T>::operator() , "Forward pass through the module for only one input.")
        ;

    // Subclasses
    py::class_<Linear<T>, Module<T>, std::shared_ptr<Linear<T>>>(m_mod, "Linear",
        "Fully connected layer: y = xW^T + b.")
        .def(py::init<const Tensor<T>&, const Tensor<T>&>(), py::arg("weight"), py::arg("bias"));

    py::class_<Conv<T>, Module<T>, std::shared_ptr<Conv<T>>>(m_mod, "Conv",
        "Convolutional layer using a learnable kernel.")
        .def(py::init<const Tensor<T>&>(), py::arg("kernel"));

    py::class_<Pool2d<T>, Module<T>, std::shared_ptr<Pool2d<T>>>(m_mod, "Pool2d",
        "2D pooling layer (e.g., max pooling).")
        .def(py::init<std::vector<size_t>>(), py::arg("kernel_shape"));

    py::class_<Softmax<T>, Module<T>, std::shared_ptr<Softmax<T>>>(m_mod, "Softmax",
        "Softmax activation module.")
        .def(py::init<>());

    py::class_<CrossEntropy<T>, Module<T>, std::shared_ptr<CrossEntropy<T>>>(m_mod, "CrossEntropy",
        "CrossEntropy loss function.")
        .def(py::init<const std::vector<size_t>&>(), py::arg("label_cache"));
    
    py::class_<ReLU<T> , Module<T> , std::shared_ptr<ReLU<T>>>(m_mod, "ReLU",
        "ReLU activation module.")
        .def(py::init<>());
    
    py::class_<Sigmoid<T> , Module<T> , std::shared_ptr<Sigmoid<T>>>(m_mod, "Sigmoid",
        "Sigmoid activation module.")
        .def(py::init<>());
}


template <typename T>
Tensor<T> tensor_from_numpy(py::array_t<float> data , Device device = Device::Cpu)
{
    std::vector<size_t> shape(data.ndim());
    for (size_t i = 0; i < data.ndim(); ++i) {
        shape[i] = data.shape(i);
    }
    Tensor<T> tensor(shape , device);
    if(device == Device::Cpu){
        memcpy(tensor.get() , data.data() , sizeof(T) * tensor.size());
    }
    else if(device == Device::Cuda){
        cudaMemcpy(tensor.get() , data.data() , sizeof(T) * tensor.size() , cudaMemcpyHostToDevice);
    }
    return tensor;
}
template <typename T>
py::array_t<float> numpy_from_tensor(const Tensor<T>& tensor)
{
    py::array_t<float> array(tensor.shape());
    if(tensor.device() == Device::Cpu){
        memcpy(array.mutable_data() , tensor.get() , sizeof(T) * tensor.size());
    }
    else if(tensor.device() == Device::Cuda){
        cudaMemcpy(array.mutable_data() , tensor.get() , sizeof(T) * tensor.size() , cudaMemcpyDeviceToHost);
    }
    return array;
}
template <typename T>
void bind_f(py::module & m){
         m.def("zeros", &zeros<T>, 
          py::arg("shape"), py::arg("device") = Device::Cpu, "Create tensor filled with 0");
    m.def("ones", &ones<T>, 
          py::arg("shape"), py::arg("device") = Device::Cpu, "Create tensor filled with 1");
    m.def("arange", &arange<T>, 
          py::arg("start"), py::arg("end"), py::arg("step") = T(1), py::arg("device") = Device::Cpu, "Create tensor with range [start, end) step");
    m.def("rand", &rand<float>, 
          py::arg("shape"), py::arg("device") = Device::Cpu, "Create tensor with random values (0~1)");
    m.def("randn", &randn<T>, 
          py::arg("shape"), py::arg("device") = Device::Cpu, "Create tensor with normal distribution (mean=0, std=1)");
    m.def("full", &full<T>, 
          py::arg("shape"), py::arg("value"), py::arg("device") = Device::Cpu, "Create tensor filled with 'value'");
    m.def("tensor_from_numpy" , &tensor_from_numpy<T> , py::arg("data") , py::arg("device")=Device::Cpu , "Create tensor from numpy array.");
    m.def("numpy_from_tensor" , &numpy_from_tensor<T> , py::arg("tensor") , "Create numpy array from tensor.");
}

// ========================== Module Registration ==========================
PYBIND11_MODULE(mytorch, m) {
    m.doc() = "MyTorch: A minimal neural network and autograd engine in C++.";

    py::enum_<Device>(m, "Device")
        .value("Cpu", Device::Cpu)
        .value("Cuda", Device::Cuda)
        .export_values();

    // Submodules
    auto m_func = m.def_submodule("Functional", "Autograd function definitions.");
    auto m_mod = m.def_submodule("nn", "Neural network modules.");

    // Bind core components
    bind_tensor<float>(m);
    bind_function<float>(m_func);
    bind_module<float>(m_mod);
    bind_f<float>(m);
}
