#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <dlpack/dlpack.h>
#include "src/tensor.cuh"
#include "src/nn.cuh"
#include "src/optim.cuh"

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
        .def(py::init<const std::vector<int>&, const Device>(),
             py::arg("shape"), py::arg("device") = Cpu,
             "Create a Tensor with the given shape and device (CPU/CUDA).")
        .def(py::init<T, const std::vector<int>&, const Device>(),
             py::arg("value"), py::arg("shape") = std::vector<int>{1}, py::arg("device") = Cpu,
             "Create a Tensor filled with a constant value.")

        .def("size", &Tensor<T>::size, "Return the total number of elements.")
        .def("shape", &Tensor<T>::shape, py::return_value_policy::reference_internal, "Return the shape of the Tensor.")
        .def("device", &Tensor<T>::device, "Return the current device (CPU or CUDA).")
        .def("requires_grad", &Tensor<T>::requires_grad, "Check whether gradient computation is enabled.")
        .def("set_requires_grad", &Tensor<T>::set_requires_grad, "Enable or disable gradient computation.")
        .def("to", &Tensor<T>::to, "Move the Tensor to another device.")
        .def("item", &Tensor<T>::item, "Return the value of the Tensor as a scalar.")

        .def("set_grad_fn", &Tensor<T>::set_grad_fn, "Attach a gradient function to this Tensor.")
        .def("get_grad_fn", &Tensor<T>::get_grad_fn, "Return the gradient function attached to this Tensor.")
        .def("set_grad",  &Tensor<T>::set_grad,
             "Set the gradient Tensor for this Tensor.")
        .def("get_grad_tensor", &Tensor<T>::get_grad_tensor, "Return the gradient Tensor of this Tensor.")
        .def("zero_grad", [](Tensor<T>& self) { self.zero_grad(); }, "Set the gradient Tensor to zero.")

        .def("__add__", [](const Tensor<T>& self, const Tensor<T>& other) { return self + other; }, "Element-wise addition.")
        .def("__add__", [](const Tensor<T>& self, const T scalar) { return self + scalar; }, "Add a scalar to the Tensor.")
        .def("__radd__", [](const Tensor<T> & self , const T scalar) { return scalar + self; }, "Add a scalar to the Tensor.")
        .def("__sub__", [](const Tensor<T>& self, const Tensor<T>& other) { return self - other; }, "Element-wise subtraction.")
        .def("__sub__", [](const Tensor<T>& self, const T scalar) { return self - scalar; }, "Subtract a scalar from the Tensor.")
        .def("__rsub__", [](const Tensor<T> & self , const T scalar) { return scalar - self; }, "Subtract a scalar from the Tensor.")
        .def("__mul__", [](const Tensor<T>& self, const Tensor<T>& other) { return self * other; }, "Element-wise multiplication.")
        .def("__mul__", [](const Tensor<T>& self, const T scalar) { return self * scalar; }, "Multiply the Tensor by a scalar.")
        .def("__rmul__", [](const Tensor<T> & self , const T scalar) { return scalar * self; }, "Multiply the Tensor by a scalar.")
        .def("__truediv__", [](const Tensor<T>& self, const Tensor<T>& other) { return self / other; }, "Element-wise division.")
        .def("__truediv__", [](const Tensor<T>& self, const T scalar) { return self / scalar; }, "Divide the Tensor by a scalar.")
        .def("__rtruediv__", [](const Tensor<T> & self , const T scalar) { return scalar / self; }, "Divide a scalar by the Tensor.")
        .def("__neg__", (Tensor<T> (Tensor<T>::*)() const) &Tensor<T>::operator-, "Negation of the Tensor.")

        .def("relu", &Tensor<T>::relu, "Apply ReLU activation.")
        .def("sigmoid", &Tensor<T>::sigmoid, "Apply Sigmoid activation.")
        .def("reshape", &Tensor<T>::reshape, "Reshape the Tensor to a new shape.")
        .def("transpose",
            [](Tensor<T>& self, py::object perm_obj) {
                if (perm_obj.is_none()) {
                    return self.transpose({});  
                } else {
                    std::vector<int> perm = perm_obj.cast<std::vector<int>>();
                    return self.transpose(perm);
                }
            }
             , py::arg("perm") = py::none(), "Transpose Tensor dimensions.")
        .def("matmul", &Tensor<T>::matmul, "Matrix multiplication.")
        .def("maxpool2d", &Tensor<T>::maxpool2d, "2D max pooling operation.")
        .def("expand", &Tensor<T>::expand, "Expand Tensor to a new shape (broadcasting).")
        .def("print", &Tensor<T>::print, "Print the Tensor contents.")
        .def("deepcopy", &Tensor<T>::deepcopy, "Return a deep copy of the Tensor.")
        .def("backward",
        [](Tensor<T>& self, py::object grad_obj) {
            if (grad_obj.is_none()) {
                self.backward(Tensor<T>());  
            } else {
                Tensor<T>& grad = grad_obj.cast<Tensor<T>&>();
                self.backward(grad);
            }
        },
        py::arg("grad_output") = py::none(),
        "Compute gradients in the backward pass.")
        .def("sum", &Tensor<T>::sum, "Sum all elements in the Tensor.")
        .def("numpy" , [](const Tensor<T>& self){
            if(self.get() == nullptr){
                std::cerr << "Error: Tensor is nullptr. Cannot convert to numpy array." << std::endl;
                throw std::runtime_error("Error: Tensor is nullptr. Cannot convert to numpy array.");
            }
            py::array_t<T> array(self.shape());
            if(self.device() == Device::Cpu){
                memcpy(array.mutable_data() , self.get() , sizeof(T) * self.size());
            }
            else if(self.device() == Device::Cuda){
                CHECK(cudaMemcpy(array.mutable_data() , self.get() , sizeof(T) * self.size() , cudaMemcpyDeviceToHost));
            }
            return array;
        })
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
        .def(py::init<>())
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
    BIND_FUNC(MaxPool2dFunc);
    BIND_FUNC(ReshapeFunc);
    BIND_FUNC(MatmulFunc);
    BIND_FUNC(ModuleFunctionWrapper);
    BIND_FUNC(SumFunc);

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
        .def(py::init<>())
        .def("forward", &Module<T>::forward, "Forward pass through the module.")
        .def("_internal_backward", &Module<T>::_internal_backward, "Backward pass (computes parameter gradients).")
        .def("parameters", &Module<T>::parameters, "Return a list of learnable parameters.")
        .def("__call__" , (Tensor<T>(Module<T>::*)(const Tensor<T>&)) &Module<T>::operator() , "Forward pass through the module for only one input.")
        .def("train" , &Module<T>::train , "Set the module to training mode.")
        .def("eval" , &Module<T>::eval , "Set the module to evaluation mode.")
        .def("zero_grad" , &Module<T>::zero_grad , "Zero the gradients of all learnable parameters.")
        ;

        // Subclasses
        py::class_<Linear<T>, Module<T>, std::shared_ptr<Linear<T>>>(m_mod, "Linear",
        "Fully connected layer: y = xW^T + b.")
        .def(py::init(
            [] (const int in_features , const int out_features , py::object device_obj)
            {
                Device device = device_obj.is_none() ? DefaultDevice : device_obj.cast<Device>();
                return std::make_shared<Linear<T>>(in_features , out_features , device);
            }
        ) , py::arg("in_features") , py::arg("out_features") , py::arg("device") = py::none())
        .def_readwrite("weight" , &Linear<T>::weight)
        .def_readwrite("bias" , &Linear<T>::bias)
        ;

        py::class_<Conv2d<T>, Module<T>, std::shared_ptr<Conv2d<T>>>(m_mod, "Conv2d",
            "Convolutional layer using a learnable kernel.")
        .def(py::init([](int in_channels,
                        int out_channels,
                        py::object kernel_size,
                        py::object padding,
                        py::object stride,
                        py::object device_obj)
        {
            Device device = device_obj.is_none() ? DefaultDevice : device_obj.cast<Device>();
            auto parse_pair = [](py::object obj, int &x, int &y) {
                if (py::isinstance<py::int_>(obj)) {
                    int v = obj.cast<int>();
                    x = y = v;
                } else if (py::isinstance<py::tuple>(obj)) {
                    auto t = obj.cast<py::tuple>();
                    if (t.size() != 2)
                        throw std::runtime_error("Tuple size must be 2");
                        auto item0 = t[0];
                        auto item1 = t[1];
                        x = py::cast<int>(item0);
                        y = py::cast<int>(item1);
                } else {
                    throw std::runtime_error("Argument must be int or tuple");
                }
            };

            int kw, kh;
            int pad_w, pad_h;
            int stride_w, stride_h;

            parse_pair(kernel_size, kw, kh);
            parse_pair(padding, pad_w, pad_h);
            parse_pair(stride, stride_w, stride_h);

            return std::make_shared<Conv2d<T>>(
                in_channels, out_channels,
                kw, kh,
                pad_h, pad_w,
                stride_h, stride_w,
                device
            );
        }),
        py::arg("in_channels"),
        py::arg("out_channels"),
        py::arg("kernel_size"),
        py::arg("padding") = 0,
        py::arg("stride") = 1,
        py::arg("device") = py::none()
        )
        .def_readwrite("kernel" , &Conv2d<T>::kernel)
        .def_readwrite("bias" , &Conv2d<T>::bias)
        ;

    
    py::class_<MaxPool2d<T>, Module<T>, std::shared_ptr<MaxPool2d<T>>>(m_mod, "MaxPool2d",
        "2D max pooling layer.")
        .def(py::init([](const std::vector<int> & kernel_shape , py::object device_obj)
        {
            Device device = device_obj.is_none() ? DefaultDevice : device_obj.cast<Device>();
            return std::make_shared<MaxPool2d<T>>(kernel_shape , device);
        }
        ) , py::arg("kernel_shape") , py::arg("device") = py::none());

    py::class_<Softmax<T>, Module<T>, std::shared_ptr<Softmax<T>>>(m_mod, "Softmax",
        "Softmax activation module.")
        .def(py::init<>());

    py::class_<CrossEntropy<T>, Module<T>, std::shared_ptr<CrossEntropy<T>>>(m_mod, "CrossEntropy",
        "CrossEntropy loss function.")
        .def(py::init<>())
        .def("__call__" , [](CrossEntropy<T> & self , const Tensor<T> & y_pred , const Tensor<T> & y_true)
        {
            return self.forward({y_pred , y_true});
        } , py::arg("y_pred") , py::arg("y_true"))
        ;
    
    py::class_<ReLU<T> , Module<T> , std::shared_ptr<ReLU<T>>>(m_mod, "ReLU",
        "ReLU activation module.")
        .def(py::init<>());
    
    py::class_<Sigmoid<T> , Module<T> , std::shared_ptr<Sigmoid<T>>>(m_mod, "Sigmoid",
        "Sigmoid activation module.")
        .def(py::init<>());
    
    py::class_<BatchNorm2d<T> , Module<T> , std::shared_ptr<BatchNorm2d<T>>>(m_mod, "BatchNorm2d",
        "Batch normalization layer for 2D input (e.g., images).")
        .def(py::init([](const int num_features  , py::object momentum_obj, py::object device_obj){
            T momentum = momentum_obj.is_none() ? T(0.1) : momentum_obj.cast<T>();
            Device device = device_obj.is_none() ? DefaultDevice : device_obj.cast<Device>();
            return std::make_shared<BatchNorm2d<T>>(num_features , momentum , device);
        }) , py::arg("num_features") , py::arg("momentum") = py::none() , py::arg("device") = py::none())
        .def("train" , &BatchNorm2d<T>::train , "Set the module to training mode.")
        .def("eval" , &BatchNorm2d<T>::eval , "Set the module to evaluation mode.")
        .def_readwrite("gamma" , &BatchNorm2d<T>::gamma)
        .def_readwrite("beta" , &BatchNorm2d<T>::beta)
        .def_readwrite("running_mean" , &BatchNorm2d<T>::running_mean)
        .def_readwrite("running_var" , &BatchNorm2d<T>::running_var)
        .def_readwrite("momentum" , &BatchNorm2d<T>::momentum)
        .def_readwrite("epsilon" , &BatchNorm2d<T>::epsilon)
        .def_readwrite("c" , &BatchNorm2d<T>::c);
        

    py::class_<Sequential<T> , Module<T> , std::shared_ptr<Sequential<T>>>(m_mod, "Sequential",
        "Container for a sequence of modules.")
        .def(py::init([](std::vector<std::shared_ptr<Module<T>>> &modules){
            return std::make_shared<Sequential<T>>(modules);
        }) , py::arg("modules"))
        .def("train" , &Sequential<T>::train , "Set the module to training mode.")
        .def("eval" , &Sequential<T>::eval , "Set the module to evaluation mode.");
    
    py::class_<Flatten<T> , Module<T> , std::shared_ptr<Flatten<T>>>(m_mod, "Flatten",
        "Flatten layer to convert multi-dimensional input to a 1D vector.")
        .def(py::init<int , int>() , py::kw_only() , py::arg("start_dim") = 0 , py::arg("end_dim") = -1);

    py::class_<ResNet18<T> , Module<T> , std::shared_ptr<ResNet18<T>>>(m_mod, "ResNet18",
        "ResNet18 model.")
        .def(py::init<int , int , int>() , py::kw_only() , py::arg("num_classes") , py::arg("h") = 32 , py::arg("w") = 32)
        .def("eval" , &ResNet18<T>::eval , "Set the module to evaluation mode.")
        .def("train" , &ResNet18<T>::train , "Set the module to training mode.");

    
    py::class_<MiniResNet<T> , Module<T> , std::shared_ptr<MiniResNet<T>>>(m_mod, "MiniResNet",
        "MiniResNet model.")
        .def(py::init<int , int , int>() , py::kw_only() , py::arg("num_classes") , py::arg("h") = 32 , py::arg("w") = 32)
        .def("eval" , &MiniResNet<T>::eval , "Set the module to evaluation mode.")
        .def("train" , &MiniResNet<T>::train , "Set the module to training mode.");
    
}

template <typename T>
void bind_optim(py::module &m_optim) {
    using namespace mytorch::optim;
    // ---------------- Optimizer ----------------
    py::class_<Optimizer<T> , std::shared_ptr<Optimizer<T>>>(m_optim, "Optimizer",
        "Base class for optimizers.")
        .def(py::init<T , T>() , py::kw_only() , py::arg("lr") = T(0.01) , py::arg("weight_decay") = T(0.0))
        .def("step", &Optimizer<T>::step, "Update all parameters in-place.");

    // ---------------- SGD ----------------
    py::class_<SGD<T> , Optimizer<T> , std::shared_ptr<SGD<T>>>(m_optim, "SGD",
        R"doc(
        Stochastic Gradient Descent optimizer.

        Args:
            params (List[Tensor]): List of tensors to optimize.
            lr (float): Learning rate. Default: 0.01
            device (Device): Compute device. Default: DefaultDevice

        Methods:
            step(): Update all parameters in-place.
        )doc")
        .def(py::init([](std::vector<Tensor<T>> &params,
                 T lr,
                 T weight_decay,
                 py::object device_obj)
            {
                Device device = device_obj.is_none() ? DefaultDevice : device_obj.cast<Device>();
                return std::make_shared<SGD<T>>(params, lr, device);
            }),
            py::arg("params"),
            py::kw_only(),
            py::arg("lr") = T(0.01),
            py::arg("weight_decay") = T(0.0),
            py::arg("device") = py::none())
        .def("zero_grad" , &SGD<T>::zero_grad , "Zero the gradients of all parameters.")
        .def("step", &SGD<T>::step, "Perform one SGD update step.");

    // ---------------- Adam ----------------
    py::class_<Adam<T> , Optimizer<T> , std::shared_ptr<Adam<T>>>(m_optim, "Adam",
        R"doc(
        Adam optimizer.

        Args:
            params (List[Tensor]): List of tensors to optimize.
            lr (float): Learning rate. Default: 0.001
            beta1 (float): Exponential decay for first moment. Default: 0.9
            beta2 (float): Exponential decay for second moment. Default: 0.999
            eps (float): Numerical stability constant. Default: 1e-8
            weight_decay (float): Weight decay (L2 penalty). Default: 0
            device (Device): Compute device. Default: DefaultDevice

        Methods:
            step(): Update all parameters using Adam algorithm.
        )doc")
        .def(py::init([](std::vector<Tensor<T>> &params,
                T lr,
                T beta1,
                T beta2,
                T eps,
                T weight_decay,
                py::object device_obj)
            {
                Device device = device_obj.is_none() ? DefaultDevice : device_obj.cast<Device>();
                return std::make_shared<Adam<T>>(params, lr, beta1, beta2, eps, weight_decay, device);
            }),
            py::arg("params"),
            py::arg("lr") = T(0.001),
            py::arg("beta1") = T(0.9),
            py::arg("beta2") = T(0.999),
            py::arg("eps") = T(1e-8),
            py::arg("weight_decay") = T(0),
            py::arg("device") = py::none())
        .def("zero_grad" , &Adam<T>::zero_grad , "Zero the gradients of all parameters.")
        .def("step", &Adam<T>::step, "Perform one Adam update step.");

        auto m_lr = m_optim.def_submodule("lr_scheduler" , "Learning rate scheduler.");
        using namespace mytorch::optim::lr_scheduler;

        // ---------------- LR Scheduler ----------------
        py::class_<StepLR<T> , std::shared_ptr<StepLR<T>>>(m_lr, "StepLR",
        R"doc(
        Step learning rate scheduler.

        Args:
            optimizer (Optimizer): Optimizer to schedule.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1
            step_size (int): Period of learning rate decay. Default: 100

        Methods:
            step(): Update learning rate for each parameter group.
        )doc")
        .def(py::init([](std::shared_ptr<Optimizer<T>> optimizer, T gamma, int step_size)
            {
                return std::make_shared<StepLR<T>>(optimizer, gamma, step_size);
            }),
            py::arg("optimizer"),
            py::arg("gamma") = T(0.1),
            py::arg("step_size") = int(100))
        .def("step", &StepLR<T>::step, "Update learning rate for each parameter group.");



        py::class_<CosineAnnealingLR<T> , std::shared_ptr<CosineAnnealingLR<T>>>(m_lr, "CosineAnnealingLR",
        R"doc(
        Cosine annealing learning rate scheduler.

        Args:
            optimizer (Optimizer): Optimizer to schedule.
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0

        Methods:
            step(): Update learning rate for each parameter group.
        )doc")
        .def(py::init([](std::shared_ptr<Optimizer<T>> optimizer, T T_max, T eta_min = T(0))
            {
                return std::make_shared<CosineAnnealingLR<T>>(optimizer, T_max, eta_min);
            }),
            py::arg("optimizer"),
            py::arg("T_max"),
            py::arg("eta_min") = T(0))
        .def("step", &CosineAnnealingLR<T>::step, "Update learning rate for each parameter group.");
}


template <typename T>
Tensor<T> tensor_from_numpy(py::array_t<float> data , Device device = DefaultDevice)
{
    std::vector<int> shape(data.ndim());
    for (int i = 0; i < data.ndim(); ++i) {
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
Tensor<T> from_dlpack_deepcopy(py::capsule dlpack_capsule , Device targetdevice = DefaultDevice)
{
    DLManagedTensor* dlm = dlpack_capsule.get_pointer<DLManagedTensor>();

    DLTensor& dl = dlm->dl_tensor;
    DLDevice ctx = dl.device;

    std::vector<int> shape(dl.ndim);
    for (int i = 0; i < dl.ndim; i++) {
        shape[i] = dl.shape[i];
    }
    Device device;
    if (ctx.device_type == kDLCPU) {
        device = Device::Cpu;
    } else if (ctx.device_type == kDLCUDA) {
        device = Device::Cuda;
    } else {
        throw std::runtime_error("Unsupported device type in DLPack");
    }
    T * ptr = reinterpret_cast<T*>(dl.data);


    cuda_shared_pointer<T> cuda_ptr(ptr , std::accumulate(shape.begin() , shape.end() , 1 , std::multiplies<int>()) , targetdevice , device);

    return make_view<T>(cuda_ptr , shape);
}

template <typename T>
void bind_f(py::module & m){

    m.def("zeros",
        [](const std::vector<int>& shape, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return zeros<T>(shape, dev);
        },
        py::arg("shape"), py::arg("device") = py::none(),
        "Create tensor filled with 0");

    m.def("ones",
        [](const std::vector<int>& shape, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return ones<T>(shape, dev);
        },
        py::arg("shape"), py::arg("device") = py::none(),
        "Create tensor filled with 1");

    m.def("arange",
        [](T start, T end, T step, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return arange<T>(start, end, step, dev);
        },
        py::arg("start"), py::arg("end"),
        py::arg("step") = T(1),
        py::arg("device") = py::none(),
        "Create tensor with range [start, end)");

    m.def("rand",
        [](const std::vector<int>& shape, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return rand<T>(shape, dev);
        },
        py::arg("shape"), py::arg("device") = py::none(),
        "Create uniform random tensor");

    m.def("randn",
        [](const std::vector<int>& shape, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return randn<T>(shape, dev);
        },
        py::arg("shape"), py::arg("device") = py::none(),
        "Create normal random tensor");

    m.def("full",
        [](const std::vector<int>& shape, T value, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return full<T>(shape, value, dev);
        },
        py::arg("shape"), py::arg("value"),
        py::arg("device") = py::none(),
        "Create tensor filled with value");

    m.def("tensor_from_numpy",
        [](py::array_t<float> data, py::object dev_obj) {
            Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
            return tensor_from_numpy<T>(data, dev);
        },
        py::arg("data"), py::arg("device") = py::none(),
        "Create tensor from numpy array.");

    m.def("numpy_from_tensor", &numpy_from_tensor<T>,
        py::arg("tensor"),
        "Create numpy array from tensor.");
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
    auto m_optim = m.def_submodule("optim", "Optimization algorithms.");


    m.def("get_default_device", []() {
        return DefaultDevice;
    }).def("set_default_device", [](Device d) {
        DefaultDevice = d;
    })
    .def("from_dlpack_deepcopy", [](py::capsule dlpack , py::object dev_obj) {
        Device dev = dev_obj.is_none() ? DefaultDevice : dev_obj.cast<Device>();
        return from_dlpack_deepcopy<float>(dlpack , dev);
    },
    py::arg("dlpack"),
    py::arg("device") = py::none(),
    "Create tensor from DLPack.");
    

    // Bind core components
    bind_tensor<float>(m);
    bind_function<float>(m_func);
    bind_module<float>(m_mod);
    bind_f<float>(m);
    bind_optim<float>(m_optim);
}
