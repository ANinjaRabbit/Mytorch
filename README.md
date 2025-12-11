# 🧠 MyTorch - A Minimal Deep Learning Framework

## ⚠️ Warning
- This repository is only for my programming in ai course(PKU). It is not intended for production use.
- The framework is still under development, and some features may be incomplete or subject to change.

## 📖 Introduction
MyTorch is a lightweight deep learning framework written in C++/CUDA with automatic differentiation, neural network modules, optimizers, and Python bindings.

## ✨ Key Features
1. **🔢 Tensor Operations**: Support for multi-dimensional tensor computations (addition, subtraction, multiplication, division, matrix multiplication, etc.)
2. **🤖 Automatic Differentiation**: Implementation of a computation graph-based automatic differentiation system
3. **🧠 Neural Network Modules**: Common neural network layers (Linear, Conv2d, Pooling, Batch Normalization, etc.)
4. **⚙️ Optimizers**: Implementation of optimization algorithms like SGD and AdamW
5. **⚡ GPU Acceleration**: High-performance computing using CUDA
6. **🐍 Python Bindings**: Python interface provided through pybind11
7. **🌈 dlpack support**: Support for transforming from dlpack tensor format to mytorch tensor format

## 🧩 Core Components

### 🔷 Tensor
- Support for both CPU and CUDA devices
- Implementation of basic mathematical operations and neural network operations
- Automatic differentiation capability

### 🔄 Autograd
- Computation graph-based backpropagation
- Topological sorting algorithm to ensure correct gradient computation order

### 🏗️ Neural Network Modules (NN)
- Basic modules: Linear, Conv2d, MaxPool2d, AvgPool2d
- Activation functions: ReLU, Sigmoid, Softmax
- Regularization: BatchNorm2d
- Loss functions: CrossEntropy
- Advanced architectures: ResNet series models

### 🎯 Optimizers (Optim)
- SGD (with momentum)
- AdamW
- Learning rate schedulers: StepLR, CosineAnnealingLR

## 🛠️ Build Instructions
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## 📦 Dependencies
- CUDA Toolkit
- cuBLAS
- cuRAND
- pybind11

## ⚡ Performance Characteristics
- Parallel computing acceleration using CUDA
- Optimized memory management
- Efficient convolution implementation (using im2col and GEMM)

## 🧪 Pre-implemented Modelss
- ResNet18/34
- ResNeXt18/34

🔬