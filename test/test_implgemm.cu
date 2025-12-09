#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include <iostream>
using namespace mytorch;

int main(){
    Tensor<float> input = ones<float>({2 , 2 , 5 , 5});
    Tensor<float> kernel = ones<float>({2 , 2 , 3 , 3});
    Tensor<float> output({2 , 2 , 2 , 2});
    Tensor<float> bias= zeros<float>({2});
    nn::conv2d_forward_gpu<float>(output.get() , input.get() , kernel.get(),
        bias.get() , 2 ,2 , 5 , 5 , 2 , 3 , 3 , 0 , 0 , 2 , 2 , 2 , 2
    );
    output.print();
    Tensor<float> gradout = ones<float>({2 , 2 , 2 , 2});
    Tensor<float> gradinput({2 , 2 ,5 , 5});
    Tensor<float> gradkernel({2 , 2 , 3 , 3});
    nn::implGemmgradinput<float>(gradinput.get() , gradout.get() , kernel.get() , 2 , 2 , 5 , 5 , 2 , 3 , 3 , 0 , 0 , 2 , 2 , 2 , 2);
    nn::implGemmgradweight<float>(gradkernel.get() , input.get() , gradout.get() , 2 , 2 , 5 , 5 , 2 , 3 , 3 , 0 , 0 , 2 , 2 , 2 , 2);  

    gradinput.print();
    gradkernel.print();
}
