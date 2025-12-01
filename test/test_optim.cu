#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;




int main(){
    auto adam = std::make_shared<optim::Adam<float>>(std::vector<Tensor<float>>(), 0.001);
    int epochs = 100;
    optim::lr_scheduler::CosineAnnealingLR<float> scheduler(adam , epochs/2 , 0.0001);
    for(int i = 0 ; i < epochs ; i++){
        std::cout << "Epoch " << i << " lr " << adam->lr_ << "\n";
        scheduler.step();
    }
}
