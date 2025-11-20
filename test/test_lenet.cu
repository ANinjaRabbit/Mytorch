#include "../src/tensor.cuh"
#include "../src/nn.cuh"
#include "../src/autograd.cuh"
#include "../src/optim.cuh"
using namespace mytorch;



int main(){
    DefaultDevice = Cuda;
    auto X = randn<float>({10 , 10});
    X.set_requires_grad(true);
    auto y = randn<float>({10 , 1});
    auto model = nn::Linear<float>(10 , 10);
    auto model2 = nn::Linear<float>(10 , 1);
    std::vector<Tensor<float>> params;
    auto params1 = model.parameters();
    auto params2 = model2.parameters();
    params.insert(params.end() , params1.begin() , params1.end());
    params.insert(params.end() , params2.begin() , params2.end());

    auto optimizer = optim::SGD<float>(params , 0.001);
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i = 0 ; i < 100 ; i++){
        std::cout << "Iteration " << i << " ";
        auto y_pred = model2(model(X));
        auto loss = (y_pred - y);
        loss = loss * loss;
        loss = loss.sum(1);
        loss.zero_grad();
        loss.backward();
        std::cout << "loss " << loss.item() << "\n";
        optimizer.step();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
}
