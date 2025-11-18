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
    auto weight = randn<float>({1 , 10});
    auto bias = randn<float>({1});
    auto model = nn::Linear<float>(weight , bias);
    auto optimizer = optim::SGD<float>(model.parameters() , 0.01 );
    for(int i = 0 ; i < 10 ; i++){
        auto y_pred = model(X);
        auto loss = (y_pred - y);
        loss = loss * loss;
        loss = loss.sum(1);
        loss.zero_grad();
        loss.backward();
        loss.print();
        optimizer.step();
    }
}
