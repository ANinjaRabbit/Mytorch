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
    auto w1 = randn<float>({10 , 10});
    auto b1 = randn<float>({10});
    auto model = nn::Linear<float>(w1 , b1);
    auto w2 = randn<float>({1 , 10});
    auto b2 = randn<float>({1});
    auto model2 = nn::Linear<float>(w2 , b2);
    auto optimizer = optim::Adam<float>(model.parameters() , 0.01);
    for(int i = 0 ; i < 100 ; i++){
        auto y_pred = model2(model(X));
        auto loss = (y_pred - y);
        loss = loss * loss;
        loss = loss.sum(1);
        loss.zero_grad();
        loss.backward();
        loss.print();
        optimizer.step();
    }
}
