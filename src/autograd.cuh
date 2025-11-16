#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_
#include "tensor.cuh"
#include "nn.cuh"
#include <set>
#include <map>
namespace mytorch{
    namespace autograd{
        template <typename T>
        void dfs_topo_sort(const Tensor<T> & root, std::set<Tensor<T> , less_addr<T>>& visited_nodes , std::vector<Tensor<T>> & topo_order){
            if(visited_nodes.find(root) != visited_nodes.end())
                return;
            visited_nodes.insert(root);
            auto fn = root.get_grad_fn();
            if(fn){
                auto inputs = fn->get_inputs();
                for(auto & input : inputs)
                    dfs_topo_sort(input , visited_nodes , topo_order);
            }
            topo_order.push_back(root);
        }

        template <typename T>
        std::vector<Tensor<T>> find_topo_sort(const Tensor<T> & root){
            std::vector<Tensor<T>> topo_order;
            std::set<Tensor<T> , less_addr<T>> visited_nodes;
            dfs_topo_sort(root , visited_nodes , topo_order);
            return topo_order;
        }

        template <typename T>
        void compute_gradients_of_variables(Tensor<T> & root ,const Tensor<T> & output_grad){
            std::map<Tensor<T> , Tensor<T> , less_addr<T>> node_to_output_grads_dict;
            node_to_output_grads_dict[root] = output_grad.deepcopy();
            auto topo_order = find_topo_sort(root);
            std::reverse(topo_order.begin() , topo_order.end());
            for(auto & node : topo_order){
                node.set_grad(node_to_output_grads_dict[node].deepcopy());
                auto grad_fn = node.get_grad_fn();
                if(grad_fn == nullptr)
                    continue;
                auto inputs = grad_fn->get_inputs();
                auto input_grads = grad_fn->backward(node_to_output_grads_dict[node]);
                for(int i = 0 ; i < input_grads.size() ; i++){
                    if(node_to_output_grads_dict.find(inputs[i]) == node_to_output_grads_dict.end())
                        node_to_output_grads_dict[inputs[i]] = input_grads[i];
                    else
                        node_to_output_grads_dict[inputs[i]] = node_to_output_grads_dict[inputs[i]] +  input_grads[i];
                }
            }


        }
    }
}

#endif