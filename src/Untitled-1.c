
    template <>
    class Conv<double> : public Module<double>{
        private:
            Tensor<double> kernel_;
            Tensor<double> input_cache;
            PaddingMode padding_mode;
        public:

            Conv(const Tensor<double> & kernel , PaddingMode padding_mode = ZeroPadding)
            : kernel_(kernel) , padding_mode(padding_mode) {}
            Tensor<double> forward(const std::vector<Tensor<double>> & inputs) override{
                auto input = inputs[0];
                // input :(N , C_in , ...) kernel (C_out , C_in , ...) result (N , C_out , ...)
                if(input.ndim() != kernel_.ndim()){
                    throw std::runtime_error("Conv: input and kernel must have the same number of dimensions");
                }
                if(input.shape()[1] != kernel_.shape()[1]){
                    throw std::runtime_error("Conv: input and kernel must have the same number of input channels");
                }
                if(input.requires_grad()){
                    input_cache = input;
                }
                kernel_.to(input.device());
                auto resultshape = input.shape();
                resultshape[1] = kernel_.shape()[0];

                if(padding_mode == NoPadding){
                    for(size_t i = 2;i<resultshape.size();i++){
                        resultshape[i] -= (kernel_.shape()[i] / 2) + 1;
                    }
                }

                Tensor<double> result(resultshape , input.device());
                size_t reduce_imsize = 1;

                for(size_t i = 2;i<resultshape.size();i++){
                    reduce_imsize *= resultshape[i];
                }

                std::vector<size_t> resultstride = result.get_strides();
                auto inputstride = input.get_strides();
                auto kernelstride = kernel_.get_strides();
                std::vector<size_t> single_kernel_shape(kernel_.shape().begin()+2 , kernel_.shape().end());
                std::vector<size_t> single_input_shape(input.shape().begin()+2 , input.shape().end());
                std::vector<size_t> single_input_stride(inputstride.begin() , inputstride.end() - 2);
                std::vector<size_t> single_output_stride(resultstride.begin() , resultstride.end() - 2);
                if(result.device() == Cuda){
                    //ready for pipeline
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    cudaStream_t stream_im2col, stream_gemm;
                    cudaStreamCreate(&stream_im2col);
                    cudaStreamCreate(&stream_gemm);

                    cudaEvent_t ev[2];
                    cudaEventCreate(&ev[0]);
                    cudaEventCreate(&ev[1]);

                    double * im2col_buf[2];

                    auto single_kernel_size = Functional::prod_vec(single_kernel_shape);
                    size_t single_input_size = Functional::prod_vec(single_input_shape);
                    size_t im2col_size = single_input_size * single_kernel_size * sizeof(double);

                    CHECK(cudaMalloc(&im2col_buf[0] , im2col_size));
                    CHECK(cudaMalloc(&im2col_buf[1] , im2col_size));

                    double alpha = 1.0f;
                    double beta = 1.0f;

                    size_t iter = 0;

                    //preprocessing
                    auto half_kernel_shape = single_kernel_shape;
                    auto instride = single_input_stride;
                    auto revinstride = instride;
                    std::reverse(revinstride.begin() , revinstride.end());
                    for(int i = 0;i<half_kernel_shape.size();i++){
                        half_kernel_shape[i] /= 2;
                    }
                    std::vector<size_t> kernel_stride = {};
                    size_t kernel_stride_ = 1;
                    for(int i = 0;i<single_kernel_shape.size();i++){
                        kernel_stride.push_back(kernel_stride_);
                        kernel_stride_ *= single_kernel_shape[single_kernel_shape.size() - 1 - i];
                    }
                    cuda_shared_pointer<size_t> kershape(single_kernel_shape , Cuda);
                    cuda_shared_pointer<size_t> imshape(single_input_shape ,Cuda);
                    size_t inputbatchoffset , resultbatchoffset , resultoutoffset , kerneloutoffset , inputinoffset , kernelinoffset;
                    size_t prev_kerneloutoffset , prev_kernelinoffset , prev_resultoutoffset , prev_resultbatchoffset;
                    if(padding_mode == ZeroPadding){
                        for(inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                            ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                            for(resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                for(inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    size_t cur = iter & 1;
                                    size_t prev = (iter + 1) & 1;

                                    Functional::im2col_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>
                                    (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                    single_kernel_size , single_input_shape.size()
                                    ,kershape.get() , imshape.get() ,single_input_size);
                                    CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                    
                                    if(iter > 0){
                                        cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                        cublasSetStream(handle , stream_gemm);
                                        CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                            1 , 
                                            single_input_size,
                                            single_kernel_size, 
                                            &alpha , 
                                            kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                            1, 
                                            im2col_buf[prev] ,
                                            single_kernel_size , 
                                            &beta , 
                                            result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                            1
                                        ));
                                    }
                                    iter++;
                                    prev_kerneloutoffset = kerneloutoffset;
                                    prev_kernelinoffset = kernelinoffset;
                                    prev_resultoutoffset = resultoutoffset;
                                    prev_resultbatchoffset = resultbatchoffset;
                                }
                            }
                        }

                        cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                        cublasSetStream(handle , stream_gemm);

                        cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            1 , 
                            single_input_size,
                            single_kernel_size, 
                            &alpha , 
                            kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                            1, 
                            im2col_buf[(iter + 1) & 1] ,
                            single_kernel_size , 
                            &beta , 
                            result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                            1
                        );
                    }
                    else{

                        for(inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                            ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                            for(resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                for(inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                    size_t cur = iter & 1;
                                    size_t prev = (iter + 1) & 1;

                                    Functional::im2col_gpu_nopadding<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>
                                    (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                    single_kernel_size , single_input_shape.size()
                                    ,kershape.get() , imshape.get() ,single_input_size , reduce_imsize);
                                    CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                    
                                    if(iter > 0){
                                        cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                        cublasSetStream(handle , stream_gemm);
                                        CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                                            1 , 
                                            reduce_imsize,
                                            single_kernel_size, 
                                            &alpha , 
                                            kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                                            1, 
                                            im2col_buf[prev] ,
                                            single_kernel_size , 
                                            &beta , 
                                            result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                                            1
                                        ));
                                    }
                                    iter++;
                                    prev_kerneloutoffset = kerneloutoffset;
                                    prev_kernelinoffset = kernelinoffset;
                                    prev_resultoutoffset = resultoutoffset;
                                    prev_resultbatchoffset = resultbatchoffset;
                                }
                            }
                        }

                        cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                        cublasSetStream(handle , stream_gemm);
                        CHECK_CUBLAS( cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N,
                            1 , 
                            reduce_imsize,
                            single_kernel_size, 
                            &alpha , 
                            kernel_.get() + prev_kerneloutoffset + prev_kernelinoffset  , 
                            1, 
                            im2col_buf[(iter + 1) & 1] ,
                            single_kernel_size , 
                            &beta , 
                            result.get() + prev_resultbatchoffset + prev_resultoutoffset , 
                            1
                        ));
                    }
                    CHECK(cudaStreamSynchronize(stream_gemm));
                    CHECK(cudaStreamSynchronize(stream_im2col));
                    CHECK(cudaStreamDestroy(stream_gemm));
                    CHECK(cudaStreamDestroy(stream_im2col));
                    cublasDestroy(handle);
                    CHECK(cudaFree(im2col_buf[0]));
                    CHECK(cudaFree(im2col_buf[1]));
                }
                else{
                    for(size_t inputbatchoffset = 0 , resultbatchoffset = 0;inputbatchoffset < input.size()
                        ;inputbatchoffset += inputstride[input.ndim() - 1] , resultbatchoffset += resultstride[result.ndim() - 1]){
                            for(size_t resultoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , resultoutoffset += resultstride[result.ndim() - 2]){
                                    for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                                    ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                        Tensor<double> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                        Functional::im2col_ptr<double>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                        ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                        for(size_t i = 0;i< inputtlide.shape()[0];i++){
                                            double sum = 0;
                                            for(size_t k = 0;k<inputtlide.shape()[1];k++){
                                                sum += inputtlide.get()[i * inputtlide.shape()[1] + k] * 
                                                kernel_.get()[kerneloutoffset+kernelinoffset+ k];
                                            }
                                            result.get()[resultbatchoffset + resultoutoffset + i] += sum;
                                        }
                                    }
                                }
                    }

                }
                if(input.requires_grad()){
                    result.set_requires_grad(true);
                    result.set_grad_fn(
                        std::make_shared<Functional::ModuleFunctionWrapper<double> >(this , input  ));
                }
                return result;
                    
            }
            std::vector<Tensor<double>> _internal_backward(const Tensor<double> & grad_out) override{
                // grad_out (N , C_out , ...) kernel (C_out , C_in , ...) input (N , C_in , ...)
                auto input = input_cache;
                // first calculating kernel 
                auto inputstride = input.get_strides();
                auto kernelstride = kernel_.get_strides();
                std::vector<size_t> outstride = grad_out.get_strides();
                std::vector<size_t> single_kernel_shape(kernel_.shape().begin()+2 , kernel_.shape().end());
                std::vector<size_t> single_input_shape(input.shape().begin()+2 , input.shape().end());
                std::vector<size_t> single_input_stride(inputstride.begin() , inputstride.end() - 2);
                std::vector<size_t> single_output_stride(outstride.begin() , outstride.end() - 2);
                Tensor<double>  grad_kernel(kernel_.shape() , kernel_.device());
                Tensor<double>  grad_input(input.shape() , input.device());
                if(grad_kernel.device() == Cuda){
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    cudaStream_t stream_im2col, stream_gemm;
                    cudaStreamCreate(&stream_im2col);
                    cudaStreamCreate(&stream_gemm);

                    cudaEvent_t ev[2];
                    cudaEventCreate(&ev[0]);
                    cudaEventCreate(&ev[1]);

                    double * im2col_buf[2];

                    auto single_kernel_size = Functional::prod_vec(single_kernel_shape);
                    size_t single_input_size = Functional::prod_vec(single_input_shape);
                    size_t im2col_size = single_input_size * single_kernel_size * sizeof(double);

                    CHECK(cudaMalloc(&im2col_buf[0] , im2col_size));
                    CHECK(cudaMalloc(&im2col_buf[1] , im2col_size));

                    double alpha = 1.0f;
                    double beta = 1.0f;

                    size_t iter = 0;

                    //preprocessing
                    auto half_kernel_shape = single_kernel_shape;
                    auto instride = single_input_stride;
                    auto revinstride = instride;
                    std::reverse(revinstride.begin() , revinstride.end());
                    for(int i = 0;i<half_kernel_shape.size();i++){
                        half_kernel_shape[i] /= 2;
                    }
                    std::vector<size_t> kernel_stride = {};
                    size_t kernel_stride_ = 1;
                    for(int i = 0;i<single_kernel_shape.size();i++){
                        kernel_stride.push_back(kernel_stride_);
                        kernel_stride_ *= single_kernel_shape[single_kernel_shape.size() - 1 - i];
                    }
                    cuda_shared_pointer<size_t> kershape(single_kernel_shape , Cuda);
                    cuda_shared_pointer<size_t> imshape(single_input_shape ,Cuda);
                    size_t prev_gradoutoffset = 0 , prev_gradbatchoffset = 0 , prev_kerneloutoffset = 0 , prev_kernelinoffset = 0;

                    if(padding_mode == ZeroPadding) {
                        for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                            ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                            for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                            ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                    ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                    size_t cur = iter & 1;
                                    size_t prev = (iter + 1) & 1;

                                    Functional::im2col_gpu_t<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>
                                    (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                    single_kernel_size , single_input_shape.size()
                                    ,kershape.get() , imshape.get() ,single_input_size);
                                    CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                    
                                    if(iter > 0){
                                        cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                        cublasSetStream(handle , stream_gemm);
                                        CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                            , 1 , single_kernel_size , single_input_size
                                            , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                            im2col_buf[prev] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                    }
                                    iter++;
                                    prev_kerneloutoffset = kerneloutoffset;
                                    prev_kernelinoffset = kernelinoffset;
                                    prev_gradoutoffset = gradoutoffset;
                                    prev_gradbatchoffset = gradbatchoffset;
                                }
                            }
                        }
                        cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                        cublasSetStream(handle , stream_gemm);
                        CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                            , 1 , single_kernel_size , single_input_size
                            , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                            im2col_buf[(iter + 1) & 1] , single_input_size  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                        kernel_.set_grad(grad_kernel);
                        size_t prev_inputbatchoffset = 0 , prev_inputinoffset = 0;
                        iter = 0;

                        for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                            ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                            for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                            ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){

                                    size_t cur = iter & 1;
                                    size_t prev = (iter + 1) & 1;

                                    CHECK(cudaMemset(im2col_buf[cur] , 0 , im2col_size));
                                    CHECK_CUBLAS(cublasSetStream(handle , stream_gemm));
                                    cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                        ,single_kernel_size , single_input_size , 1
                                        , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                        grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta ,
                                        im2col_buf[cur] , single_kernel_size
                                    );
                                    CHECK(cudaEventRecord(ev[cur] , stream_gemm));

                                    if(iter > 0){
                                        cudaStreamWaitEvent(stream_im2col , ev[prev] , 0);
                                        Functional::col2im_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                                        , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                                        ,kershape.get() , imshape.get() , single_input_size);
                                    }

                                    iter++;
                                    prev_inputbatchoffset = inputbatchoffset;
                                    prev_inputinoffset = inputinoffset;
                                }
                            }
                        }

                        cudaStreamWaitEvent(stream_im2col , ev[(iter + 1) & 1] , 0);
                        Functional::col2im_gpu<<<CudaGetBlocks(single_input_size) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                        , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                        ,kershape.get() , imshape.get() , single_input_size);

                    }
                    else{
                        size_t reduce_imsize = 1;

                        for(size_t i = 2;i<grad_out.shape().size();i++){
                            reduce_imsize *= grad_out.shape()[i];
                        }
                        for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                            ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                            for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                            ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                    ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                    size_t cur = iter & 1;
                                    size_t prev = (iter + 1) & 1;

                                    Functional::im2col_gpu_nopadding_t<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>
                                    (im2col_buf[cur] , input.get() + inputbatchoffset + inputinoffset ,
                                    single_kernel_size , single_input_shape.size()
                                    ,kershape.get() , imshape.get() ,single_input_size , reduce_imsize);
                                    CHECK(cudaEventRecord(ev[cur] , stream_im2col)) ;
                                    
                                    if(iter > 0){
                                        cudaStreamWaitEvent(stream_gemm , ev[prev] , 0);
                                        cublasSetStream(handle , stream_gemm);
                                        CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                                            , 1 , single_kernel_size , reduce_imsize
                                            , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                                            im2col_buf[prev] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                                    }
                                    iter++;
                                    prev_kerneloutoffset = kerneloutoffset;
                                    prev_kernelinoffset = kernelinoffset;
                                    prev_gradoutoffset = gradoutoffset;
                                    prev_gradbatchoffset = gradbatchoffset;
                                }
                            }
                        }
                        cudaStreamWaitEvent(stream_gemm , ev[(iter + 1) & 1] , 0);
                        cublasSetStream(handle , stream_gemm);
                        CHECK_CUBLAS(cublasSgemm(handle , CUBLAS_OP_N , CUBLAS_OP_N
                            , 1 , single_kernel_size , reduce_imsize
                            , &alpha , grad_out.get() + prev_gradbatchoffset + prev_gradoutoffset, 1 , 
                            im2col_buf[(iter + 1) & 1] , reduce_imsize  , &beta , grad_kernel.get() + prev_kerneloutoffset + prev_kernelinoffset , 1));
                        kernel_.set_grad(grad_kernel);
                        size_t prev_inputbatchoffset = 0 , prev_inputinoffset = 0;
                        iter = 0;

                        for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                            ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                            for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                            ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                                for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                                ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){

                                    size_t cur = iter & 1;
                                    size_t prev = (iter + 1) & 1;

                                    CHECK(cudaMemset(im2col_buf[cur] , 0 , im2col_size));
                                    CHECK_CUBLAS(cublasSetStream(handle , stream_gemm));
                                    cublasSgemm(handle , CUBLAS_OP_T , CUBLAS_OP_N
                                        ,single_kernel_size , reduce_imsize , 1
                                        , &alpha , kernel_.get() + kerneloutoffset + kernelinoffset , 1,
                                        grad_out.get() + gradbatchoffset + gradoutoffset , 1 , &beta ,
                                        im2col_buf[cur] , single_kernel_size
                                    );
                                    CHECK(cudaEventRecord(ev[cur] , stream_gemm));

                                    if(iter > 0){
                                        cudaStreamWaitEvent(stream_im2col , ev[prev] , 0);
                                        Functional::col2im_gpu_nopadding<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                                        , im2col_buf[prev] , single_kernel_size , single_input_shape.size()
                                        ,kershape.get() , imshape.get() , single_input_size , reduce_imsize);
                                    }

                                    iter++;
                                    prev_inputbatchoffset = inputbatchoffset;
                                    prev_inputinoffset = inputinoffset;
                                }
                            }
                        }

                        cudaStreamWaitEvent(stream_im2col , ev[(iter + 1) & 1] , 0);
                        Functional::col2im_gpu_nopadding<<<CudaGetBlocks(reduce_imsize) , kCudaThreadsNum , 0 , stream_im2col>>>(grad_input.get() + prev_inputbatchoffset + prev_inputinoffset 
                        , im2col_buf[(iter + 1) & 1] , single_kernel_size , single_input_shape.size()
                        ,kershape.get() , imshape.get() , single_input_size , reduce_imsize);

                    }

                    CHECK(cudaStreamSynchronize(stream_im2col));
                    CHECK(cudaStreamSynchronize(stream_gemm));
                    CHECK(cudaStreamDestroy(stream_gemm));
                    CHECK(cudaStreamDestroy(stream_im2col));
                    cublasDestroy(handle);
                    CHECK(cudaFree(im2col_buf[0]));
                    CHECK(cudaFree(im2col_buf[1]));
                }
                else{
                    for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                        ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                        for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                        ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                            for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                                ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                                    Tensor<double> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                    Functional::im2col_ptr<double>(inputtlide.get() , input.get() + inputbatchoffset + inputinoffset
                                        ,single_input_shape , single_kernel_shape ,  single_input_stride , single_output_stride , input.device());
                                    for(size_t i = 0;i< inputtlide.shape()[1];i++){
                                        double sum = 0;
                                        for(size_t k = 0;k<inputtlide.shape()[0];k++){
                                            sum += inputtlide.get()[i + inputtlide.shape()[1] * k] * 
                                            grad_out.get()[gradbatchoffset + gradoutoffset + k];
                                        }
                                        grad_kernel.get()[kerneloutoffset + kernelinoffset + i] += sum;
                                    }
                            }
                        }
                    }

                    kernel_.set_grad(grad_kernel);
                    for(size_t inputbatchoffset = 0 , gradbatchoffset = 0;inputbatchoffset < input.size()
                        ;inputbatchoffset += inputstride[input.ndim() - 1] , gradbatchoffset += outstride[outstride.size() - 1]){
                        for(size_t inputinoffset = 0 , kernelinoffset = 0;kernelinoffset < kernelstride[kernelstride.size() - 1]
                        ;inputinoffset += inputstride[input.ndim() - 2] , kernelinoffset += kernelstride[kernelstride.size() - 2]){
                            for(size_t gradoutoffset = 0 , kerneloutoffset = 0;kerneloutoffset < kernel_.size()
                            ;kerneloutoffset += kernelstride.back() , gradoutoffset += outstride[outstride.size() - 2]){
                                Tensor<double> inputtlide({inputstride[input.ndim() - 2] ,Functional::prod_vec(single_kernel_shape)} , input.device()) ;
                                    for(size_t i = 0;i<inputtlide.shape()[0];i++){
                                        for(size_t j = 0;j < inputtlide.shape()[1];j++){
                                            inputtlide.get()[i * inputtlide.shape()[1] + j] += 
                                            grad_out.get()[gradbatchoffset + gradoutoffset + i] * 
                                            kernel_.get()[kerneloutoffset + kernelinoffset + j];
                                        }
                                    }
                                Functional::col2im_ptr(grad_input.get() + inputbatchoffset + inputinoffset , inputtlide.get()
                                , single_input_shape , single_kernel_shape , single_input_stride , single_output_stride , input.device());
                            }
                        }
                    }

                }
                return {grad_input};
            }
            std::vector<Tensor<double>> parameters() override{
                return {kernel_};
            }
    };