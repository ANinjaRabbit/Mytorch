#include "nn.cuh"

namespace mytorch{

namespace nn{
    template <typename T , size_t N>
    __device__ T warpReduceMax(T val) {
        #pragma unroll
        for (int offset = N/2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        return val;
    }

    template <typename T , size_t N>
    __device__ T warpReduceSum(T val) {
        #pragma unroll
        for (int offset = N/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        return val;
    }

    template __device__ float warpReduceMax<float,32>(float val);
    template __device__ float warpReduceSum<float,32>(float val);
     template __device__ double warpReduceMax<double,32>(double val);
    template __device__ double warpReduceSum<double,32>(double val);

    __global__ void _softmax_kernel_small_512f(float * output , const float * input ,const int N, const int C){
        // for smaller than 512 size softmax
        extern __shared__ float shared[];
        int idx = blockIdx.x; // N
        int tid = threadIdx.x; // C
        int warpId = tid / 32;
        int laneId = tid % 32;

        constexpr size_t warpsPerBlock = kCudaThreadsNum / 32; // 1024 / 32 = 32

        float * maxvals = shared;
        float * sumvals = &shared[warpsPerBlock];

        const float * x = input + idx * C; // the row to process
        float maxval = tid < C ? x[tid] : -FLT_MAX;
        maxval = warpReduceMax<float>(maxval); // get the warp maximum
        if(laneId== 0) {
            maxvals[warpId] = maxval;
        }
        __syncthreads();
        if(tid < warpsPerBlock){
            maxval = maxvals[tid];
            maxval = warpReduceMax<float , warpsPerBlock>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
        }
        maxval = maxvals[0];
        if(tid < C){
            float val = expf(x[tid] - maxval);
            output[tid + idx * C] = val == 0 ? 1e-8 : val; // compute exp(x - max)
        }
        __syncthreads();
        float sum = tid < C ? output[tid + idx * C] : 0.0f;
        sum = warpReduceSum<float>(sum); // get the warp sum
        if(laneId== 0) {
            sumvals[warpId] = sum;
        }
        __syncthreads();
        if(tid < warpsPerBlock){
            sum = sumvals[tid];
            sum = warpReduceSum<float , warpsPerBlock>(sum);
            if(tid == 0){
                sumvals[0] = sum;
            }
        } // get the block sum
        sum = sumvals[0];
        if(tid < C){
            output[tid + idx * C] /= sum; // normalize
        }
    }

    __global__ void _softmax_kernel_small_512d(double * output , const double * input ,const int N, const int C){
        // for smaller than 512 size softmax
        extern __shared__ double sharedd[];
        int idx = blockIdx.x; // N
        int tid = threadIdx.x; // C
        int warpId = tid / 32;
        int laneId = tid % 32;

        constexpr size_t warpsPerBlock = kCudaThreadsNum / 32; // 1024 / 32 = 32

        double * maxvals = sharedd;
        double * sumvals = &sharedd[warpsPerBlock];

        const double * x = input + idx * C; // the row to process
        double maxval = tid < C ? x[tid] : -DBL_MAX;
        maxval = warpReduceMax<double>(maxval); // get the warp maximum
        if(laneId== 0) {
            maxvals[warpId] = maxval;
        }
        __syncthreads();
        if(tid < warpsPerBlock){
            maxval = maxvals[tid];
            maxval = warpReduceMax<double , warpsPerBlock>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
        } // get the block maximum
        maxval = maxvals[0];
        if(tid < C){
            output[tid + idx * C] = exp(x[tid] - maxval); // compute exp(x - max)
        }
        __syncthreads();
        double sum = tid < C ? output[tid + idx * C] : 0.0;
        sum = warpReduceSum<double>(sum); // get the warp sum
        if(laneId== 0) {
            sumvals[warpId] = sum;
        }
        __syncthreads();
        if(tid < warpsPerBlock){
            sum = sumvals[tid];
            sum = warpReduceSum<double , warpsPerBlock>(sum);
            if(tid == 0){
                sumvals[0] = sum;
            }
        } // get the block sum
        sum = sumvals[0];
        if(tid < C){
            output[tid + idx * C] /= sum; // normalize
        }
    }

    __global__ void _softmax_kernel_general_f(float * output , const float * input , const int N , const int C){
        extern __shared__ float shared[];
        int idx = blockIdx.x;
        int tid = threadIdx.x;
        int warpId = threadIdx.x / 32; 
        int laneId = threadIdx.x % 32;


        constexpr size_t warpsPerBlock = kCudaThreadsNum / 32;

        float* maxvals = shared;
        float* sumvals = &shared[warpsPerBlock];

        const float* x = input + idx * C;

        float maxval = -FLT_MAX;
        for (int i = tid; i < C; i += blockDim.x) {
            maxval = fmaxf(maxval, x[i]); 
        }
        maxval = warpReduceMax<float>(maxval);
        if (laneId == 0) 
            maxvals[warpId] = maxval;
        __syncthreads();
        if (tid < warpsPerBlock) {
            maxval = maxvals[tid];
            maxval = warpReduceMax<float , warpsPerBlock>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
            // store the final max in the first position
        }
        __syncthreads();
        maxval = maxvals[0];
        float sum = 0.0f;
        for (int i = tid; i < C; i += blockDim.x) {
            output[i + idx * C] = expf(x[i] - maxval);
            sum += output[i + idx * C];
        }   
        __syncthreads();
        sum = warpReduceSum<float>(sum);
        if( laneId == 0 ) 
            sumvals[warpId] = sum;
        __syncthreads();
        if (tid < warpsPerBlock) {
            sum = sumvals[tid];
            sum = warpReduceSum<float , warpsPerBlock>(sum);
            if(tid == 0){
                sumvals[0] = sum;
            }
        }
        __syncthreads();
        sum = sumvals[0];
        for (int i = tid; i < C; i += blockDim.x) {
            output[i + idx * C] /= sum;
        }
        
    }

    __global__ void _softmax_kernel_general_d(double * output , const double * input , const int N , const int C){
        extern __shared__ double sharedd[];
        int idx = blockIdx.x;
        int tid = threadIdx.x;
        int warpId = threadIdx.x / 32; 
        int laneId = threadIdx.x % 32;

        constexpr size_t warpsPerBlock = kCudaThreadsNum / 32;

        double* maxvals = sharedd;
        double* sumvals = &sharedd[warpsPerBlock];

        const double* x = input + idx * C;

        double maxval = -FLT_MAX;
        for (int i = tid; i < C; i += blockDim.x) {
            maxval = fmax(maxval, x[i]); 
        }
        maxval = warpReduceMax<double>(maxval);
        if (laneId == 0) 
            maxvals[warpId] = maxval;
        __syncthreads();
        if (tid < warpsPerBlock) {
            maxval = maxvals[tid];
            maxval = warpReduceMax<double , warpsPerBlock>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
            // store the final max in the first position
        }
        __syncthreads();
        maxval = maxvals[0];
        double sum = 0.0f;
        for (int i = tid; i < C; i += blockDim.x) {
            output[i + idx * C] = exp(x[i] - maxval);
            sum += output[i + idx * C];
        }   
        __syncthreads();
        sum = warpReduceSum<double>(sum);
        if( laneId == 0 ) 
            sumvals[warpId] = sum;
        __syncthreads();
        if (tid < warpsPerBlock) {
            sum = sumvals[tid];
            sum = warpReduceSum<double , warpsPerBlock>(sum);
            if(tid == 0){
                sumvals[0] = sum;
            }
        }
        __syncthreads();
        sum = sumvals[0];
        for (int i = tid; i < C; i += blockDim.x) {
            output[i + idx * C] /= sum;
        }
        
    }


    template<typename T , bool trans>
    __global__ void im2col_gpu_2d(T * col , const T * im  , const size_t kernel_size 
        , const size_t kh , const size_t kw ,  const size_t h , const size_t w
        , const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < h * w){
            size_t ix, iy;
            ix = index % w;
            iy = index / w;
            size_t m0 , m1;
            m0 = iy - (kh >> 1);
            m1 = ix - (kw >> 1);
            size_t grid_offset =  0;
            size_t col_offset;
            if constexpr (trans){
                col_offset = index;
            }
            else{
                col_offset = index * kernel_size;
            }
            size_t grid_index0 , grid_index1 , hw = h * w;
            for(size_t coffset = 0;coffset < imsize; coffset += hw){
                for(grid_index0 = 0;grid_index0 < kh;grid_index0++){
                    for(grid_index1 = 0;grid_index1 < kw;grid_index1++)
                    {
                        bool is_valid = true;
                        size_t k0 = m0 + grid_index0;
                        size_t k1 = m1 + grid_index1;
                        is_valid = k0 < h && k1 < w;
                        size_t im_offset = k0 * w + k1;
                        col[col_offset] =  is_valid ? im[im_offset + coffset] : 0;
                        grid_offset++;
                        if constexpr (trans){
                            col_offset += hw;
                        }
                        else{
                            col_offset++;
                        }
                    }
                }
            }

        }
    }
    template
    __global__ void im2col_gpu_2d<float , false>(float * col , const float * im  , const size_t kernel_size 
        , const size_t kh , const size_t kw ,  const size_t h , const size_t w
        , const size_t imsize);
    template
    __global__ void im2col_gpu_2d<float , true>(float * col , const float * im  , const size_t kernel_size 
        , const size_t kh , const size_t kw ,  const size_t h , const size_t w
        , const size_t imsize);



    template<typename T , bool trans>
    __global__ void im2col_gpu_nopadding_2d(T * col , const T * im , 
        const size_t kernel_size ,
        const size_t kh , const size_t kw ,  const size_t h , const size_t w , const size_t rhw
        , const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < rhw){
            size_t m0 , m1; // index in image
            size_t reduce_w = w - (kw - 1); // assume kw is odd
            m1 = index % reduce_w;
            m0 = index / reduce_w;


            size_t grid_offset =  0;
            size_t col_offset;
            if constexpr (trans){
                col_offset = index;
            }
            else {
                col_offset = index * kernel_size;
            }
            size_t hw = h * w;
            for(size_t coffset = 0 ; coffset < imsize ; coffset += hw){
                for(size_t g0 = 0;g0 < kh ; g0 ++ ){
                    for(size_t g1 = 0;g1 < kw; g1 ++){
                        size_t ix = m0 + g0;
                        size_t iy = m1 + g1;
                        size_t im_offset = ix * w + iy;
                        col[col_offset] =  im[im_offset + coffset];
                        grid_offset++;
                        if constexpr (trans){
                            col_offset += rhw;
                        }
                        else {
                            col_offset++;
                        }
                    } 
                }
            }

        }
    }

     template
    __global__ void im2col_gpu_nopadding_2d<float , false>(float * col , const float * im , 
        const size_t kernel_size ,
        const size_t kh , const size_t kw ,  const size_t h , const size_t w , const size_t rhw
        , const size_t imsize);
     template
    __global__ void im2col_gpu_nopadding_2d<float , true>(float * col , const float * im , 
        const size_t kernel_size ,
        const size_t kh , const size_t kw ,  const size_t h , const size_t w , const size_t rhw
        , const size_t imsize);


    template<typename T>
    __global__ void col2im_gpu_2d(T * im , const T * col  , const size_t kernel_size 
        , const size_t kh , const size_t kw ,  const size_t h , const size_t w
        , const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < h * w){
            size_t ix, iy;
            ix = index % w;
            iy = index / w;
            size_t m0 , m1;
            m0 = iy - (kh >> 1);
            m1 = ix - (kw >> 1);
            size_t grid_offset =  0;
            size_t col_offset;
            col_offset = index * kernel_size;
            size_t grid_index0 , grid_index1 , hw = h * w;
            for(size_t coffset = 0;coffset < imsize; coffset += hw){
                for(grid_index0 = 0;grid_index0 < kh;grid_index0++){
                    for(grid_index1 = 0;grid_index1 < kw;grid_index1++)
                    {
                        bool is_valid = true;
                        size_t k0 = m0 + grid_index0;
                        size_t k1 = m1 + grid_index1;
                        is_valid = k0 < h && k1 < w;
                        size_t im_offset = k0 * w + k1;
                        if(is_valid){
                            atomicAdd( im + im_offset + coffset , col[col_offset]);
                        }

                        grid_offset++;
                        col_offset++;
                    }
                }
            }

        }
    }
     template
    __global__ void col2im_gpu_2d<float>(float * im , const float * col  , const size_t kernel_size 
        , const size_t kh , const size_t kw ,  const size_t h , const size_t w
        , const size_t imsize);


    template<typename T>
    __global__ void col2im_gpu_nopadding_2d(T * im , const T * col , 
        const size_t kernel_size ,
        const size_t kh , const size_t kw ,  const size_t h , const size_t w , const size_t rhw
        , const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < rhw){
            size_t m0 , m1; // index in image
            size_t reduce_imshape = w - (kw - 1); // assume kw is odd
            m1 = index % reduce_imshape;
            m0 = index / reduce_imshape;

            size_t grid_offset =  0;
            size_t col_offset;
            col_offset = index * kernel_size;
            size_t hw = h * w;
            for(size_t coffset = 0 ; coffset < imsize ; coffset += hw){
                for(size_t g0 = 0;g0 < kh ; g0 ++ ){
                    for(size_t g1 = 0;g1 < kw; g1 ++){
                        size_t ix = m0 + g0;
                        size_t iy = m1 + g1;
                        size_t im_offset = ix * w + iy;
                        atomicAdd(im + im_offset + coffset , col[col_offset]);
                        grid_offset++;
                        col_offset++;
                    } 
                }
            }

        }
    }

    template
    __global__ void col2im_gpu_nopadding_2d<float>(float * im , const float * col , 
        const size_t kernel_size ,
        const size_t kh , const size_t kw ,  const size_t h , const size_t w , const size_t rhw
        , const size_t imsize);

    template<typename T , bool trans>
    __global__ void im2col_gpu(T * col , const T * im , const size_t kernel_size , const size_t ndim , const size_t * kernel_shape ,  const size_t * imshape , const size_t hw ,  const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < hw){
            size_t imidx[kCudaMultiDimMax];
            for(int i = 0 , index_ = index;i<ndim;i++){
                imidx[i] = index_ % imshape[ndim - i - 1];
                index_ /= imshape[ndim - i - 1];
            }
            size_t grid_min[kCudaMultiDimMax];
            for(int i = 0;i<ndim;i++){
                grid_min[i] = imidx[ndim - i - 1] - kernel_shape[i] / 2;
            }
            size_t grid_offset =  0;
            size_t col_offset;
            if constexpr (trans){
                col_offset = index;
            }
            else{
                col_offset = index * kernel_size;
            }
            for(size_t coffset = 0 ; coffset < imsize ; coffset += hw){
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    bool is_valid = true;
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                        if( kernel_index[i] >= imshape[i]){
                            is_valid = false;
                            break;
                        }
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    col[col_offset] =  is_valid ? im[im_offset + coffset] : 0;
                    grid_index.next();
                    grid_offset++;
                    if constexpr (trans){
                        col_offset += hw;
                    }
                    else{
                        col_offset++;
                    }

                }while(!grid_index.is_zero());
            }

        }
    }


    template<typename T , bool trans>
    __global__ void im2col_gpu_nopadding(T * col , const T * im , 
        const size_t kernel_size , const size_t ndim , 
        const size_t * kernel_shape ,  const size_t * imshape , const size_t rhw , const size_t hw,
        const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < rhw){
            size_t grid_min[kCudaMultiDimMax]; // index in image
            for(size_t i = 0 , index_ = index;i<ndim;i++){
                size_t reduce_imshape = imshape[ndim - i - 1] - (kernel_shape[ndim - i - 1] >> 1) - 1;
                grid_min[ndim - 1 - i] = index_ % reduce_imshape;
                index_ /= reduce_imshape;
            }

            size_t grid_offset =  0;
            size_t col_offset;
            if constexpr (trans){
                col_offset = index;
            }
            else{
                col_offset = index * kernel_size;
            }

            for(size_t coffset = 0 ; coffset < imsize ; coffset += hw){
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    col[col_offset] =  im[im_offset];
                    grid_index.next();
                    grid_offset++;
                    if constexpr (trans){
                        col_offset += rhw;
                    }
                    else{
                        col_offset++;
                    }
                }while(!grid_index.is_zero());
            }

        }
    }

    template<typename T>
    __global__ void col2im_gpu(T * im , const T * col  , const size_t kernel_size 
        , const size_t ndim , const size_t * kernel_shape ,  const size_t * imshape  , const size_t hw
        , const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < hw){
            size_t imidx[kCudaMultiDimMax];
            for(int i = 0 , index_ = index;i<ndim;i++){
                imidx[i] = index_ % imshape[ndim - i - 1];
                index_ /= imshape[ndim - i - 1];
            }
            size_t grid_min[kCudaMultiDimMax];
            for(int i = 0;i<ndim;i++){
                grid_min[i] = imidx[ndim - i - 1] - kernel_shape[i] / 2;
            }
            size_t grid_offset =  0;
            size_t col_offset = index * kernel_size;
            for(size_t coffset = 0 ; coffset < imsize ; coffset += hw){
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    bool is_valid = true;
                    size_t kernel_index[kCudaMultiDimMax];
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                        if( kernel_index[i] >= imshape[i]){
                            is_valid = false;
                            break;
                        }
                    }
                    if(is_valid){
                        size_t im_offset = 0;
                        for(int i = 0;i<ndim;i++){
                            im_offset *= imshape[i];
                            im_offset += kernel_index[i];
                        }
                        atomicAdd( &im[im_offset + coffset] , col[col_offset] );
                    }
                    grid_index.next();
                    grid_offset++;
                    col_offset++;

                }while(!grid_index.is_zero());
            }

        }
    }
    template<typename T>
    __global__ void col2im_gpu_nopadding(T * im , const T * col , 
        const size_t kernel_size , const size_t ndim , 
        const size_t * kernel_shape ,  const size_t * imshape , const size_t rhw, const size_t hw,
        const size_t imsize){
        size_t index = threadIdx.x + blockDim.x * blockIdx.x;
        if(index < rhw){
            size_t grid_min[kCudaMultiDimMax]; // index in image
            for(size_t i = 0 , index_ = index;i<ndim;i++){
                size_t reduce_imshape = imshape[ndim - i - 1] - (kernel_shape[ndim - i - 1] >> 1) - 1;
                grid_min[ndim - 1 - i] = index_ % reduce_imshape;
                index_ /= reduce_imshape;
            }

            size_t grid_offset =  0;
            size_t col_offset = index * kernel_size;
            for(size_t coffset = 0 ; coffset < imsize ; coffset += hw){
                size_t kernel_index[kCudaMultiDimMax];
                CudaMultiDimIndex grid_index(kernel_shape , ndim);
                do{
                    for(int i = 0;i<ndim;i++){
                        kernel_index[i] = grid_min[i] + grid_index.get_index()[i]; 
                    }
                    size_t im_offset = 0;
                    for(int i = 0;i<ndim;i++){
                        im_offset *= imshape[i];
                        im_offset += kernel_index[i];
                    }
                    atomicAdd( &im[im_offset + coffset] , col[col_offset] );
                    grid_index.next();
                    grid_offset++;
                    col_offset++;

                }while(!grid_index.is_zero());
            }

        }
    }




    namespace Functional {
        __global__ void _sum_forward_kernel_f(float * output , const float * input ,  const size_t reduce , const size_t inner){
            size_t ridx = threadIdx.x;
            size_t iidx = blockIdx.x % inner;
            size_t oidx = blockIdx.x / inner;
            extern __shared__ float smem_sum_f[];
            size_t warpId = ridx / 32;
            size_t laneId = ridx % 32;

            constexpr size_t warpsPerBlock = kCudaThreadsNum / 32;
            float sum = 0.0f;
            for(size_t i = ridx;i<reduce;i+=blockDim.x){
                sum += input[oidx * reduce + i * inner + iidx];
            }
            sum = warpReduceSum<float>(sum);
            if(laneId == 0) smem_sum_f[warpId] = sum;
            __syncthreads();
            if(ridx < warpsPerBlock){
                sum = smem_sum_f[ridx];
                sum = warpReduceSum<float , warpsPerBlock>(sum);
                if(ridx == 0){
                    output[oidx * reduce + iidx] = sum;
                }
            }

        }
        __global__ void _sum_forward_kernel_d(double * output , const double * input ,  const size_t reduce , const size_t inner){
            size_t ridx = threadIdx.x;
            size_t iidx = blockIdx.x % inner;
            size_t oidx = blockIdx.x / inner;
            extern __shared__ double smemd[];
            size_t warpId = ridx / 32;
            size_t laneId = ridx % 32;
            constexpr size_t warpsPerBlock = kCudaThreadsNum / 32;
            double sum = 0.0f;
            for(size_t i = ridx;i<reduce;i+=blockDim.x){
                sum += input[oidx * reduce + i * inner + iidx];
            }
            sum = warpReduceSum<double>(sum);
            if(laneId == 0) smemd[warpId] = sum;
            __syncthreads();
            if(ridx < warpsPerBlock){
                sum = smemd[ridx];
                sum = warpReduceSum<double , warpsPerBlock>(sum);
                if(ridx == 0){
                    output[oidx * reduce + iidx] = sum;
                }
            }
        }

    }

}

    
} // namespace mytorch