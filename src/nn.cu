#include "nn.cuh"

namespace mytorch{

namespace nn{
    template <typename T>
    __device__ T warpReduceMax(T val) {

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        return val;
    }

    template <typename T>
    __device__ T warpReduceSum(T val) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        return val;
    }

    template __device__ float warpReduceMax<float>(float val);
    template __device__ float warpReduceSum<float>(float val);
    template __device__ double warpReduceMax<double>(double val);
    template __device__ double warpReduceSum<double>(double val);

    __global__ void _softmax_kernel_small_512f(float * output , const float * input ,const int N, const int C){
        // for smaller than 512 size softmax
        extern __shared__ float shared[];
        int idx = blockIdx.x; // N
        int tid = threadIdx.x; // C
        int warpId = tid / 32;
        int laneId = tid % 32;

        constexpr int warpsPerBlock = kCudaThreadsNum / 32; // 1024 / 32 = 32

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
            maxval = warpReduceMax<float >(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
        }
        __syncthreads();
        maxval = maxvals[0];
        if(tid < C){
            float val = nn_exp_device<float>(x[tid] - maxval);
            output[tid + idx * C] = val; // compute exp(x - max)
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
            sum = warpReduceSum<float>(sum);
            if(tid == 0){
                sumvals[0] = sum < 1e-8 ? 1e-8 : sum;
            }
        } // get the block sum
        __syncthreads();
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

        constexpr int warpsPerBlock = kCudaThreadsNum / 32; // 1024 / 32 = 32

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
            maxval = warpReduceMax<double>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
        } // get the block maximum
        __syncthreads();
        maxval = maxvals[0];
        if(tid < C){
            output[tid + idx * C] = nn_exp_device<double>(x[tid] - maxval); // compute exp(x - max)
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
            sum = warpReduceSum<double>(sum);
            if(tid == 0){
                sumvals[0] = sum;
            }
        } // get the block sum
        __syncthreads();
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


        constexpr int warpsPerBlock = kCudaThreadsNum / 32;

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
            maxval = warpReduceMax<float>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
            // store the final max in the first position
        }
        __syncthreads();
        maxval = maxvals[0];
        float sum = 0.0f;
        for (int i = tid; i < C; i += blockDim.x) {
            output[i + idx * C] = nn_exp_device<float>(x[i] - maxval);
            sum += output[i + idx * C];
        }   
        __syncthreads();
        sum = warpReduceSum<float>(sum);
        if( laneId == 0 ) 
            sumvals[warpId] = sum;
        __syncthreads();
        if (tid < warpsPerBlock) {
            sum = sumvals[tid];
            sum = warpReduceSum<float>(sum);
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

        constexpr int warpsPerBlock = kCudaThreadsNum / 32;

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
            maxval = warpReduceMax<double>(maxval);
            if(tid == 0){
                maxvals[0] = maxval;    
            }
            // store the final max in the first position
        }
        __syncthreads();
        maxval = maxvals[0];
        double sum = 0.0f;
        for (int i = tid; i < C; i += blockDim.x) {
            output[i + idx * C] = nn_exp_device<double>(x[i] - maxval);
            sum += output[i + idx * C];
        }   
        __syncthreads();
        sum = warpReduceSum<double>(sum);
        if( laneId == 0 ) 
            sumvals[warpId] = sum;
        __syncthreads();
        if (tid < warpsPerBlock) {
            sum = sumvals[tid];
            sum = warpReduceSum<double>(sum);
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

    template <typename T , bool transpose> // transpose mode
    __global__ void im2col_gpu_2d(T * col , const T * im,
        const int n, const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    ){
        if constexpr (transpose){
        CUDA_KERNEL_LOOP(index , n){
            // launch c * height_col * width_col threads
            const int h_index = index / width_col;
            const int h_col = h_index  % height_col;
            const int c_im = h_index / height_col;
            const int w_col = index % width_col;
            const int h_offset = h_col * stride_h - pad_h;
            const int w_offset = w_col * stride_w - pad_w;
            const int c_col = c_im * kh * kw;
            T * col_ptr = col + (c_col * height_col + h_col) * width_col + w_col;
            const T * im_ptr = im + (c_im * height + h_offset) * width + w_offset;
            for(int i = 0; i < kh ; i++){
                for(int j = 0; j < kw; j++){
                    int h_im = i + h_offset;
                    int w_im = j + w_offset;
                    *col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                        im_ptr[i * width + j] : 0;

                    col_ptr += height_col * width_col;
                }
            }


        }
        }
        else{
        CUDA_KERNEL_LOOP(index , n){
            // launch c * height_col * width_col threads
            const int h_index = index / width_col;
            const int h_col = h_index  % height_col;
            const int c_im = h_index / height_col;
            const int w_col = index % width_col;
            const int h_offset = h_col * stride_h - pad_h;
            const int w_offset = w_col * stride_w - pad_w;
            const int ckernelsize = kh * kw * channels; // size of whole kernel
            T * col_ptr = col + (ckernelsize * (h_col * width_col + w_col) + c_im * kh * kw);
            const T * im_ptr = im + (c_im * height + h_offset) * width + w_offset;
            for(int i = 0; i < kh ; i++){
                for(int j = 0; j < kw; j++){
                    int h_im = i + h_offset;
                    int w_im = j + w_offset;
                    *col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                        im_ptr[i * width + j] : 0;
                    
                    col_ptr ++;
                }
            }
        }
            
        }
    }
    template
    __global__ void im2col_gpu_2d<float , true>(float * col , const float * im,
        const int n, const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template
    __global__ void im2col_gpu_2d<float , false>(float * col , const float * im,
        const int n, const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template
    __global__ void im2col_gpu_2d<double , true>(double * col , const double * im,
        const int n, const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template
    __global__ void im2col_gpu_2d<double , false>(double * col , const double * im,
        const int n, const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );




    template <typename T , bool transpose> // non transpose mode
    __global__ void col2im_gpu_2d(T * im , T * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    ){
        if constexpr (!transpose){
            CUDA_KERNEL_LOOP(index , n){ // n for im
                T val = 0;
                const int w_im = index % width + pad_w;
                const int h_im = (index / width) % height + pad_h;
                const int c_im = index / (height * width);
                const int ckernelsize = kh * kw * channels; // size of whole kernel

                const int w_col_start = (w_im < kw) ? 0 : (w_im - kw) /stride_w + 1;
                const int w_col_end = min(w_im / stride_w + 1 , width_col);
                const int h_col_start = (h_im < kh) ? 0 : (h_im - kh) /stride_h + 1;
                const int h_col_end = min(h_im / stride_h + 1 , height_col);

                for(int h_col = h_col_start; h_col < h_col_end; h_col++){
                    for(int w_col = w_col_start; w_col < w_col_end; w_col++){
                        int w_k = w_im - w_col * stride_w;
                        int h_k = h_im - h_col * stride_h;

                        int col_index = (h_col * width_col + w_col) * ckernelsize + (c_im * kh   +   h_k) * kw + w_k;
                        val += col[col_index];
                    }
                }
                im[index] = val;
            }
        }
        else{
            CUDA_KERNEL_LOOP(index , n){ // n for im
                T val = 0;
                const int w_im = index % width + pad_w;
                const int h_im = (index / width) % height + pad_h;
                const int c_im = index / (height * width);

                const int w_col_start = (w_im < kw) ? 0 : (w_im - kw) /stride_w + 1;
                const int w_col_end = min(w_im / stride_w + 1 , width_col);
                const int h_col_start = (h_im < kh) ? 0 : (h_im - kh) /stride_h + 1;
                const int h_col_end = min(h_im / stride_h + 1 , height_col);

                for(int h_col = h_col_start; h_col < h_col_end; h_col++){
                    for(int w_col = w_col_start; w_col < w_col_end; w_col++){
                        int w_k = w_im - w_col * stride_w;
                        int h_k = h_im - h_col * stride_h;

                        int col_index = (((c_im * kh + h_k) * kw + w_k) *
                                height_col + h_col) * width_col + w_col;
                        val += col[col_index];
                    }
                }
                im[index] = val;
            }

        }
    }

    template
    __global__ void col2im_gpu_2d<float,true>(float * im , float * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template
    __global__ void col2im_gpu_2d<float,false>(float * im , float * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template
    __global__ void col2im_gpu_2d<double , true>(double * im , double * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );
    template
    __global__ void col2im_gpu_2d<double , false>(double * im , double * col , const int n, // n for im
        const int channels , const int height , const int width,
        const int kh , const int kw , const int pad_h , const int pad_w,
        const int stride_h , const int stride_w,
        const int height_col , const int width_col
    );




    namespace Functional {
        __global__ void _sum_forward_kernel_f(float * output , const float * input ,  const int reduce , const int inner){
            int ridx = threadIdx.x;
            int iidx = blockIdx.x % inner;
            int oidx = blockIdx.x / inner;
            extern __shared__ float smem_sum_f[];
            int warpId = ridx / 32;
            int laneId = ridx % 32;

            constexpr int warpsPerBlock = kCudaThreadsNum / 32;
            float sum = 0.0f;
            int ri = reduce * oidx * inner + iidx;
            for(int i = ridx;i<reduce;i+=blockDim.x){
                sum += input[ri + i * inner];
            }
            sum = warpReduceSum<float>(sum);
            if(laneId == 0) smem_sum_f[warpId] = sum;
            __syncthreads();
            if(ridx < warpsPerBlock){
                sum = smem_sum_f[ridx];
                sum = warpReduceSum<float>(sum);
                if(ridx == 0){
                    output[oidx * inner + iidx] = sum;
                }
            }

        }
        __global__ void _sum_forward_kernel_d(double * output , const double * input ,  const int reduce , const int inner){
            int ridx = threadIdx.x;
            int iidx = blockIdx.x % inner;
            int oidx = blockIdx.x / inner;
            extern __shared__ double smem_sum_d[];
            int warpId = ridx / 32;
            int laneId = ridx % 32;

            constexpr int warpsPerBlock = kCudaThreadsNum / 32;
            double sum = 0.0f;
            int ri = reduce * oidx * inner + iidx;
            for(int i = ridx;i<reduce;i+=blockDim.x){
                sum += input[ri + i * inner];
            }
            sum = warpReduceSum<double>(sum);
            if(laneId == 0) smem_sum_d[warpId] = sum;
            __syncthreads();
            if(ridx < warpsPerBlock){
                sum = smem_sum_d[ridx];
                sum = warpReduceSum<double>(sum);
                if(ridx == 0){
                    output[oidx * inner + iidx] = sum;
                }
            }
        }


    }

}

    
} // namespace mytorch