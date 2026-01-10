#include "nn.cuh"
#include "math.cuh"
#include <cuda_runtime.h>

namespace mytorch{

namespace nn{

    #define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
    #define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])



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
            maxval = warpReduceMax<float>(maxval);
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


    __global__ void implSgemmgroup( float * output , const float * input ,const float * weight_ ,const float * bias,
        const int n , const int c , const int h , const int w, const int g ,  const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[24 * 1024];
        float *smemweight = reinterpret_cast<float *>(smem);
        float * smeminput = reinterpret_cast<float *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the output: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the output tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int input_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output

        int z = blockIdx.z; // batchsize
        int gid = blockIdx.z % g; // groupid

        // register for load from global memory
        // for pipeline
        float weight_ldg_reg[4];
        float input_ldg_reg[4];

        // original position in the image for the points to load 
        // in the input (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int posh_ori[4];
        int posw_ori[4];

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            posh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / ow) * stride_h - pad_h;
            posw_ori[i] = ((bx * 128 + tid % 32 + i * 32) % ow) * stride_w - pad_w;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int inputbatchoffset = z * c * h * w;
        int weightoffset = (by * 128 + tid / 8 * 4) * c * kh * kw;
        int inputchannelstep = h * w;
        int weightkstep = c * kh * kw;
        const float * weight = weight_ + gid * weightkstep * k;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int input_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        float weight_frag[2][8]; 
        float input_frag[2][8];
        float output_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize output frag
            FETCH_FLOAT4(output_frag[i][0]) = make_float4(0.0f,0.0f,0.0f,0.0f);
            FETCH_FLOAT4(output_frag[i][4]) = make_float4(0.0f,0.0f,0.0f,0.0f);
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if(tid % 8 < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                weight_ldg_reg[i] = weight[weightoffset + tid % 8 + i * weightkstep];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }

        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curC = (tid / 32) / (kh * kw);
        int curkH = ((tid / 32) % (kh * kw)) / kw;
        int curkW = ((tid / 32) % (kh * kw)) % kw;


    #pragma unroll
        for(int i = 0;i < 4;i++){
            int curH = posh_ori[i] + curkH;
            int curW = posw_ori[i] + curkW;
            int inoffsettmp = curC * inputchannelstep + curH * w + curW;
            if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
            }
            else{
                input_ldg_reg[i] = 0;
            }
        }

        // stores to shared (sts)
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
        FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr]);
        FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr + 16]);
        FETCH_FLOAT4(input_frag[0][0]) = FETCH_FLOAT4_CONST(smeminput[input_lds_addr]);
        FETCH_FLOAT4(input_frag[0][4]) = FETCH_FLOAT4_CONST(smeminput[input_lds_addr + 32]);
        for(int crs = 0; crs < c * kh * kw;crs+= 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
           int weightoffsettmp = crs + 8 + tid % 8;// +8 for prefetch
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if(weightoffsettmp < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                    weight_ldg_reg[i] = weight[weightoffset + weightoffsettmp + i * weightkstep];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }

            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */
            int curC = (crs + 8 + tid / 32) / (kh * kw);
            int curkH = ((crs + 8 + tid / 32) % (kh * kw)) / kw;
            int curkW = ((crs + 8 + tid / 32) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int curH = posh_ori[i] + curkH;
                int curW = posw_ori[i] + curkW;
                int inoffsettmp = curC * inputchannelstep + curH * w + curW;
                if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                    input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
                }
                else{
                    input_ldg_reg[i] = 0;
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
            {

                FETCH_FLOAT4(weight_frag[(subcrs + 1) % 2][0]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 ]);
                FETCH_FLOAT4(weight_frag[(subcrs + 1) % 2][4]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + 16]);


                FETCH_FLOAT4( input_frag[(subcrs + 1) % 2][0]) =FETCH_FLOAT4_CONST( smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 ]);
                FETCH_FLOAT4( input_frag[(subcrs + 1) % 2][4]) =FETCH_FLOAT4_CONST( smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + 32]);
    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                    }
                }
            }

            // store to shared mem
            FETCH_FLOAT4(  smemweight[write_flag * 132 * 8 + weight_sts_addr ] )= FETCH_FLOAT4_CONST(weight_ldg_reg[0]);
            for (int i = 0; i < 4; ++i)
            {
                smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;
            FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr]);
            FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + 16]);
            FETCH_FLOAT4( input_frag[0][0]) = FETCH_FLOAT4( smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr ]);
            FETCH_FLOAT4( input_frag[0][4]) = FETCH_FLOAT4( smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + 32]);
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
                }
            }
        }

        float * smemoutput = reinterpret_cast<float *>(smem);
        float * smembias = reinterpret_cast<float *>(smem + 16 * 1024);

        if(tid < 128){
            smembias[tid] = bias ? bias[by * 128 + tid] : 0;
        }

        // only store a quater of the output!

        int output_sts_addr = warp_id * 512 +  warp_tile_y * 4 * 8 * 4 + warp_tile_x * 4;
        // load:
        /*
        (0 ~ 32) * 16
        */
        int output_lds_addr = warp_id * 512 + lane_id;
        int bias_lds_addr = warp_id / 2 * 32;

        int m_idx = by * 128 + warp_id / 2 * 32;
        int n_idx = bx * 128 + warp_id % 2 * 64 + lane_id;

    #pragma unroll
        for(int i = 0;i < 2;i++){
    #pragma unroll
            for(int j = 0;j <2 ;j++){
                __syncthreads();
#pragma unroll
                for(int subi = 0; subi < 4;subi++){ 
#pragma unroll
                    for(int subj = 0;subj < 4;subj++){
                        smemoutput[output_sts_addr + subi * 32 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                    }
                }
                __syncthreads();
#pragma unroll

                for(int subk = 0;subk < 16;subk++){
                    int outOffset = z * k * oh * ow + (m_idx + i * 16 + subk) * oh * ow + n_idx + j * 32;
                    if((m_idx + i * 16 + subk) < k && (n_idx + j * 32) < oh * ow){
                        output[outOffset] = smemoutput[output_lds_addr + subk * 32] + smembias[bias_lds_addr + i * 16 + subk];
                    }
                }

            }
        }

    }

    __global__ void implDgemmgroup( double * output , const double * input ,const double * weight_ ,const double * bias,
        const int n , const int c , const int h , const int w, const int g ,  const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[32 * 1024];
        double *smemweight = reinterpret_cast<double *>(smem);
        double * smeminput = reinterpret_cast<double *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the output: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the output tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int input_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output

        int z = blockIdx.z; // batchsize
        int gid = blockIdx.z % g; // groupid

        // register for load from global memory
        // for pipeline
        double weight_ldg_reg[4];
        double input_ldg_reg[4];

        // original position in the image for the points to load 
        // in the input (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int posh_ori[4];
        int posw_ori[4];

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            posh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / ow) * stride_h - pad_h;
            posw_ori[i] = ((bx * 128 + tid % 32 + i * 32) % ow) * stride_w - pad_w;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int inputbatchoffset = z * c * h * w;
        int weightoffset = (by * 128 + tid / 8 * 4) * c * kh * kw;
        int inputchannelstep = h * w;
        int weightkstep = c * kh * kw;
        const double * weight = weight_ + gid * weightkstep * k;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int input_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        double weight_frag[2][8]; 
        double input_frag[2][8];
        double output_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize output frag
            for(int j = 0;j < 8;j++){
                output_frag[i][j] = 0;
            }
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if(tid % 8 < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                weight_ldg_reg[i] = weight[weightoffset + tid % 8 + i * weightkstep];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }

        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curC = (tid / 32) / (kh * kw);
        int curkH = ((tid / 32) % (kh * kw)) / kw;
        int curkW = ((tid / 32) % (kh * kw)) % kw;


    #pragma unroll
        for(int i = 0;i < 4;i++){
            int curH = posh_ori[i] + curkH;
            int curW = posw_ori[i] + curkW;
            int inoffsettmp = curC * inputchannelstep + curH * w + curW;
            if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
            }
            else{
                input_ldg_reg[i] = 0;
            }
        }

        // stores to shared (sts)
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
    #pragma unroll
        for(int i = 0;i < 4;i++){
            weight_frag[0][i] = smemweight[weight_lds_addr + i];
            weight_frag[0][i+4] = smemweight[weight_lds_addr + i + 16];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            input_frag[0][i] = smeminput[input_lds_addr + i];
            input_frag[0][i+4] = smeminput[input_lds_addr + i + 32];
        }
        for(int crs = 0; crs < c * kh * kw;crs+= 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
           int weightoffsettmp = crs + 8 + tid % 8;// +8 for prefetch
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if(weightoffsettmp < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                    weight_ldg_reg[i] = weight[weightoffset + weightoffsettmp + i * weightkstep];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }

            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */
            int curC = (crs + 8 + tid / 32) / (kh * kw);
            int curkH = ((crs + 8 + tid / 32) % (kh * kw)) / kw;
            int curkW = ((crs + 8 + tid / 32) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int curH = posh_ori[i] + curkH;
                int curW = posw_ori[i] + curkW;
                int inoffsettmp = curC * inputchannelstep + curH * w + curW;
                if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                    input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
                }
                else{
                    input_ldg_reg[i] = 0;
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
            {
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i];
                    weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i + 16];
                }
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i];
                    input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i + 32];
                }

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                    }
                }
            }

            // store to shared mem
            for (int i = 0; i < 4; ++i)
            {
                smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
            }
            for (int i = 0; i < 4; ++i)
            {
                smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
                weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
            }
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[0][i] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i];
                input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i + 32];
            }
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
                }
            }
        }

        double * smemoutput = reinterpret_cast<double *>(smem);
        double * smembias = reinterpret_cast<double *>(smem + 16 * 1024);

        if(tid < 128){
            smembias[tid] = bias ? bias[by * 128 + tid] : 0;
        }

        // only store a quater of the output!

        int output_sts_addr = warp_id * 512 +  warp_tile_y * 4 * 8 * 4 + warp_tile_x * 4;
        // load:
        /*
        (0 ~ 32) * 16
        */
        int output_lds_addr = warp_id * 512 + lane_id;
        int bias_lds_addr = warp_id / 2 * 32;

        int m_idx = by * 128 + warp_id / 2 * 32;
        int n_idx = bx * 128 + warp_id % 2 * 64 + lane_id;

    #pragma unroll
        for(int i = 0;i < 2;i++){
    #pragma unroll
            for(int j = 0;j <2 ;j++){
                __syncthreads();
#pragma unroll
                for(int subi = 0; subi < 4;subi++){ 
#pragma unroll
                    for(int subj = 0;subj < 4;subj++){
                        smemoutput[output_sts_addr + subi * 32 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                    }
                }
                __syncthreads();
#pragma unroll

                for(int subk = 0;subk < 16;subk++){
                    int outOffset = z * k * oh * ow + (m_idx + i * 16 + subk) * oh * ow + n_idx + j * 32;
                    if((m_idx + i * 16 + subk) < k && (n_idx + j * 32) < oh * ow){
                        output[outOffset] = smemoutput[output_lds_addr + subk * 32] + smembias[bias_lds_addr + i * 16 + subk];
                    }
                }

            }
        }

    }


    __global__ void implSgemm( float * output , const float * input ,const float * weight ,const float * bias,
        const int n , const int c , const int h , const int w, const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[24 * 1024];
        float *smemweight = reinterpret_cast<float *>(smem);
        float * smeminput = reinterpret_cast<float *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the output: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the output tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int input_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output

        int z = blockIdx.z; // batchsize

        // register for load from global memory
        // for pipeline
        float weight_ldg_reg[4];
        float input_ldg_reg[4];

        // original position in the image for the points to load 
        // in the input (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int posh_ori[4];
        int posw_ori[4];

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            posh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / ow) * stride_h - pad_h;
            posw_ori[i] = ((bx * 128 + tid % 32 + i * 32) % ow) * stride_w - pad_w;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int inputbatchoffset = z * c * h * w;
        int weightoffset = (by * 128 + tid / 8 * 4) * c * kh * kw;
        int inputchannelstep = h * w;
        int weightkstep = c * kh * kw;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int input_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        float weight_frag[2][8]; 
        float input_frag[2][8];
        float output_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize output frag
            FETCH_FLOAT4(output_frag[i][0]) = make_float4(0.0f,0.0f,0.0f,0.0f);
            FETCH_FLOAT4(output_frag[i][4]) = make_float4(0.0f,0.0f,0.0f,0.0f);
        }
        //replace using float4

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if(tid % 8 < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                weight_ldg_reg[i] = weight[weightoffset + tid % 8 + i * weightkstep];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }

        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curC = (tid / 32) / (kh * kw);
        int curkH = ((tid / 32) % (kh * kw)) / kw;
        int curkW = ((tid / 32) % (kh * kw)) % kw;


    #pragma unroll
        for(int i = 0;i < 4;i++){
            int curH = posh_ori[i] + curkH;
            int curW = posw_ori[i] + curkW;
            int inoffsettmp = curC * inputchannelstep + curH * w + curW;
            if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
            }
            else{
                input_ldg_reg[i] = 0;
            }
        }

        // stores to shared (sts)
        FETCH_FLOAT4(smemweight[weight_sts_addr]) = FETCH_FLOAT4_CONST(weight_ldg_reg[0]);

    #pragma unroll
        for(int i = 0;i < 4;i++){
            smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
        FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr]);
        FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr + 16]);
        FETCH_FLOAT4(input_frag[0][0]) = FETCH_FLOAT4_CONST(smeminput[input_lds_addr]);
        FETCH_FLOAT4(input_frag[0][4]) = FETCH_FLOAT4_CONST(smeminput[input_lds_addr + 32]);
        for(int crs = 0; crs < c * kh * kw;crs+= 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
           int weightoffsettmp = crs + 8 + tid % 8;// +8 for prefetch
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if(weightoffsettmp < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                    weight_ldg_reg[i] = weight[weightoffset + weightoffsettmp + i * weightkstep];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }

            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */
            int curC = (crs + 8 + tid / 32) / (kh * kw);
            int curkH = ((crs + 8 + tid / 32) % (kh * kw)) / kw;
            int curkW = ((crs + 8 + tid / 32) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int curH = posh_ori[i] + curkH;
                int curW = posw_ori[i] + curkW;
                int inoffsettmp = curC * inputchannelstep + curH * w + curW;
                if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                    input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
                }
                else{
                    input_ldg_reg[i] = 0;
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
            {
                FETCH_FLOAT4(weight_frag[(subcrs + 1) % 2][0]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 ]);
                FETCH_FLOAT4(weight_frag[(subcrs + 1) % 2][4]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + 16]);


                FETCH_FLOAT4( input_frag[(subcrs + 1) % 2][0]) =FETCH_FLOAT4_CONST( smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 ]);
                FETCH_FLOAT4( input_frag[(subcrs + 1) % 2][4]) =FETCH_FLOAT4_CONST( smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + 32]);

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                    }
                }
            }

            FETCH_FLOAT4(  smemweight[write_flag * 132 * 8 + weight_sts_addr ] )= FETCH_FLOAT4_CONST(weight_ldg_reg[0]);

    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
            }

            __syncthreads();

            write_flag ^= 1;
            FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr]);
            FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + 16]);
            FETCH_FLOAT4( input_frag[0][0]) = FETCH_FLOAT4( smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr ]);
            FETCH_FLOAT4( input_frag[0][4]) = FETCH_FLOAT4( smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + 32]);
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
                }
            }
        }

        float * smemoutput = reinterpret_cast<float *>(smem);
        float * smembias = reinterpret_cast<float *>(smem + 16 * 1024);

        if(tid < 128){
            smembias[tid] = bias ? bias[by * 128 + tid] : 0;
        }

        // only store a quater of the output!

        int output_sts_addr = warp_id * 512 +  warp_tile_y * 4 * 8 * 4 + warp_tile_x * 4;
        // load:
        /*
        (0 ~ 32) * 16
        */
        int output_lds_addr = warp_id * 512 + lane_id;
        int bias_lds_addr = warp_id / 2 * 32;

        int m_idx = by * 128 + warp_id / 2 * 32;
        int n_idx = bx * 128 + warp_id % 2 * 64 + lane_id;

    #pragma unroll
        for(int i = 0;i < 2;i++){
    #pragma unroll
            for(int j = 0;j <2 ;j++){
                __syncthreads();
#pragma unroll
                for(int subi = 0; subi < 4;subi++){ 
#pragma unroll
                    for(int subj = 0;subj < 4;subj++){
                        smemoutput[output_sts_addr + subi * 32 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                    }
                }
                __syncthreads();
#pragma unroll

                for(int subk = 0;subk < 16;subk++){
                    int outOffset = z * k * oh * ow + (m_idx + i * 16 + subk) * oh * ow + n_idx + j * 32;
                    if((m_idx + i * 16 + subk) < k && (n_idx + j * 32) < oh * ow){
                        output[outOffset] = smemoutput[output_lds_addr + subk * 32] + smembias[bias_lds_addr + i * 16 + subk];
                    }
                }

            }
        }

    }

    __global__ void implDgemm( double * output , const double * input ,const double * weight ,const double * bias,
        const int n , const int c , const int h , const int w, const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[32 * 1024];
        double *smemweight = reinterpret_cast<double *>(smem);
        double * smeminput = reinterpret_cast<double *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the output: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the output tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int input_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output

        int z = blockIdx.z; // batchsize

        // register for load from global memory
        // for pipeline
        double weight_ldg_reg[4];
        double input_ldg_reg[4];

        // original position in the image for the points to load 
        // in the input (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int posh_ori[4];
        int posw_ori[4];

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            posh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / ow) * stride_h - pad_h;
            posw_ori[i] = ((bx * 128 + tid % 32 + i * 32) % ow) * stride_w - pad_w;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int inputbatchoffset = z * c * h * w;
        int weightoffset = (by * 128 + tid / 8 * 4) * c * kh * kw;
        int inputchannelstep = h * w;
        int weightkstep = c * kh * kw;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int input_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        double weight_frag[2][8]; 
        double input_frag[2][8];
        double output_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize output frag
            for(int j = 0;j < 8;j++){
                output_frag[i][j] = 0;
            }
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if(tid % 8 < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                weight_ldg_reg[i] = weight[weightoffset + tid % 8 + i * weightkstep];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }

        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curC = (tid / 32) / (kh * kw);
        int curkH = (tid / 32) % (kh * kw) / kw;
        int curkW = (tid / 32) % (kh * kw) % kw;

    #pragma unroll
        for(int i = 0;i < 4;i++){
            int curH = posh_ori[i] + curkH;
            int curW = posw_ori[i] + curkW;
            int inoffsettmp = curC * inputchannelstep + curH * w + curW;
            if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
            }
            else{
                input_ldg_reg[i] = 0;
            }
        }

        // stores to shared (sts)
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smeminput[input_sts_addr + i * 32] = input_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
    #pragma unroll
        for(int i = 0;i < 4;i++){
            weight_frag[0][i] = smemweight[weight_lds_addr + i];
            weight_frag[0][i+4] = smemweight[weight_lds_addr + i + 16];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            input_frag[0][i] = smeminput[input_lds_addr + i];
            input_frag[0][i+4] = smeminput[input_lds_addr + i + 32];
        }
        for(int crs = 0; crs < c * kh * kw;crs+= 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
           int weightoffsettmp = crs + 8 + tid % 8;// +8 for prefetch
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if(weightoffsettmp < weightkstep && (by * 128 + tid / 8 * 4 + i) < k){
                    weight_ldg_reg[i] = weight[weightoffset + weightoffsettmp + i * weightkstep];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }

            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */
            int curC = (crs + 8 + tid / 32) / (kh * kw);
            int curkH = (crs + 8 + tid / 32) % (kh * kw) / kw;
            int curkW = (crs + 8 + tid / 32) % (kh * kw) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int curH = posh_ori[i] + curkH;
                int curW = posw_ori[i] + curkW;
                int inoffsettmp = curC * inputchannelstep + curH * w + curW;
                 if(curH >= 0 && curH < h && curW >= 0 && curW < w && curC < c){
                    input_ldg_reg[i] = input[inputbatchoffset + inoffsettmp];
                }
                else{
                    input_ldg_reg[i] = 0;
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subcrs = 0; subcrs < 8 - 1; ++subcrs)
            {
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    weight_frag[(subcrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i];
                    weight_frag[(subcrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subcrs + 1) * 132 + i + 16];
                }
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    input_frag[(subcrs + 1) % 2][i] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i];
                    input_frag[(subcrs + 1) % 2][i + 4] = smeminput[load_flag * 128 * 8 + input_lds_addr + (subcrs + 1) * 128 + i + 32];
                }

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        output_frag[i][j] += weight_frag[subcrs % 2][i] * input_frag[subcrs % 2][j];
                    }
                }
            }

            // store to shared mem
            for (int i = 0; i < 4; ++i)
            {
                smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
            }
            for (int i = 0; i < 4; ++i)
            {
                smeminput[write_flag * 128 * 8 + input_sts_addr + i * 32] = input_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
                weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
            }
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                input_frag[0][i] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i];
                input_frag[0][i + 4] = smeminput[(load_flag ^ 1) * 128 * 8 + input_lds_addr + i + 32];
            }
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    output_frag[i][j] += weight_frag[1][i] * input_frag[1][j];
                }
            }
        }

        double * smemoutput = reinterpret_cast<double *>(smem);
        double * smembias = reinterpret_cast<double *>(smem + 16 * 1024);

        if(tid < 128){
            smembias[tid] = bias ? bias[by * 128 + tid] : 0;
        }

        // only store a quater of the output!

        int output_sts_addr = warp_id * 512 +  warp_tile_y * 4 * 8 * 4 + warp_tile_x * 4;
        // load:
        /*
        (0 ~ 32) * 16
        */
        int output_lds_addr = warp_id * 512 + lane_id;
        int bias_lds_addr = warp_id / 2 * 32;

        int m_idx = by * 128 + warp_id / 2 * 32;
        int n_idx = bx * 128 + warp_id % 2 * 64 + lane_id;

    #pragma unroll
        for(int i = 0;i < 2;i++){
    #pragma unroll
            for(int j = 0;j <2 ;j++){
                __syncthreads();
#pragma unroll
                for(int subi = 0; subi < 4;subi++){ 
#pragma unroll
                    for(int subj = 0;subj < 4;subj++){
                        smemoutput[output_sts_addr + subi * 32 + subj] = output_frag[i * 4 + subi][j * 4 + subj];
                    }
                }
                __syncthreads();
#pragma unroll

                for(int subk = 0;subk < 16;subk++){
                    int outOffset = z * k * oh * ow + (m_idx + i * 16 + subk) * oh * ow + n_idx + j * 32;
                    if((m_idx + i * 16 + subk) < k && (n_idx + j * 32) < oh * ow){
                        output[outOffset] = smemoutput[output_lds_addr + subk * 32] + smembias[bias_lds_addr + i * 16 + subk];
                    }
                }

            }
        }

    }

    __global__ void implSgemmgradinputgroup( float * gradinput , const float * gradout ,const float * weight_ ,
        const int n , const int c , const int h , const int w, const int g ,  const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[24 * 1024];
        float *smemweight = reinterpret_cast<float *>(smem);
        float * smemgradout = reinterpret_cast<float *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the gradinput: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the gradinput tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int gradout_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output
        int x = bx * 128 + gradout_lds_addr;
        int y = by * 128 + weight_lds_addr;
        int z = blockIdx.z; // batchid
        int gid = blockIdx.z % g; // groupid

        // register for load from global memory
        // for pipeline
        float weight_ldg_reg[4];
        float gradout_ldg_reg[4];

        // original position in the image for the points to load 
        // in the gradout (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int possoh_ori[4];
        int possow_ori[4];
        // calculate the pad for output grad
        int soh = (oh - 1) * stride_h + 1;
        int pad_h_out = (kh + h - 1 - soh + 1) / 2;
        int sow = (ow - 1) * stride_w + 1;
        int pad_w_out = (kw + w - 1 - sow + 1) / 2;

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            possoh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / w) - pad_h_out;
            possow_ori[i] = ((bx * 128 + tid % 32 + i * 32) % w) - pad_w_out;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int gradoutbatchoffset = z * k * oh * ow;
        int weightC = (by * 128 + tid / 8 * 4);
        int outkstep = oh * ow;
        int weicstep = kh * kw;
        int weikstep = c * kh * kw;
        const float * weight = weight_ + gid * weikstep * k;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int gradout_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        float weight_frag[2][8]; 
        float gradout_frag[2][8];
        float gradinput_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize gradinput frag
            FETCH_FLOAT4(gradinput_frag[i][0]) = make_float4(0.0f,0.0f,0.0f,0.0f);
            FETCH_FLOAT4(gradinput_frag[i][4]) = make_float4(0.0f,0.0f,0.0f,0.0f);
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
       int curKRS = tid % 8;
       int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
       int curK = curKRS / (kh * kw);
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if( (curK * kh * kw + rs) < kh * kw * k && weightC + i < c){
                weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }


        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curK2 = (tid / 32) / (kh * kw);
        int curkH = (tid / 32) % (kh * kw) / kw;
        int curkW = (tid / 32) % (kh * kw) % kw;

    #pragma unroll
        for(int i = 0;i < 4;i++){
            int cursOh = possoh_ori[i] + curkH;
            int cursOw = possow_ori[i] + curkW;
            int curOh = cursOh  / stride_h;
            int curOw = cursOw  / stride_w;
            if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                gradout_ldg_reg[i] = 0;
            }
            else{
                int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                    gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                }
                else{
                    gradout_ldg_reg[i] = 0;
                }
            }
        }

        // stores to shared (sts)
        FETCH_FLOAT4(smemweight[weight_sts_addr]) = FETCH_FLOAT4_CONST(weight_ldg_reg[0]);
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemgradout[gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
        FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr]);
        FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr + 16]);
        FETCH_FLOAT4(gradout_frag[0][0]) = FETCH_FLOAT4_CONST(smemgradout[gradout_lds_addr]);
        FETCH_FLOAT4(gradout_frag[0][4]) = FETCH_FLOAT4_CONST(smemgradout[gradout_lds_addr + 32]);
        for(int krs = 0; krs < k * kh * kw ; krs += 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
            int curKRS = krs + tid % 8 + 8;
            int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
            int curK = curKRS / (kh * kw);
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if( (curK * kh * kw + rs) < kh * kw * k){
                    weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }



            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */

            int curK2 = ( krs + tid/ 32 + 8) / (kh * kw);
            int curkH = (( krs + tid / 32 + 8) % (kh * kw)) / kw;
            int curkW = (( krs + tid / 32 + 8) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int cursOh = possoh_ori[i] + curkH;
                int cursOw = possow_ori[i] + curkW;
                int curOh = cursOh  / stride_h;
                int curOw = cursOw  / stride_w;
                if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                    gradout_ldg_reg[i] = 0;
                }
                else{
                    int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                    if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                        gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                    }
                    else{
                        gradout_ldg_reg[i] = 0;
                    }
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subkrs = 0; subkrs < 8 - 1; ++subkrs)
            {
                FETCH_FLOAT4(weight_frag[(subkrs + 1) % 2][0]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 ]);
                FETCH_FLOAT4(weight_frag[(subkrs + 1) % 2][4]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 + 16]);


                FETCH_FLOAT4( gradout_frag[(subkrs + 1) % 2][0]) =FETCH_FLOAT4_CONST( smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 ]);
                FETCH_FLOAT4( gradout_frag[(subkrs + 1) % 2][4]) =FETCH_FLOAT4_CONST( smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 + 32]);

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        gradinput_frag[i][j] += weight_frag[subkrs % 2][i] * gradout_frag[subkrs % 2][j];
                    }
                }
            }

            // store to shared mem
            FETCH_FLOAT4(  smemweight[write_flag * 132 * 8 + weight_sts_addr ] )= FETCH_FLOAT4_CONST(weight_ldg_reg[0]);
            for (int i = 0; i < 4; ++i)
            {
                smemgradout[write_flag * 128 * 8 + gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;

            FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr]);
            FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + 16]);
            FETCH_FLOAT4( gradout_frag[0][0]) = FETCH_FLOAT4( smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr]);
            FETCH_FLOAT4( gradout_frag[0][4]) = FETCH_FLOAT4( smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr + 32]);
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradinput_frag[i][j] += weight_frag[1][i] * gradout_frag[1][j];
                }
            }
        }

        int gradinputOffset;
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j;
                if (x + j < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j];
                }
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j + 4];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j;
                if (x + j < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j + 4];
                }
            }
        }

    }

    __global__ void implDgemmgradinputgroup( double * gradinput , const double * gradout ,const double * weight_ ,
        const int n , const int c , const int h , const int w, const int g ,  const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[32 * 1024];
        double *smemweight = reinterpret_cast<double *>(smem);
        double * smemgradout = reinterpret_cast<double *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the gradinput: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the gradinput tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int gradout_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output
        int x = bx * 128 + gradout_lds_addr;
        int y = by * 128 + weight_lds_addr;
        int z = blockIdx.z; // batchid
        int gid = blockIdx.z % g; // groupid

        // register for load from global memory
        // for pipeline
        double weight_ldg_reg[4];
        double gradout_ldg_reg[4];

        // original position in the image for the points to load 
        // in the gradout (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int possoh_ori[4];
        int possow_ori[4];
        // calculate the pad for output grad
        int soh = (oh - 1) * stride_h + 1;
        int pad_h_out = (kh + h - 1 - soh + 1) / 2;
        int sow = (ow - 1) * stride_w + 1;
        int pad_w_out = (kw + w - 1 - sow + 1) / 2;

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            possoh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / w) - pad_h_out;
            possow_ori[i] = ((bx * 128 + tid % 32 + i * 32) % w) - pad_w_out;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int gradoutbatchoffset = z * k * oh * ow;
        int weightC = (by * 128 + tid / 8 * 4);
        int outkstep = oh * ow;
        int weicstep = kh * kw;
        int weikstep = c * kh * kw;
        const double * weight = weight_ + gid * weikstep * k;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int gradout_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        double weight_frag[2][8]; 
        double gradout_frag[2][8];
        double gradinput_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize gradinput frag
            for(int j = 0;j < 8;j++){
                gradinput_frag[i][j] = 0;
            }
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
       int curKRS = tid % 8;
       int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
       int curK = curKRS / (kh * kw);
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if( (curK * kh * kw + rs) < kh * kw * k && weightC + i < c){
                weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }


        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curK2 = (tid / 32) / (kh * kw);
        int curkH = (tid / 32) % (kh * kw) / kw;
        int curkW = (tid / 32) % (kh * kw) % kw;

    #pragma unroll
        for(int i = 0;i < 4;i++){
            int cursOh = possoh_ori[i] + curkH;
            int cursOw = possow_ori[i] + curkW;
            int curOh = cursOh  / stride_h;
            int curOw = cursOw  / stride_w;
            if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                gradout_ldg_reg[i] = 0;
            }
            else{
                int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                    gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                }
                else{
                    gradout_ldg_reg[i] = 0;
                }
            }
        }

        // stores to shared (sts)
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemgradout[gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
    #pragma unroll
        for(int i = 0;i < 4;i++){
            weight_frag[0][i] = smemweight[weight_lds_addr + i];
            weight_frag[0][i+4] = smemweight[weight_lds_addr + i + 16];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            gradout_frag[0][i] = smemgradout[gradout_lds_addr + i];
            gradout_frag[0][i+4] = smemgradout[gradout_lds_addr + i + 32];
        }
        for(int krs = 0; krs < k * kh * kw ; krs += 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
            int curKRS = krs + tid % 8 + 8;
            int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
            int curK = curKRS / (kh * kw);
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if( (curK * kh * kw + rs) < kh * kw * k){
                    weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }



            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */

            int curK2 = ( krs + tid/ 32 + 8) / (kh * kw);
            int curkH = (( krs + tid / 32 + 8) % (kh * kw)) / kw;
            int curkW = (( krs + tid / 32 + 8) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int cursOh = possoh_ori[i] + curkH;
                int cursOw = possow_ori[i] + curkW;
                int curOh = cursOh  / stride_h;
                int curOw = cursOw  / stride_w;
                if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                    gradout_ldg_reg[i] = 0;
                }
                else{
                    int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                    if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                        gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                    }
                    else{
                        gradout_ldg_reg[i] = 0;
                    }
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subkrs = 0; subkrs < 8 - 1; ++subkrs)
            {
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    weight_frag[(subkrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 + i];
                    weight_frag[(subkrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 + i + 16];
                }
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    gradout_frag[(subkrs + 1) % 2][i] = smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 + i];
                    gradout_frag[(subkrs + 1) % 2][i + 4] = smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 + i + 32];
                }

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        gradinput_frag[i][j] += weight_frag[subkrs % 2][i] * gradout_frag[subkrs % 2][j];
                    }
                }
            }

            // store to shared mem
            for (int i = 0; i < 4; ++i)
            {
                smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
            }
            for (int i = 0; i < 4; ++i)
            {
                smemgradout[write_flag * 128 * 8 + gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
                weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
            }
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                gradout_frag[0][i] = smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr + i];
                gradout_frag[0][i + 4] = smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr + i + 32];
            }
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradinput_frag[i][j] += weight_frag[1][i] * gradout_frag[1][j];
                }
            }
        }

        int gradinputOffset;
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j;
                if (x + j < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j];
                }
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j + 4];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j;
                if (x + j < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j + 4];
                }
            }
        }

    }

    __global__ void implSgemmgradinput( float * gradinput , const float * gradout ,const float * weight ,
        const int n , const int c , const int h , const int w, const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[24 * 1024];
        float *smemweight = reinterpret_cast<float *>(smem);
        float * smemgradout = reinterpret_cast<float *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the gradinput: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the gradinput tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int gradout_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output
        int x = bx * 128 + gradout_lds_addr;
        int y = by * 128 + weight_lds_addr;
        int z = blockIdx.z; // batchsize

        // register for load from global memory
        // for pipeline
        float weight_ldg_reg[4];
        float gradout_ldg_reg[4];

        // original position in the image for the points to load 
        // in the gradout (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int possoh_ori[4];
        int possow_ori[4];
        // calculate the pad for output grad
        int soh = (oh - 1) * stride_h + 1;
        int pad_h_out = (kh + h - 1 - soh + 1) / 2;
        int sow = (ow - 1) * stride_w + 1;
        int pad_w_out = (kw + w - 1 - sow + 1) / 2;

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            possoh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / w) - pad_h_out;
            possow_ori[i] = ((bx * 128 + tid % 32 + i * 32) % w) - pad_w_out;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int gradoutbatchoffset = z * k * oh * ow;
        int weightC = (by * 128 + tid / 8 * 4);
        int outkstep = oh * ow;
        int weicstep = kh * kw;
        int weikstep = c * kh * kw;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int gradout_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        float weight_frag[2][8]; 
        float gradout_frag[2][8];
        float gradinput_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize gradinput frag
            FETCH_FLOAT4(gradinput_frag[i][0]) = make_float4(0.0f,0.0f,0.0f,0.0f);
            FETCH_FLOAT4(gradinput_frag[i][4]) = make_float4(0.0f,0.0f,0.0f,0.0f);
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
       int curKRS = tid % 8;
       int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
       int curK = curKRS / (kh * kw);
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if( (curK * kh * kw + rs) < kh * kw * k && weightC + i < c){
                weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }


        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curK2 = (tid / 32) / (kh * kw);
        int curkH = (tid / 32) % (kh * kw) / kw;
        int curkW = (tid / 32) % (kh * kw) % kw;

    #pragma unroll
        for(int i = 0;i < 4;i++){
            int cursOh = possoh_ori[i] + curkH;
            int cursOw = possow_ori[i] + curkW;
            int curOh = cursOh  / stride_h;
            int curOw = cursOw  / stride_w;
            if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                gradout_ldg_reg[i] = 0;
            }
            else{
                int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                    gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                }
                else{
                    gradout_ldg_reg[i] = 0;
                }
            }
        }

        // stores to shared (sts)
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemgradout[gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
        FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr]);
        FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[weight_lds_addr + 16]);
        FETCH_FLOAT4(gradout_frag[0][0]) = FETCH_FLOAT4_CONST(smemgradout[gradout_lds_addr]);
        FETCH_FLOAT4(gradout_frag[0][4]) = FETCH_FLOAT4_CONST(smemgradout[gradout_lds_addr + 32]);
        for(int krs = 0; krs < k * kh * kw ; krs += 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
            int curKRS = krs + tid % 8 + 8;
            int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
            int curK = curKRS / (kh * kw);
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if( (curK * kh * kw + rs) < kh * kw * k){
                    weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }



            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */

            int curK2 = ( krs + tid/ 32 + 8) / (kh * kw);
            int curkH = (( krs + tid / 32 + 8) % (kh * kw)) / kw;
            int curkW = (( krs + tid / 32 + 8) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int cursOh = possoh_ori[i] + curkH;
                int cursOw = possow_ori[i] + curkW;
                int curOh = cursOh  / stride_h;
                int curOw = cursOw  / stride_w;
                if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                    gradout_ldg_reg[i] = 0;
                }
                else{
                    int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                    if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                        gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                    }
                    else{
                        gradout_ldg_reg[i] = 0;
                    }
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subkrs = 0; subkrs < 8 - 1; ++subkrs)
            {
                FETCH_FLOAT4(weight_frag[(subkrs + 1) % 2][0]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 ]);
                FETCH_FLOAT4(weight_frag[(subkrs + 1) % 2][4]) = FETCH_FLOAT4_CONST(smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 + 16]);


                FETCH_FLOAT4( gradout_frag[(subkrs + 1) % 2][0]) =FETCH_FLOAT4_CONST( smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 ]);
                FETCH_FLOAT4( gradout_frag[(subkrs + 1) % 2][4]) =FETCH_FLOAT4_CONST( smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 + 32]);

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        gradinput_frag[i][j] += weight_frag[subkrs % 2][i] * gradout_frag[subkrs % 2][j];
                    }
                }
            }

            // store to shared mem
            FETCH_FLOAT4(  smemweight[write_flag * 132 * 8 + weight_sts_addr ] )= FETCH_FLOAT4_CONST(weight_ldg_reg[0]);
            for (int i = 0; i < 4; ++i)
            {
                smemgradout[write_flag * 128 * 8 + gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;
            FETCH_FLOAT4(weight_frag[0][0]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr]);
            FETCH_FLOAT4(weight_frag[0][4]) = FETCH_FLOAT4_CONST(smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + 16]);
            FETCH_FLOAT4( gradout_frag[0][0]) = FETCH_FLOAT4( smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr]);
            FETCH_FLOAT4( gradout_frag[0][4]) = FETCH_FLOAT4( smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr + 32]);
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradinput_frag[i][j] += weight_frag[1][i] * gradout_frag[1][j];
                }
            }
        }

        int gradinputOffset;
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j;
                if (x + j < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j];
                }
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j + 4];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j;
                if (x + j < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j + 4];
                }
            }
        }

    }

    __global__ void implDgemmgradinput( double * gradinput , const double * gradout ,const double * weight ,
        const int n , const int c , const int h , const int w, const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    ){
        // for alignment
        __shared__ __align__(16 * 1024) char smem[32 * 1024];
        double *smemweight = reinterpret_cast<double *>(smem);
        double * smemgradout = reinterpret_cast<double *>(smem + 16 * 1024);

        int tid = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        // arrange like this: (to avoid bank conflict)
        /*
        in a warp tile
        0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31
        map to the gradinput: (each number correspond to 4x4 tile)
        (so each laneid correspond to 8x8)
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        0   2   4   6   8   10  12  14 0   2   4   6   8   10  12  14
        1   3   5   7   9   11  13  15 1   3   5   7   9   11  13  15
        16  18  20  22  24  26  28  30 16  18  20  22  24  26  28  30
        17  19  21  23  25  27  29  31 17  19  21  23  25  27  29  31
        */
        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;
        const int warp_tile_x = (lane_id / 2) % 8;
        const int warp_tile_y = (lane_id / 16) * 2 + (lane_id % 2);

        //lds address :in the gradinput tile
        // correspond to the pos of the first laneid show in preceding tile
        int weight_lds_addr = (warp_id / 2) * 32 + warp_tile_y * 4;
        int gradout_lds_addr = (warp_id % 2) * 64 + warp_tile_x * 4;

        //address in the whole output
        int x = bx * 128 + gradout_lds_addr;
        int y = by * 128 + weight_lds_addr;
        int z = blockIdx.z; // batchsize

        // register for load from global memory
        // for pipeline
        double weight_ldg_reg[4];
        double gradout_ldg_reg[4];

        // original position in the image for the points to load 
        // in the gradout (individual with output)
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int possoh_ori[4];
        int possow_ori[4];
        // calculate the pad for output grad
        int soh = (oh - 1) * stride_h + 1;
        int pad_h_out = (kh + h - 1 - soh + 1) / 2;
        int sow = (ow - 1) * stride_w + 1;
        int pad_w_out = (kw + w - 1 - sow + 1) / 2;

    #pragma unroll
        for(int i = 0; i < 4;i++){ // intialize ori
            possoh_ori[i] = ((bx * 128 + tid % 32 + i * 32) / w) - pad_h_out;
            possow_ori[i] = ((bx * 128 + tid % 32 + i * 32) % w) - pad_w_out;
        }

        // for load!
        // kernel load like this:(transpose to store)
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
        int gradoutbatchoffset = z * k * oh * ow;
        int weightC = (by * 128 + tid / 8 * 4);
        int outkstep = oh * ow;
        int weicstep = kh * kw;
        int weikstep = c * kh * kw;

        // sts addr : where to store in smem (the first position)
        int weight_sts_addr = (tid % 8) * 132 + (tid / 8) * 4;
        int gradout_sts_addr = (tid / 32) * 128 + (tid % 32);

        // pipeline!
        int write_flag = 1;
        // frag for matmul
        double weight_frag[2][8]; 
        double gradout_frag[2][8];
        double gradinput_frag[8][8];

    #pragma unroll
        for(int i = 0;i < 8;i++){ // initialize gradinput frag
            for(int j = 0;j < 8;j++){
                gradinput_frag[i][j] = 0;
            }
        }

        // perform the first load from global(ldg) for pipeline

        // first kernel
        /*
        0 0 0 0 8 8 8 8 ...
        1 1 1 1 9 9 9 9
        2 2 2 2 10 10 10 10
        3 3 3 3 11 11 11 11
        ....
        */
       int curKRS = tid % 8;
       int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
       int curK = curKRS / (kh * kw);
    #pragma unroll
        for(int i = 0;i < 4;i++){
            if( (curK * kh * kw + rs) < kh * kw * k){
                weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
            }
            else{
                weight_ldg_reg[i] = 0;
            }
        }


        // next input
        /*
        [0~32] , [0~32] , [0~32] , [0~32]
        [32~64] , [32~64] , [32~64] , [32~64]
        ...
        */
        int curK2 = (tid / 32) / (kh * kw);
        int curkH = (tid / 32) % (kh * kw) / kw;
        int curkW = (tid / 32) % (kh * kw) % kw;

    #pragma unroll
        for(int i = 0;i < 4;i++){
            int cursOh = possoh_ori[i] + curkH;
            int cursOw = possow_ori[i] + curkW;
            int curOh = cursOh  / stride_h;
            int curOw = cursOw  / stride_w;
            if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                gradout_ldg_reg[i] = 0;
            }
            else{
                int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                    gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                }
                else{
                    gradout_ldg_reg[i] = 0;
                }
            }
        }

        // stores to shared (sts)
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemweight[weight_sts_addr + i] = weight_ldg_reg[i];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            smemgradout[gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
        }
        __syncthreads();
        // load from shared(lds)
        // load the number we need in matmul
    #pragma unroll
        for(int i = 0;i < 4;i++){
            weight_frag[0][i] = smemweight[weight_lds_addr + i];
            weight_frag[0][i+4] = smemweight[weight_lds_addr + i + 16];
        }
    #pragma unroll
        for(int i = 0;i < 4;i++){
            gradinput_frag[0][i] = smemgradout[gradout_lds_addr + i];
            gradinput_frag[0][i+4] = smemgradout[gradout_lds_addr + i + 32];
        }
        for(int krs = 0; krs < k * kh * kw ; krs += 8){
            // prefetch for pipeline
            // first kernel
            /*
            0 0 0 0 8 8 8 8 ...
            1 1 1 1 9 9 9 9
            2 2 2 2 10 10 10 10
            3 3 3 3 11 11 11 11
            ....
            */
            int curKRS = krs + tid % 8;
            int rs = kh  * kw -  1 - curKRS % ( kh * kw); // transpose
            int curK = curKRS / (kh * kw);
        #pragma unroll
            for(int i = 0;i < 4;i++){
                if( (curK * kh * kw + rs) < kh * kw * k){
                    weight_ldg_reg[i] = weight[curK * weikstep + (weightC + i) * weicstep + rs];
                }
                else{
                    weight_ldg_reg[i] = 0;
                }
            }



            // next input
            /*
            [0~32] , [0~32] , [0~32] , [0~32]
            [32~64] , [32~64] , [32~64] , [32~64]
            ...
            */

            int curK2 = ( krs + tid/ 32) / (kh * kw);
            int curkH = (( krs + tid / 32) % (kh * kw)) / kw;
            int curkW = (( krs + tid / 32) % (kh * kw)) % kw;

        #pragma unroll
            for(int i = 0;i < 4;i++){
                int cursOh = possoh_ori[i] + curkH;
                int cursOw = possow_ori[i] + curkW;
                int curOh = cursOh  / stride_h;
                int curOw = cursOw  / stride_w;
                if(curOh * stride_h!= cursOh || curOw * stride_w!= cursOw){
                    gradout_ldg_reg[i] = 0;
                }
                else{
                    int outoffsettmp = curK2 * outkstep + curOh * ow + curOw;
                    if(curOh >= 0 && curOh < oh && curOw >= 0 && curOw < ow && curK2 < k){
                        gradout_ldg_reg[i] = gradout[gradoutbatchoffset + outoffsettmp];
                    }
                    else{
                        gradout_ldg_reg[i] = 0;
                    }
                }
            }

            int load_flag = write_flag ^ 1;
    #pragma unroll
            for (int subkrs = 0; subkrs < 8 - 1; ++subkrs)
            {
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    weight_frag[(subkrs + 1) % 2][i] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 + i];
                    weight_frag[(subkrs + 1) % 2][i + 4] = smemweight[load_flag * 132 * 8 + weight_lds_addr + (subkrs + 1) * 132 + i + 16];
                }
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    gradinput_frag[(subkrs + 1) % 2][i] = smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 + i];
                    gradinput_frag[(subkrs + 1) % 2][i + 4] = smemgradout[load_flag * 128 * 8 + gradout_lds_addr + (subkrs + 1) * 128 + i + 32];
                }

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        gradinput_frag[i][j] += weight_frag[subkrs % 2][i] * gradout_frag[subkrs % 2][j];
                    }
                }
            }

            // store to shared mem
            for (int i = 0; i < 4; ++i)
            {
                smemweight[write_flag * 132 * 8 + weight_sts_addr + i] = weight_ldg_reg[i];
            }
            for (int i = 0; i < 4; ++i)
            {
                smemgradout[write_flag * 128 * 8 + gradout_sts_addr + i * 32] = gradout_ldg_reg[i];
            }
            __syncthreads();

            write_flag ^= 1;
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                weight_frag[0][i] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i];
                weight_frag[0][i + 4] = smemweight[(load_flag ^ 1) * 132 * 8 + weight_lds_addr + i + 16];
            }
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                gradinput_frag[0][i] = smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr + i];
                gradinput_frag[0][i + 4] = smemgradout[(load_flag ^ 1) * 128 * 8 + gradout_lds_addr + i + 32];
            }
    #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
    #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    gradinput_frag[i][j] += weight_frag[1][i] * gradinput_frag[1][j];
                }
            }
        }

        // 计算输出偏移
        int gradinputOffset;
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j;
                if (x + j < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j];
                }
                gradinputOffset = z * c * h * w + (y + i) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i][j + 4];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j;
                if (x + j < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j];
                }
                gradinputOffset = z * c * h * w + (y + i + 16) * h * w + x + j + 32;
                if (x + j + 32 < h * w && y + i + 16 < c)
                {
                    gradinput[gradinputOffset] = gradinput_frag[i + 4][j + 4];
                }
            }
        }

    }




    // compute grad of weight for all batch
    __global__ void implSgemmgradweight(float * gradweight , const float * input , const float * gradout,
        const int n , const int c , const int h , const int w, const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    )
    {
        uint32_t tx = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        const uint32_t lane_id = threadIdx.x % 32;
        const uint32_t warp_id = threadIdx.x / 32;
        const uint32_t mma_tid_x = (lane_id / 2) % 8;
        const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
        // lds addr
        uint32_t gradoutput_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
        uint32_t input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

        int x = bx * 128 + input_lds_addr;
        int y = by * 128 + gradoutput_lds_addr;
        int z = blockIdx.z;

        __shared__ float smeminput[8 * 128];
        __shared__ float smemgradoutput[8 * 132];

        int posh_ori[4];
        int posw_ori[4];
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / kw) - pad_h;
            posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % kw) - pad_w;
        }

        int inOffset = z * h * w;
        int outK = (by * 128 + tx / 8 * 4);
        int inNOffset = c * h * w;
        int outKOffset = oh * ow;
        int outNOffset = k * oh * ow;


        // sts addr
        uint32_t gradoutput_sts_addr = (tx % 8) * 132 +
                                    (tx / 8) * 4;
        uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);

        float gradoutput_frag[8];
        float input_frag[8];
        float gradweight_frag[8][8];
    #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                gradweight_frag[i][j] = 0;
            }
        }

        for (int nohow = 0; nohow < oh * ow * n; nohow += 8)
        {
            int curNOHOW = nohow + tx % 8;
            int ohow = curNOHOW % (oh * ow);
            int curN_1 = curNOHOW / (oh * ow);
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (curNOHOW < oh * ow * n && outK + i < k)
                {
                    smemgradoutput[gradoutput_sts_addr + i] = gradout[curN_1 * outNOffset + (outK + i) * outKOffset + ohow];
                }
                else
                {
                    smemgradoutput[gradoutput_sts_addr + i] = 0.0;
                }
            }

            int curN_2 = (nohow + tx / 32) / (oh * ow);             // output n offset
            int curOh = ((nohow + tx / 32) % (oh * ow)) / ow; // output h offset
            int cursOh = curOh * stride_h;
            int curOw = ((nohow + tx / 32) % (oh * ow)) % ow; // output w offset
            int cursOw = curOw * stride_w;

    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int curH = posh_ori[i] + cursOh; // input h
                int curW = posw_ori[i] + cursOw; // input w
                int inOffsetTmp = curN_2 * inNOffset + curH * w + curW;
                if (curH >= 0 && curW >= 0 && curW < w && curH < h && curN_2 < n)
                {
                    smeminput[input_sts_addr + i * 32] = input[inOffset + inOffsetTmp];
                }
                else
                {
                    smeminput[input_sts_addr + i * 32] = 0.0;
                }
            }
            __syncthreads();
    #pragma unroll
            for (int subnohow = 0; subnohow < 8; ++subnohow)
            {
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    gradoutput_frag[i] = smemgradoutput[gradoutput_lds_addr + subnohow * 132 + i];
                    gradoutput_frag[i + 4] = smemgradoutput[gradoutput_lds_addr + subnohow * 132 + i + 16];
                }
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    input_frag[i] = smeminput[input_lds_addr + subnohow * 128 + i];
                    input_frag[i + 4] = smeminput[input_lds_addr + subnohow * 128 + i + 32];
                }

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        gradweight_frag[i][j] += gradoutput_frag[i] * input_frag[j];
                    }
                }
            }
            __syncthreads();
        }

        int gradweightoffset;
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gradweightoffset = z * kh * kw + (y + i) * c * kh * kw + x + j;
                if (x + j < kh * kw && y + i < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i][j];
                }
                gradweightoffset = z * kh * kw + (y + i) * c * kh * kw + x + j + 32;
                if (x + j + 32 < kh * kw && y + i < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i][j + 4];
                }
                gradweightoffset = z * kh * kw + (y + i + 16) * c * kh * kw + x + j;
                if (x + j < kh * kw && y + i + 16 < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i + 4][j];
                }
                gradweightoffset = z * kh * kw + (y + i + 16) * c * kh * kw + x + j + 32;
                if (x + j + 32 < kh * kw && y + i + 16 < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i + 4][j + 4];
                }
            }
        }
    }

    __global__ void implDgemmgradweight(double * gradweight , const double * input , const double * gradout,
        const int n , const int c , const int h , const int w, const int k , const int kh , const int kw , 
        const int pad_h , const int pad_w , const int stride_h , const int stride_w,
        const int oh , const int ow
    )
    {
        uint32_t tx = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Warp tile
        const uint32_t lane_id = threadIdx.x % 32;
        const uint32_t warp_id = threadIdx.x / 32;
        const uint32_t mma_tid_x = (lane_id / 2) % 8;
        const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);
        // lds addr
        uint32_t gradoutput_lds_addr = (warp_id / 2) * 32 + mma_tid_y * 4;
        uint32_t input_lds_addr = (warp_id % 2) * 64 + mma_tid_x * 4;

        int x = bx * 128 + input_lds_addr;
        int y = by * 128 + gradoutput_lds_addr;
        int z = blockIdx.z;

        __shared__ double smeminput[8 * 128];
        __shared__ double smemgradoutput[8 * 132];

        int posh_ori[4];
        int posw_ori[4];
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            posh_ori[i] = ((bx * 128 + tx % 32 + i * 32) / kw) - pad_h;
            posw_ori[i] = ((bx * 128 + tx % 32 + i * 32) % kw) - pad_w;
        }

        int inOffset = z * h * w;
        int outK = (by * 128 + tx / 8 * 4);
        int inNOffset = c * h * w;
        int outKOffset = oh * ow;
        int outNOffset = k * oh * ow;


        // sts addr
        uint32_t gradoutput_sts_addr = (tx % 8) * 132 +
                                    (tx / 8) * 4;
        uint32_t input_sts_addr = (tx / 32) * 128 + (tx % 32);

        double gradoutput_frag[8];
        double input_frag[8];
        double gradweight_frag[8][8];
    #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                gradweight_frag[i][j] = 0;
            }
        }

        for (int nohow = 0; nohow < oh * ow * n; nohow += 8)
        {
            int curNOHOW = nohow + tx % 8;
            int ohow = curNOHOW % (oh * ow);
            int curN_1 = curNOHOW / (oh * ow);
    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                if (curNOHOW < oh * ow * n && outK + i < k)
                {
                    smemgradoutput[gradoutput_sts_addr + i] = gradout[curN_1 * outNOffset + (outK + i) * outKOffset + ohow];
                }
                else
                {
                    smemgradoutput[gradoutput_sts_addr + i] = 0.0;
                }
            }

            int curN_2 = (nohow + tx / 32) / (oh * ow);             // output n offset
            int curOh = ((nohow + tx / 32) % (oh * ow)) / ow; // output h offset
            int cursOh = curOh * stride_h;
            int curOw = ((nohow + tx / 32) % (oh * ow)) % ow; // output w offset
            int cursOw = curOw * stride_w;

    #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                int curH = posh_ori[i] + cursOh; // input h
                int curW = posw_ori[i] + cursOw; // input w
                int inOffsetTmp = curN_2 * inNOffset + curH * w + curW;
                if (curH >= 0 && curW >= 0 && curW < w && curH < h)
                {
                    smeminput[input_sts_addr + i * 32] = input[inOffset + inOffsetTmp];
                }
                else
                {
                    smeminput[input_sts_addr + i * 32] = 0.0;
                }
            }
            __syncthreads();
    #pragma unroll
            for (int subnohow = 0; subnohow < 8; ++subnohow)
            {
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    gradoutput_frag[i] = smemgradoutput[gradoutput_lds_addr + subnohow * 132 + i];
                    gradoutput_frag[i + 4] = smemgradoutput[gradoutput_lds_addr + subnohow * 132 + i + 16];
                }
    #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    input_frag[i] = smeminput[input_lds_addr + subnohow * 128 + i];
                    input_frag[i + 4] = smeminput[input_lds_addr + subnohow * 128 + i + 32];
                }

    #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        gradweight_frag[i][j] += gradoutput_frag[i] * input_frag[j];
                    }
                }
            }
            __syncthreads();
        }

        int gradweightoffset;
    #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
    #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gradweightoffset = z * kh * kw + (y + i) * c * kh * kw + x + j;
                if (x + j < kh * kw && y + i < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i][j];
                }
                gradweightoffset = z * kh * kw + (y + i) * c * kh * kw + x + j + 32;
                if (x + j + 32 < kh * kw && y + i < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i][j + 4];
                }
                gradweightoffset = z * kh * kw + (y + i + 16) * c * kh * kw + x + j;
                if (x + j < kh * kw && y + i + 16 < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i + 4][j];
                }
                gradweightoffset = z * kh * kw + (y + i + 16) * c * kh * kw + x + j + 32;
                if (x + j + 32 < kh * kw && y + i + 16 < k)
                {
                    gradweight[gradweightoffset] = gradweight_frag[i + 4][j + 4];
                }
            }
        }
    }





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