#ifndef _MATH_H_
#define _MATH_H_
#include "tensor.cuh"
#include <cublas_v2.h>

namespace mytorch{
    namespace nn{
        // defines general math functions for use
        template <typename T>
        __device__ __host__ inline T nn_rsqrt(T x){
            return rsqrt(x);
        }
        template <> 
        __device__ __host__ inline float nn_rsqrt(float x){
            return rsqrtf(x);
        }

        template <typename T>
        __device__ inline T nn_device_rsqrt(T x){
            return rsqrt(x);
        }
        template <> 
        __device__ inline float nn_device_rsqrt(float x){
            return __frsqrt_rn(x);
        }


        template <typename T>
        __device__ __host__ inline T nn_sqrt(T x){
            return sqrt(x);
        }
        template <> 
        __device__ __host__ inline float nn_sqrt(float x){
            return sqrtf(x);
        }

        template <typename T>
        __device__ inline T nn_device_sqrt(T x){
            return __fsqrt_rn(x);
        }

        template <typename T>
        __device__ __host__ inline T nn_exp(T x){
            return exp(x);
        }
        template <> 
        __device__ __host__ inline float nn_exp(float x){
            return expf(x);
        }
        template <typename T>
        __device__ inline T nn_exp_device(T x){
            return exp(x);
        }
        template <>
        __device__ inline float nn_exp_device(float x){
            return __expf(x);
        }

        template <typename T , bool transA = false , bool transB = false> // a trivial gemm implementation for cpu
        void __cpu_gemm( 
            const int M , const int N , const int K ,
            const T * A , const T * B , T * C , const T alpha = 1 , const T beta = 0){
            // C = alpha * A * B + beta * C
            if constexpr (!transA && !transB){
                for (int m = 0 ; m < M ; m++){
                    for (int n = 0 ; n < N ; n++){
                        T sum = 0;
                        for (int k = 0 ; k < K ; k++){
                            sum += A[m * K + k] * B[k * N + n];
                        }
                        C[m * N + n] = alpha * sum + beta * C[m * N + n];
                    }
                }
            }
            else if constexpr (transA && !transB){
                for (int m = 0 ; m < M ; m++){
                    for (int n = 0 ; n < N ; n++){
                        T sum = 0;
                        for (int k = 0 ; k < K ; k++){
                            sum += A[k * M + m] * B[k * N + n];
                        }
                        C[m * N + n] = alpha * sum + beta * C[m * N + n];
                    }
                }
            }
            else if constexpr (!transA && transB){
                for (int m = 0 ; m < M ; m++){
                    for (int n = 0 ; n < N ; n++){
                        T sum = 0;
                        for (int k = 0 ; k < K ; k++){
                            sum += A[m * K + k] * B[n * K + k];
                        }
                        C[m * N + n] = alpha * sum + beta * C[m * N + n];
                    }
                }
            }
            else{
                for (int m = 0 ; m < M ; m++){
                    for (int n = 0 ; n < N ; n++){
                        T sum = 0;
                        for (int k = 0 ; k < K ; k++){
                            sum += A[k * M + m] * B[n * K + k];
                        }
                        C[m * N + n] = alpha * sum + beta * C[m * N + n];
                    }
                }
            }
        }

        template <typename T , bool transA = false , bool transB = false>
        void __cpu_gemm_strided_batch( 
            const int M , const int N , const int K ,
            const T * A , const int stepA , const T * B , const int stepB , T * C , const int stepC , const T alpha = 1 , const T beta = 0 , const int batch = 1){
            for (int b = 0 ; b < batch ; b++){ // just a trivial implementation
                __cpu_gemm<T , transA , transB>(
                    M , N , K ,
                    A + b * stepA , B + b * stepB , C + b * stepC ,
                    alpha , beta);
            }
        }


        template <typename T>
        inline cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            const T* alpha,
            const T* A,
            int lda,
            long long int strideA,
            const T* B,
            int ldb,
            long long int strideB,
            const T* beta,
            T* C,
            int ldc,
            long long int strideC,
            int batchCount){
            
            if constexpr (std::is_same_v<T , float>){
                return cublasSgemmStridedBatched(handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    A,
                    lda,
                    strideA,
                    B,
                    ldb,
                    strideB,
                    beta,
                    C,
                    ldc,
                    strideC,
                    batchCount);
            }
            else if constexpr (std::is_same_v<T , double>){
                return cublasDgemmStridedBatched(handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    A,
                    lda,
                    strideA,
                    B,
                    ldb,
                    strideB,
                    beta,
                    C,
                    ldc,
                    strideC,
                    batchCount);
            }
        }

        template <typename T>
        cublasStatus_t cublasGemm(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            const T* alpha,
            const T* A,
            int lda,
            const T* B,
            int ldb,
            const T* beta,
            T* C,
            int ldc){
                if constexpr (std::is_same_v<T , float>){
                    return cublasSgemm(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc);
                }
                else if constexpr (std::is_same_v<T , double>){
                    return cublasDgemm(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc);
                }

        }


        template <typename T>
        __global__ void __setup_2dstride_index(
            const T ** Aarray ,
            const T ** Barray ,
            T ** Carray ,
            const T * A,
            const T * B,
            T * C,
            long long int strideAx ,
            long long int strideAy ,
            long long int strideBx ,
            long long int strideBy ,
            long long int strideCx ,
            long long int strideCy ,
             int batchCountx ,
             int size
        ){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index < size){
                int batchx = index % batchCountx;
                int batchy = index / batchCountx;
                Aarray[index] = A + batchx * strideAx + batchy * strideAy;
                Barray[index] = B + batchx * strideBx + batchy * strideBy;
                Carray[index] = C + batchx * strideCx + batchy * strideCy;
            }
        }

        template <typename T>
        cublasStatus_t cublasGemmBatched(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            const T* alpha,
            const T* const Aarray[],
            int lda,
            const T* const Barray[],
            int ldb,
            const T* beta,
            T* const Carray[],
            int ldc,
            int batchCount){
                if constexpr (std::is_same_v<T , float>){
                    return cublasSgemmBatched(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        Aarray,
                        lda,
                        Barray,
                        ldb,
                        beta,
                        Carray,
                        ldc,
                        batchCount);
                }
                else if constexpr (std::is_same_v<T , double>){
                    return cublasDgemmBatched(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        Aarray,
                        lda,
                        Barray,
                        ldb,
                        beta,
                        Carray,
                        ldc,
                        batchCount);
                }

        }


        template <typename T>
        cublasStatus_t cublasGemm2DStridedBatched(cublasHandle_t handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            const T* alpha,
            const T ** Aarray,
            const T* A,
            int lda,
            long long int strideAx,
            long long int strideAy,
            const T ** Barray,
            const T* B,
            int ldb,
            long long int strideBx,
            long long int strideBy,
            const T* beta,
            T ** Carray,
            T* C,
            int ldc,
            long long int strideCx,
            long long int strideCy,
            int batchCountx,
            int batchCounty){

            
            int a = 0;
            for(int i = 0; i < batchCounty;i++){
                for(int j = 0; j < batchCountx;j++){
                    a = max(a , (int)(j * strideAx + i * strideAy));
                    Aarray[i * batchCountx + j] = A + j * strideAx + i * strideAy;
                    Barray[i * batchCountx + j] = B + j * strideBx + i * strideBy;
                    Carray[i * batchCountx + j] = C + j * strideCx + i * strideCy;
                }
            }
            std::cout << a << std::endl;

            return cublasGemmBatched<T>(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                Aarray,
                lda,
                Barray,
                ldb,
                beta,
                Carray,
                ldc,
                batchCountx * batchCounty
            );


        }

    }

}

#endif