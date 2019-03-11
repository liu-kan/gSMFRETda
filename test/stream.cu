#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  cudaStream_t stream1;  
  cudaStreamCreate(&stream1);
  cudaMemcpyAsync(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice,stream1);
  cudaMemcpyAsync(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice,stream1);

  cudaStream_t stream2;  
  cudaStreamCreate(&stream2);
    
  // Perform SAXPY on 1M elements
//   saxpy<<<(N+255)/256, 256,0,stream1>>>(N, 2.0f, d_x, d_y);
cudaStreamSynchronize(stream1);
  saxpy<<<(N+255)/256, 256,0,stream2>>>(N, 2.0f, d_x, d_y);
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}