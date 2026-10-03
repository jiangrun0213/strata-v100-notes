#include <cuda_runtime.h>
#include <cstdio>
int main(){
  int n=0; cudaError_t e=cudaGetDeviceCount(&n);
  if(e!=cudaSuccess){ printf("CUDA init 失败: %s\n", cudaGetErrorString(e)); return 1; }
  printf("CUDA 正常, 检测到 %d 张卡: ", n);
  for(int i=0;i<n;i++){ cudaDeviceProp p; cudaGetDeviceProperties(&p,i); printf("[%s] ", p.name); }
  printf("\n"); return 0;
}
