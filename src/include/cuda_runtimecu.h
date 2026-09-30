/*
cuda_runtimecu.h - cuda_runtime
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef __CUDA_RUNTIMECU_H__
#define __CUDA_RUNTIMECU_H__
#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
#ifdef __cplusplus
extern "C" {
#endif

	extern bool gpuAssert(cudaError_t code, const char *action, const char *file = nullptr, int line = 0, bool abort = true);
	extern int gpuGetMaxGflopsDevice();
	extern char **cudaDeviceTransferStringArray(size_t length, char *const value[], cudaError_t *error = nullptr);

#ifdef __cplusplus
}
#endif

#define cudaErrorCheck(x) { gpuAssert((x), #x, __FILE__, __LINE__, true); }
#define cudaErrorCheckA(x) { gpuAssert((x), #x, __FILE__, __LINE__, false); }
#define cudaErrorCheckF(x, f) { if (!gpuAssert((x), #x, __FILE__, __LINE__, false)) f; }
#define cudaErrorCheckLast() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }

#endif /* __CUDA_RUNTIMECU_H__ */