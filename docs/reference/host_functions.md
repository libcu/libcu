## #include <host_functions.h>

Also includes:
```
#include <cuda_runtime_api.h>
```

## Host Side
Prototype | Description | Tags
--- | --- | :---:
```bool gpuAssert(cudaError_t code, const char *action, const char *file = nullptr, int line = 0, bool abort = true);``` | xxxx
```int gpuGetMaxGflopsDevice();``` | xxxx
```char **cudaDeviceTransferStringArray(size_t length, char *const value[], cudaError_t *error = nullptr);``` | xxxx
```#define cudaErrorCheck(x)``` | xxxx
```#define cudaErrorCheckA(x)``` | xxxx
```#define cudaErrorCheckF(x, f)``` | xxxx
```#define cudaErrorCheckLast()``` | xxxx