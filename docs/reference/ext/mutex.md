## #include <ext\mutex.h>

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ void mutex_lock(unsigned int *mutex);``` | Mutex with exponential back-off.
```__device__ void mutex_unlock(unsigned int *mutex);``` | Mutex unlock.
