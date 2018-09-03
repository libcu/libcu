## #include <stddefcu.h>

Also includes:
```
#include <stddef.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```#define panic(fmt, ...)``` | xxxx
```__device__ void *tagalloc(void *tag, size_t size)``` | xxxx
```__device__ void tagfree(void *tag, void *p)``` | xxxx
```__device__ void *tagrealloc(void *tag, void *old, size_t size)``` | xxxx