## #include <crtdefscu.h>

Also includes:
```
#include <crtdefs.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```#define _ROUNDT(x, T)``` | Memory allocation - rounds to the type in T
```#define _ROUND8(x)``` | Memory allocation - rounds up to 8
```#define _ROUND64(x)``` | Memory allocation - rounds up to 64
```#define _ROUNDN(x, size)``` | Memory allocation - rounds up to "size"
```#define _ROUNDDOWN8(x)``` | Memory allocation - rounds down to 8
```#define _ROUNDDOWNN(x, size)``` | Memory allocation - rounds down to "size"
```#define HASALIGNMENT8(x)``` | Test to see if you are on aligned boundary, affected by BYTEALIGNED4
```#define _LENGTHOF(symbol)``` | Returns the length of an array at compile time (via math)
```#define UNUSED_SYMBOL(x)``` | Removes compiler warning for unused parameter(s)
```#define UNUSED_SYMBOL2(x,y)``` | Removes compiler warning for unused parameter(s)
```__device__ void libcuReset();``` | Resets/shutdown libcu environment
