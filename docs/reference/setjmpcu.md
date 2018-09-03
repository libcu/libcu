## #include <setjmpcu.h>

Also includes:
```
#include <setjmp.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ int setjmp(jmp_buf env);``` | Store the calling environment in ENV, also saving the signal mask. Return 0. | #notimplemented
```__device__ int __sigsetjmp(struct __jmp_buf_tag env[1], int savemask);``` | Store the calling environment in ENV, also saving the signal mask if SAVEMASK is nonzero.  Return 0. This is the internal name for `sigsetjmp'. | #notsupported
```__device__ int _setjmp(struct __jmp_buf_tag env[1]);``` | Store the calling environment in ENV, not saving the signal mask. Return 0. | #notsupported
```__device__ void longjmp(jmp_buf env, int val);``` | Jump to the environment saved in ENV, making the `setjmp' call there return VAL, or 1 if VAL is 0. | #notimplemented
