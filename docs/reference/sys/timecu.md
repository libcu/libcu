## #include <sys\timecu.h>

Also includes:
```
#include <sys\time.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```extern __device__ int gettimeofday_(struct timeval *__restrict tv, void *tz);``` | Get the current time of day and timezone information, putting it into *TV and *TZ.  If TZ is NULL, *TZ is not filled. Returns 0 on success, -1 on errors.
