## #include <fcntlcu.h>

Also includes:
```
#include <fcntl.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ int fcntl(int fd, int cmd, ...);``` | Do the file control operation described by CMD on FD. The remaining arguments are interpreted depending on CMD. | #stdarg
```__device__ int fcntl64(int fd, int cmd, ...);``` | Do the file control operation described by CMD on FD. The remaining arguments are interpreted depending on CMD. | #stdarg
```__device__ int open(const char *file, int oflag, ...);``` | Open FILE and return a new file descriptor for it, or -1 on error. OFLAG determines the type of access used. If O_CREAT is on OFLAG, the third argument is taken as a 'mode_t', the mode of the created file. | #stdarg
```__device__ int open64(const char *file, int oflag, ...);``` | Open FILE and return a new file descriptor for it, or -1 on error. OFLAG determines the type of access used. If O_CREAT is on OFLAG, the third argument is taken as a 'mode_t', the mode of the created file. | #stdarg
```#define creat(file, mode)``` | Create and open FILE, with mode MODE.  This takes an 'int' MODE argument because that is what `mode_t' will be widened to. | #branch-isdevicepath
```#define creat64(file, mode)``` | Create and open FILE, with mode MODE.  This takes an 'int' MODE argument because that is what `mode_t' will be widened to. | #branch-isdevicepath #file64