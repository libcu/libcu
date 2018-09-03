## #include <sys\stat.h>

Also includes:
```
#include <sys\time.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ int stat(const char *__restrict file, struct stat *__restrict buf);``` | Get file attributes about FILE and put them in BUF. If FILE is a symbolic link, do not follow it. | #sentinel-isdevicepath
```#define lstat(file, buf)``` | Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF. | #sentinel-isdevicepath
```__device__ int fstat(int fd, struct stat *buf);``` | Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF. | #sentinel-isdevicehandle
```__device__ int stat64(const char *__restrict file, struct stat64 *__restrict buf);``` | Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF. | #sentinel-isdevicepath #file64
```__device__ int fstat64(int fd, struct stat64 *buf);``` | Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF. | #sentinel-isdevicehandle #file64
```__device__ int chmod(const char *file, mode_t mode);``` | Set file access permissions for FILE to MODE. If FILE is a symbolic link, this affects its target instead. | #sentinel-isdevicepath
```__device__ mode_t umask(mode_t mask);``` | Set the file creation mask of the current process to MASK, and return the old creation mask.
```__device__ int mkdir(const char *path, mode_t mode);``` | Create a new directory named PATH, with permission bits MODE. | #sentinel-isdevicepath
```__device__ int mkfifo(const char *path, mode_t mode);``` | Create a new FIFO named PATH, with permission bits MODE. | #sentinel-isdevicepath