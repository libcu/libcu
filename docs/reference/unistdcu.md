## #include <unistdcu.h>

Also includes:
```
#include <unistd.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ int access(const char *name, int type);``` | Test for access to NAME using the real UID and real GID. | #sentinel-isdevicepath
```__device__ off_t lseek(int fd, off_t offset, int whence);``` | Move FD's file position to OFFSET bytes from the beginning of the file (if WHENCE is SEEK_SET),
the current position (if WHENCE is SEEK_CUR), or the end of the file (if WHENCE is SEEK_END). Return the new file position. | #sentinel-isdevicehandle
```__device__ off64_t lseek64(int fd, off64_t offset, int whence);``` | Move FD's file position to OFFSET bytes from the beginning of the file (if WHENCE is SEEK_SET),
the current position (if WHENCE is SEEK_CUR), or the end of the file (if WHENCE is SEEK_END). Return the new file position. | #sentinel-isdevicehandle
```__device__ int close(int fd);``` | Close the file descriptor FD. | #sentinel-isdevicehandle
```__device__ size_t read(int fd, void *buf, size_t nbytes, bool wait = true);``` | Read NBYTES into BUF from FD.  Return the number read, -1 for errors or 0 for EOF. | #sentinel-isdevicehandle
```__device__ size_t write(int fd, void *buf, size_t nbytes, bool wait = true);``` | Write N bytes of BUF to FD.  Return the number written, or -1. | #sentinel-isdevicehandle
```__device__ int pipe(int pipedes[2]);``` | Create a one-way communication channel (pipe). | #notsupported
```__device__ unsigned int alarm(unsigned int seconds);``` | Schedule an alarm. | #notsupported
```__device__ void usleep(unsigned long milliseconds);``` | Make the process sleep for SECONDS seconds, or until a signal arrives and is not ignored.
```__device__ void sleep(unsigned int seconds);``` | Make the process sleep for SECONDS seconds, or until a signal arrives and is not ignored.
```__device__ int pause(void);``` | Suspend the process until a signal arrives. | #notsupported
```__device__ int chown(const char *file, uid_t owner, gid_t group);``` | Change the owner and group of FILE. | #sentinel-isdevicepath
```__device__ int chdir(const char *path);``` | Change the process's working directory to PATH.
```__device__ char *getcwd(char *buf, size_t size);``` | Get the pathname of the current working directory, and put it in SIZE bytes of BUF.
```__device__ int dup(int fd);``` | Duplicate FD, returning a new file descriptor on the same file. | #sentinel-isdevicehandle
```__device__ int dup2(int fd, int fd2);``` | Duplicate FD to FD2, closing FD2 and making it open on the same file. | #sentinel-isdevicehandle
```__device__ char **__environ;``` | NULL-terminated array of "NAME=VALUE" environment variables.
```__device__ void exit(int status);``` | Terminate program execution with the low-order 8 bits of STATUS. | #duplicate
```__device__ long int pathconf(const char *path, int name);``` | Get file-specific configuration information about PATH. | #notsupported
```__device__ long int fpathconf(int fd, int name);``` | Get file-specific configuration about descriptor FD. | #notsupported
```__device__ int unlink(const char *filename);``` | Remove the link FILENAME. | #sentinel-isdevicepath
```__device__ int rmdir(const char *path);``` | Remove the directory PATH. | #sentinel-isdevicepath
