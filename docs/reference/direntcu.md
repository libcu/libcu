## #include <direntcu.h>

Also includes: (portable)
```
#include <_dirent.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ DIR *opendir(const char *name);``` | Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. | #sentinel-ishostpath
```__device__ int closedir(DIR *dirp)``` | Close the directory stream DIRP. Return 0 if successful, -1 if not. | #sentinel-isdeviceptr
```__device__ struct dirent *readdir(DIR *dirp);``` | Read a directory entry from DIRP.  Return a pointer to a `struct dirent' describing the entry, or NULL for EOF or error.  The storage returned may be overwritten by a later readdir call on the same DIR stream. | #sentinel-isdeviceptr
```__device__ struct dirent64 *readdir64(DIR *dirp);``` | Read a directory entry from DIRP.  Return a pointer to a `struct dirent' describing the entry, or NULL for EOF or error.  The storage returned may be overwritten by a later readdir call on the same DIR stream. | #sentinel-isdeviceptr #file64
```__device__ void rewinddir(DIR *dirp)``` | Rewind DIRP to the beginning of the directory. | #sentinel-isdeviceptr
