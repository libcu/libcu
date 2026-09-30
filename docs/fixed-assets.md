# fixed-assets

Libcu uses fixed-assets to solve two issues.

1. Typically librarys have a need for initialize, and shutdown methods to prepare and un-prepare the environment respectivly while at the same time tring to make the api simple to use. In somecases a library like SQLite will auto-initalize upon first use, but this approach adds additional call overhead, and boiler plate code in each method.

   Libcu uses fixed-assets, and a libcuReset() method, which pre-initialize values in the original image, and resets values to originals respectivly.

2. Files, streams, and other assets can be created on either the host or device, but are indistinguishable when looking just at the handle or pointer. Libcu should be smart about keeping device assets on the device and host assets in the host.

    Libcu uses fixed-assets instead of allocating at runtime, and determine if a handle or pointer is on device or host based on the location of the pointer vs the fixed-assets locations.


The following files define some of these behaviors.


## crtdefscu.h

This header includes most of the libcu base definitions, of which are also the primary fixed-asset definitions.

`LIBCU_MAXFILESTREAM` defines the maximum number of files and streams as hard-assets
```
#ifndef LIBCU_MAXFILESTREAM
#define LIBCU_MAXFILESTREAM 10
#endif
```

`LIBCU_MAXHOSTPTR` defines the maximum number of host pointers as hard-assets
```
#ifndef LIBCU_MAXHOSTPTR
#define LIBCU_MAXHOSTPTR 10
#endif
```

### IsHost support
These definitions determind if an assets is on the host or device:
* `__cwd` - holds the current device relative base, this value should not be modified directly.
* `ISHOSTENV(path)` - if `name` begins with `:` then it is a device name otherwise `name` is a host name.
* `ISHOSTPATH(path)` - if `path` begins with `:\` as an absolute path, or if `path` is relative and a device relative base has been set in `__cwd`, then `path` is a device path otherwise `path` is a host path.
* `ISHOSTHANDLE(handle)` - host if `handle` falls outside fixed-asset handle range.
* `ISHOSTPTR(ptr)` - host if `ptr` falls outside fixed-asset handle range.
```
/* IsHost support  */
extern "C" __device__ char __cwd[];
#define ISHOSTENV(name) (name[0] != ':')
#define ISHOSTPATH(path) ((path)[1] == ':' || ((path)[0] != ':' && __cwd[0] == 0))
#define ISHOSTHANDLE(handle) (handle < INT_MAX-LIBCU_MAXFILESTREAM)
#define ISHOSTPTR(ptr) ((hostptr_t *)(ptr) >= __iob_hostptrs && (hostptr_t *)(ptr) <= __iob_hostptrs+LIBCU_MAXHOSTPTR)
```

### Host pointer support
These methods are used for working with the HostPtr system:
* `newhostptr(p)` - returns a device `hostptr_t` object which wraps a host `p` pointer
* `freehostptr(p)` - frees the device `hostptr_t` object in `p`
* `hostptr(p)` - returns the wrapped host pointer contained in `p`
```
/* Host pointer support  */
extern "C" __constant__ hostptr_t __iob_hostptrs[LIBCU_MAXHOSTPTR];
extern "C" __device__ hostptr_t *__hostptrGet(void *host);
extern "C" __device__ void __hostptrFree(hostptr_t *p);
template <typename T> __forceinline __device__ T *newhostptr(T *p) { return (T *)(p ? __hostptrGet(p) : nullptr); }
template <typename T> __forceinline __device__ void freehostptr(T *p) { if (p) __hostptrFree((hostptr_t *)p); }
template <typename T> __forceinline __device__ T *hostptr(T *p) { return (T *)(p ? ((hostptr_t *)p)->host : nullptr); }
```

### Library support
This method will reset the libcu library:
* `libcuReset()` - resets the library
```
/* Reset library */
extern "C" __device__ void libcuReset();
```


## stdiocu.h

This header is included when working with streams.

### IsHost support
These methods are used when working with device side streams:
* `ISHOSTFILE(stream)` - host if `stream` falls outside the fixed-asset stream range.
* `stdin` - alias to the device standard input stream
* `stdout` - alias to the device standard output stream
* `stderr` - alias to the device standard error stream
```
/* IsHost support  */
#define ISHOSTFILE(stream) ((cuFILE*)(stream) < __iob_streams || (cuFILE*)(stream) > __iob_streams + LIBCU_MAXFILESTREAM+3)
extern __constant__ cuFILE __iob_streams[LIBCU_MAXFILESTREAM+3];
#undef stdin
#undef stdout
#undef stderr
#define stdin  ((FILE*)&__iob_streams[0]) /* Standard input stream.  */
#define stdout ((FILE*)&__iob_streams[1]) /* Standard output stream.  */
#define stderr ((FILE*)&__iob_streams[2]) /* Standard error stream.  */
```


## fsystem transparent sub-system

Fsystem is a transparent sub-system which defines the device side filesystem.

### State
These are the initalization values for the filesystem:
* `__iob_files` - folds the file handles which access the filesystem
* `__cwd` - holds the current device relative base
* `__iob_root` - represents the root `dirEnt_t` node
* `__iob_dir` - holds all files and directories in a hash<`path`, `dirEnt_t`>
```
__constant__ file_t __iob_files[LIBCU_MAXFILESTREAM];
__device__ char __cwd[MAX_PATH] = ":\\";
__device__ dirEnt_t __iob_root = { { 0, 0, 0, 1, ":\\" }, nullptr, nullptr };
__device__ hash_t __iob_dir = HASHINIT;
```

### File support
These methods are used internally when working the device side filesystem:
* `fsystemReset()` - resets the filesystem back to original state
* `GETFD(fd)` - translates a file pointer to a file handle
* `GETFILE(fd)` - transaltes a file handle to a file pointer
```
__device__ void fsystemReset();

/* File support  */
extern __device__ dirEnt_t __iob_root;
extern __constant__ file_t __iob_files[LIBCU_MAXFILESTREAM];
#define GETFD(fd) (INT_MAX-(fd))
#define GETFILE(fd) (&__iob_files[GETFD(fd)])
```