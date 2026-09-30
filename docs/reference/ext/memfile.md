## #include <ext\memfile.h>

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__host_device__ int memfileRead(vsysfile *p, void *buf, int amount, int64_t offset);``` | Read data from the file.
```__host_device__ int memfileWrite(vsysfile *p, const void *buf, int amount, int64_t offset);``` | Write data to the file.
```__host_device__ int memfileTruncate(vsysfile *p, int64_t size);``` | Truncate the file.
```__host_device__ int memfileClose(vsysfile *p);``` | Close the file.
```__host_device__ int memfileFileSize(vsysfile *p, int64_t *size);``` | Query the size of the file in bytes.
```__host_device__ int memfileOpen(vsystem *vsys, const char *name, vsysfile *p, int flags, int spill);``` | Open a journal file.
```__host_device__ void memfileMemOpen(vsysfile *p);``` | Open an in-memory journal file.
#if defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
```__host_device__ int memfileCreate(vsysfile *p);``` | xxx
#endif
```__host_device__ int memfileIsInMemory(vsysfile *p);``` | Return true if this "journal file" is currently stored in heap memory, or false otherwise.
```__host_device__ int memfileSize(vsystem *p);``` | Return the number of bytes required to store a JournalFile that uses vsystem to create the underlying on-disk files.
