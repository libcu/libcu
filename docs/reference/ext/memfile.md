## #include <ext\memfile.h>

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__constant__ int __sizeofMemfile_t;``` | xxxx
```__device__ void memfileOpen(memfile_t *f);``` | xxxx
```__device__ void memfileRead(memfile_t *f, void *buffer, int amount, int64_t offset);``` | xxxx
```__device__ bool memfileWrite(memfile_t *f, const void *buffer, int amount, int64_t offset);``` | xxxx
```__device__ void memfileTruncate(memfile_t *f, int64_t size);``` | xxxx
```__device__ void memfileClose(memfile_t *f);``` | xxxx
```__device__ int64_t memfileGetFileSize(memfile_t *f);``` | xxxx
