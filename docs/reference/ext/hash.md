## #include <ext\hash.h>

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ void hashInit(hash_t *h);``` | xxxx
```__device__ void *hashInsert(hash_t *h, const char *key, void *data);``` | xxxx
```__device__ void *hashFind(hash_t *h, const char *key);``` | xxxx
```__device__ void hashClear(hash_t *h);``` | xxxx
```#define hashFirst(h)``` | xxxx
```#define hashNext(e)``` | xxxx
```#define hashData(e)``` | xxxx
```#define HASHINIT``` | xxxx
