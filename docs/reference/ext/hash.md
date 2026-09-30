## #include <ext\hash.h>

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ void hashInit(hash_t *h);``` | Turn bulk memory into a hash table object by initializing the fields of the Hash structure.
```__device__ void *hashInsert(hash_t *h, const char *key, void *data);``` | Insert an element into the hash table pH.  The key is pKey and the data is "data".
```__device__ void *hashFind(hash_t *h, const char *key);``` | Attempt to locate an element of the hash table pH with a key that matches pKey.  Return the data for this element if it is found, or NULL if there is no match.
```__device__ void hashClear(hash_t *h);``` | Remove all entries from a hash table.  Reclaim all memory. Call this routine to delete a hash table or to reset a hash table to the empty state.
```#define hashFirst(h)``` | Returns the first element of the hash.
```#define hashNext(e)``` | Returns the next element of the hash.
```#define hashData(e)``` | Returns the data of the hash element.
```#define HASHINIT``` | Used to initize a hash.
