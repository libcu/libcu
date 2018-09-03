#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\hash.h>
#include <assert.h>

static __global__ void g_ext_hash_test1() {
	printf("ext_hash_test1\n");

	///* Turn bulk memory into a hash table object by initializing the fields of the Hash structure. */
	//extern __device__ void hashInit(hash_t *h);
	///* Insert an element into the hash table pH.  The key is pKey and the data is "data". */
	//extern __device__ void *hashInsert(hash_t *h, const char *key, void *data);
	///* Attempt to locate an element of the hash table pH with a key that matches pKey.  Return the data for this element if it is found, or NULL if there is no match. */
	//extern __device__ void *hashFind(hash_t *h, const char *key);
	///* Remove all entries from a hash table.  Reclaim all memory. Call this routine to delete a hash table or to reset a hash table to the empty state. */
	//extern __device__ void hashClear(hash_t *h);
}
cudaError_t ext_hash_test1() { g_ext_hash_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
