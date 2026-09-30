#include <stdlibcu.h>
#include <ext/hash.h>
#include <stringcu.h>
#include <ctypecu.h>
#include <assert.h>

/* Turn bulk memory into a hash table object by initializing the fields of the hash_t structure.
**
** "h" is a pointer to the hash table that is to be initialized.
*/
__host_device__ void hashInit(hash_t *h) {
	assert(h);
	h->first = nullptr;
	h->count = 0;
	h->tableSize = 0;
	h->table = nullptr;
}

/* Remove all entries from a hash table.  Reclaim all memory. Call this routine to delete a hash table or to reset a hash table
** to the empty state.
*/
__host_device__ void hashClear(hash_t *h) {
	hashElem_t *elem = h->first; // For looping over all elements of the table
	h->first = nullptr;
	free(h->table); h->table = nullptr;
	h->tableSize = 0;
	while (elem) {
		hashElem_t *nextElem = elem->next;
		free(elem);
		elem = nextElem;
	}
	h->count = 0;
}

/* The hashing function.  */
__host_device__ static unsigned int getHashCode(const char *key) {
	/* Knuth multiplicative hashing.  (Sorting & Searching, p. 510). 0x9e3779b1 is 2654435761 which is the closest prime number to (2**32)*golden_ratio, where golden_ratio = (sqrt(5) - 1)/2. */
	unsigned int h = 0;
	unsigned char c;
	while ((c = (unsigned char)*key++)) { h += __curtUpperToLower[c]; h *= 0x9e3779b1; }
	return h;
}

/* Link "newElem" element into the hash table "h".  If "entry!=0" then also insert "newElem" into the "entry" hash bucket. */
static __host_device__ void insertElement(hash_t *h, hash_t::htable_t *entry, hashElem_t *newElem) {
	hashElem_t *headElem; // First element already in entry
	if (entry) {
		headElem = entry->count ? entry->chain : nullptr;
		entry->count++;
		entry->chain = newElem;
	}
	else
		headElem = nullptr;
	if (headElem) {
		newElem->next = headElem;
		newElem->prev = headElem->prev;
		if (headElem->prev) headElem->prev->next = newElem;
		else h->first = newElem;
		headElem->prev = newElem;
	}
	else {
		newElem->next = h->first;
		if (h->first) h->first->prev = newElem;
		newElem->prev = nullptr;
		h->first = newElem;
	}
}

/* Resize the hash table so that it cantains "new_size" buckets.
**
** The hash table might fail to resize if malloc() fails or if the new size is the same as the prior size.
** Return true if the resize occurs and false if not.
*/
static __host_device__ bool rehash(hash_t *h, unsigned int newSize) {
#if MALLOC_SOFT_LIMIT > 0
	if (newSize * sizeof(htable_t) > MALLOC_SOFT_LIMIT)
		newSize = MALLOC_SOFT_LIMIT / sizeof(htable_t);
	if (newSize == h->tableSize) return false;
#endif
	hash_t::htable_t *newTable = (hash_t::htable_t *)malloc(newSize * sizeof(hash_t::htable_t)); // The new hash table
	if (!newTable)
		return false;
	free(h->table);
	h->table = newTable;
	h->tableSize = newSize = (int)_msize(newTable) / sizeof(hash_t::htable_t);
	memset(newTable, 0, newSize * sizeof(hash_t::htable_t));
	hashElem_t *elem, *nextElem;
	for (elem = h->first, h->first = nullptr; elem; elem = nextElem) {
		unsigned int hash = getHashCode(elem->key) % newSize;
		nextElem = elem->next;
		insertElement(h, &newTable[hash], elem);
	}
	return true;
}

/* This function (for internal use only) locates an element in an hash table that matches the given key.  The hash for this key is
** also computed and returned in the "h" parameter.
*/
static __host_device__ hashElem_t *findElementWithHash(const hash_t *h, const char *key, unsigned int *hash) {
	hashElem_t *elem;
	unsigned int hash2; // The computed hash
	int count; // Number of elements left to test
	if (h->table) {
		hash2 = getHashCode(key) % h->tableSize;
		hash_t::htable_t *entry = &h->table[hash2];
		elem = entry->chain;
		count = entry->count;
	}
	else {
		hash2 = 0;
		elem = h->first;
		count = h->count;
	}
	*hash = hash2;
	while (count--) {
		assert(elem);
		if (!stricmp(elem->key, key))
			return elem;
		elem = elem->next;
	}
	return nullptr;
}

/* Remove a single entry from the hash table given a pointer to that element and a hash on the element's key. */
static __host_device__ void removeElementGivenHash(hash_t *h, hashElem_t *elem, unsigned int hash) {
	if (elem->prev)
		elem->prev->next = elem->next;
	else
		h->first = elem->next;
	if (elem->next)
		elem->next->prev = elem->prev;
	if (h->table)
	{
		hash_t::htable_t *entry = &h->table[hash];
		if (entry->chain == elem)
			entry->chain = elem->next;
		entry->count--;
		assert(entry->count >= 0);
	}
	free(elem);
	h->count--;
	if (!h->count) {
		assert(!h->first);
		assert(!h->count);
		hashClear(h);
	}
}

/* Attempt to locate an element of the hash table "h" with a key that matches pKey.  Return the data for this element if it is
** found, or nullptr if there is no match.
*/
__host_device__ void *hashFind(hash_t *h, const char *key) {
	assert(h);
	assert(key);
	unsigned int hash; // A hash on key
	hashElem_t *elem = findElementWithHash(h, key, &hash);
	return elem ? elem->data : nullptr;
}

/* Insert an element into the hash table "h".  The key is "key" and the data is "data".
**
** If no element exists with a matching key, then a new element is created and NULL is returned.
**
** If another element already exists with the same key, then the new data replaces the old data and the old data is returned.
** The key is not copied in this instance.  If a malloc fails, then the new data is returned and the hash table is unchanged.
**
** If the "data" parameter to this function is NULL, then the element corresponding to "key" is removed from the hash table.
*/
__host_device__ void *hashInsert(hash_t *h, const char *key, void *data) {
	assert(h);
	assert(key);
	unsigned int hash; // the hash of the key modulo hash table size
	hashElem_t *elem = findElementWithHash(h, key, &hash);
	if (elem) {
		void *oldData = elem->data;
		if (!data)
			removeElementGivenHash(h, elem, hash);
		else {
			elem->data = data;
			elem->key = key;
		}
		return oldData;
	}
	if (!data)
		return nullptr;
	hashElem_t *newElem = (hashElem_t *)malloc(sizeof(hashElem_t));
	if (!newElem)
		return data;
	newElem->key = key;
	newElem->data = data;
	h->count++;
	if (h->count >= 10 && h->count > 2 * h->tableSize) {
		if (rehash(h, h->count * 2)) {
			assert(h->tableSize > 0);
			hash = getHashCode(key) % h->tableSize;
		}
	}
	insertElement(h, h->table ? &h->table[hash] : nullptr, newElem);
	return nullptr;
}
