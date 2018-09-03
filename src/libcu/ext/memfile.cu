#include <ext/memfile.h>
#include <stdlibcu.h>
#include <assert.h>

/* Forward references to internal structures */
typedef struct memfile_t memfile_t;
typedef struct filePoint_t filePoint_t;
typedef struct fileChunk_t fileChunk_t;

/* The rollback journal is composed of a linked list of these structures.
**
** The zChunk array is always at least 8 bytes in size - usually much more. Its actual size is stored in the MemJournal.nChunkSize variable.
*/
struct fileChunk_t {
	fileChunk_t *next;			// Next chunk in the journal
	uint8_t chunk[8];			// Content of this chunk
};

/* By default, allocate this many bytes of memory for each FileChunk object. */
#define MEMFILE_DFLT_FILECHUNKSIZE 1024

/* For chunk size nChunkSize, return the number of bytes that should be allocated for each FileChunk structure. */
#define fileChunkSize(chunkSize) (sizeof(fileChunk_t) + ((chunkSize) - 8))

/* An instance of this object serves as a cursor into the rollback journal. The cursor can be either for reading or writing. */
struct filePoint_t {
	int64_t offset;				// Offset from the beginning of the file
	fileChunk_t *chunk;			// Specific chunk into which cursor points
};

/* This structure is a subclass of sqlite3_file. Each open memory-journal is an instance of this class. */
typedef struct memfile_t {
	const vsysfile_methods *method;			// Parent class. MUST BE FIRST
	int chunkSize;              // In-memory chunk-size
	int spill;                  // Bytes of data before flushing
	int size;                   // Bytes of data currently in memory
	fileChunk_t *first;			// Head of in-memory chunk-list
	filePoint_t endpoint;		// Pointer to the end of the file
	filePoint_t readpoint;		// Pointer to the end of the last xRead()
	//
	int flags;					// xOpen flags
	vsystem *vsys;				// The "real" underlying VFS
	const char *name;			// Name of the journal file
} memfile_t;

//__constant__ const int __sizeofMemfile_t = sizeof(memfile_t);

//__host_device__ void memfileOpen(memfile_t *f)
//{
//	memset(f, 0, sizeof(memfile_t));
//	f->opened = true;
//}

#define RC_OK 0 
#define RC_IOERR 10 
#define RC_IOERR_SHORT_READ (RC_IOERR | (2<<8))
#define RC_IOERR_NOMEM_BKPT 10
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

/* Read data from the in-memory journal file.  This is the implementation of the sqlite3_vfs.xRead method. */
__host_device__ int memfileRead(vsysfile *p, void *buf, int amount, int64_t offset) {
	memfile_t *f = (memfile_t *)p;
#if defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
	if (amount + offset > p->endpoint.offset)
		return RC_IOERR_SHORT_READ;
#endif

	// never try to read past the end of an in-memory file
	fileChunk_t *chunk;
	assert(amount + offset <= f->endpoint.offset);
	assert(!f->readpoint.offset || f->readpoint.chunk);
	if (f->readpoint.offset != offset || !offset) { int64_t off = 0; for (chunk = f->first; ALWAYS_(chunk) && (off + f->chunkSize) <= offset; chunk = chunk->next) off += f->chunkSize; }
	else { chunk = f->readpoint.chunk; assert(chunk); }

	int chunkOffset = (int)(offset % f->chunkSize);
	uint8_t *out = (uint8_t *)buf;
	int read = amount;
	do {
		int space = f->chunkSize - chunkOffset;
		int copy = MIN(read, (f->chunkSize - chunkOffset));
		memcpy(out, chunk->chunk + chunkOffset, copy);
		out += copy;
		read -= space;
		chunkOffset = 0;
	} while (read >= 0 && (chunk = chunk->next) && read > 0);
	f->readpoint.offset = chunk ? offset + amount : 0;
	f->readpoint.chunk = chunk;
	return RC_OK;
}

/* Free the list of FileChunk structures headed at MemJournal.pFirst. */
static __host_device__ void memfileFreeChunks(memfile_t *f) {
	fileChunk_t *next; for (fileChunk_t *p = f->first; p; p = next) {
		next = p->next;
		free(p);
	}
	f->first = nullptr;
}

/* Flush the contents of memory to a real file on disk. */
static __host_device__ int memfileCreateFile(memfile_t *f) {
	memfile_t copy = *f;
	memset(f, 0, sizeof(memfile_t));
	int rc = __extsystem.vsys_open(copy.vsys, copy.name, (vsysfile *)f, copy.flags, 0);
	if (!rc) {
		int chunkSize = copy.chunkSize;
		int64_t off = 0;
		for (fileChunk_t *p = copy.first; p; p = p->next) {
			if (off + chunkSize > copy.endpoint.offset)
				chunkSize = copy.endpoint.offset - off;
			rc = __extsystem.vsys_write((vsysfile *)f, (uint8_t *)p->chunk, chunkSize, off);
			if (rc) break;
			off += chunkSize;
		}
		// No error has occurred. Free the in-memory buffers.
		if (!rc) memfileFreeChunks(&copy);
	}
	if (rc) {
		// If an error occurred while creating or writing to the file, restore the original before returning. This way, SQLite uses the in-memory
		// journal data to roll back changes made to the internal page-cache before this function was called.
		__extsystem.vsys_close((vsysfile *)f);
		*f = copy;
	}
	return rc;
}

/* Write data to the file. */
__host_device__ int memfileWrite(vsysfile *p, const void *buf, int amount, int64_t offset) {
	memfile_t *f = (memfile_t *)p;
	// If the file should be created now, create it and write the new data into the file on disk.
	if (f->spill > 0 && amount + offset > f->spill) {
		int rc = memfileCreateFile(f);
		if (!rc) rc = __extsystem.vsys_write(p, buf, amount, offset);
		return rc;
	}
	// If the contents of this write should be stored in memory
	else {
		// An in-memory journal file should only ever be appended to. Random access writes are not required. The only exception to this is when
		// the in-memory journal is being used by a connection using the atomic-write optimization. In this case the first 28 bytes of the
		// journal file may be written as part of committing the transaction.
		assert(offset == f->endpoint.offset || !offset);
#if defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
		if (!offset && f->first) {
			assert(f->chunkSize > amount);
			memcpy((uint8_t *)f->first->chunk, buf, amount);
		}
		else
#else
		assert(offset > 0 || !f->first);
#endif
		{
			int write = amount;
			uint8_t *b = (uint8_t *)buf;
			while (write > 0) {
				fileChunk_t *chunk = f->endpoint.chunk;
				int chunkOffset = (int)(f->endpoint.offset % f->chunkSize);
				int space = MIN(write, f->chunkSize - chunkOffset);

				if (!chunkOffset) {
					// New chunk is required to extend the file
					fileChunk_t *newChunk = (fileChunk_t *)malloc(fileChunkSize(f->chunkSize));
					if (!newChunk)
						return RC_IOERR_NOMEM_BKPT;
					newChunk->next = nullptr;
					if (chunk) { assert(f->first); chunk->next = newChunk; }
					else { assert(!f->first); f->first = newChunk; }
					f->endpoint.chunk = newChunk;
				}

				memcpy(&f->endpoint.chunk->chunk[chunkOffset], b, space);
				b += space;
				write -= space;
				f->endpoint.offset += space;
			}
			f->size = amount + offset;
		}
	}
	return RC_OK;
}

/* Truncate the file.
**
** If the journal file is already on disk, truncate it there. Or, if it is still in main memory but is being truncated to zero bytes in size, ignore
*/
__host_device__ int memfileTruncate(vsysfile *p, int64_t size) {
	memfile_t *f = (memfile_t *)p;
	if (ALWAYS_(!size)) {
		memfileFreeChunks(f);
		f->size = 0;
		f->endpoint.chunk = nullptr;
		f->endpoint.offset = 0;
		f->readpoint.chunk = nullptr;
		f->readpoint.offset = 0;
	}
	return RC_OK;
}

/* Close the file. */
__host_device__ int memfileClose(vsysfile *p) {
	memfile_t *f = (memfile_t *)p;
	memfileFreeChunks(f);
	return RC_OK;
}

/* Sync the file.
**
** If the real file has been created, call its xSync method. Otherwise,  syncing an in-memory journal is a no-op.
*/
static __host_device__ int memfileSync(vsysfile *p, int flags) {
	UNUSED_SYMBOL2(p, flags);
	return RC_OK;
}

/* Query the size of the file in bytes. */
__host_device__ int memfileFileSize(vsysfile *p, int64_t *size) {
	memfile_t *f = (memfile_t *)p;
	*size = (int64_t)f->endpoint.offset;
	return RC_OK;
}

/* Table of methods for MemJournal sqlite3_file object. */
static __hostb_device__ const struct vsysfile_methods _memFileMethods = {
	1,					// iVersion
	memfileClose,		// xClose
	memfileRead,		// xRead
	memfileWrite,		// xWrite
	memfileTruncate,	// xTruncate
	memfileSync,		// xSync
	memfileFileSize,	// xFileSize
	nullptr,			// xLock
	nullptr,			// xUnlock
	nullptr,			// xCheckReservedLock
	nullptr,			// xFileControl
	nullptr,			// xSectorSize
	nullptr,			// xDeviceCharacteristics
	nullptr,			// xShmMap
	nullptr,			// xShmLock
	nullptr,			// xShmBarrier
	nullptr,			// xShmUnmap
	nullptr,			// xFetch
	nullptr				// xUnfetch
};

/* Open a journal file.
**
** The behaviour of the journal file depends on the value of parameter nSpill. If nSpill is 0, then the journal file is always create and
** accessed using the underlying VFS. If nSpill is less than zero, then all content is always stored in main-memory. Finally, if nSpill is a
** positive value, then the journal file is initially created in-memory but may be flushed to disk later on. In this case the journal file is
** flushed to disk either when it grows larger than nSpill bytes in size, or when sqlite3JournalCreate() is called.
*/
__host_device__ int memfileOpen(vsystem *vsys, const char *name, vsysfile *p, int flags, int spill) {
	memfile_t *f = (memfile_t *)p;
	// Zero the file-handle object. If nSpill was passed zero, initialize it using the sqlite3OsOpen() function of the underlying VFS. In this
	// case none of the code in this module is executed as a result of calls made on the journal file-handle.
	memset(f, 0, sizeof(memfile_t));
	if (!spill)
		return __extsystem.vsys_open(vsys, name, p, flags, 0);

	if (spill > 0) f->chunkSize = spill;
	else { f->chunkSize = 8 + MEMFILE_DFLT_FILECHUNKSIZE - sizeof(fileChunk_t); assert(MEMFILE_DFLT_FILECHUNKSIZE == fileChunkSize(f->chunkSize)); }

	f->method = (const vsysfile_methods *)&_memFileMethods;
	f->spill = spill;
	f->flags = flags;
	f->name = name;
	f->vsys = vsys;
	return RC_OK;
}

/* Open an in-memory journal file. */
__host_device__ void memfileMemOpen(vsysfile *p) {
	memfileOpen(nullptr, nullptr, p, 0, -1);
}

#if 1 || defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
/* If the argument p points to a MemJournal structure that is not an in-memory-only journal file (i.e. is one that was opened with a +ve
** nSpill parameter or as SQLITE_OPEN_MAIN_JOURNAL), and the underlying  file has not yet been created, create it now.
*/
__host_device__ int memfileCreate(vsysfile *p) {
	memfile_t *f = (memfile_t *)p;
	int rc = RC_OK;
	if (f->method == &_memFileMethods && (
#ifdef ENABLE_ATOMIC_WRITE
		f->spill > 0
#else
		NEVER_(f->spill > 0) // While this appears to not be possible without ATOMIC_WRITE, the paths are complex, so it seems prudent to leave the test in as a NEVER(), in case our analysis is subtly flawed.
#endif
#ifdef ENABLE_BATCH_ATOMIC_WRITE
		|| (f->flags & SQLITE_OPEN_MAIN_JOURNAL)
#endif
		)) rc = memfileCreateFile(f);
	return rc;
}
#endif

/* The file-handle passed as the only argument is open on a journal file. Return true if this "journal file" is currently stored in heap memory, or false otherwise. */
__host_device__ int memfileIsInMemory(vsysfile *p) {
	return p->methods == &_memFileMethods;
}

/* Return the number of bytes required to store a JournalFile that uses vfs pVfs to create the underlying on-disk files. */
struct vsystem_partial {
	int version;			// Structure version number (currently 3)
	int sizeOsFile;			// Size of subclassed vsysfile
};
__host_device__ int memfileSize(vsystem *p) {
	return !p ? (int)sizeof(memfile_t) :
		MAX(((struct vsystem_partial *)p)->sizeOsFile, (int)sizeof(memfile_t));
}