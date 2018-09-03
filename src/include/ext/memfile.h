/*
memfile.h - xxx
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _EXT_MEMFILE_H
#define _EXT_MEMFILE_H
#include <stdint.h>
#ifdef  __cplusplus
extern "C" {
#endif
	/* Read data from the in-memory journal file.  This is the implementation of the sqlite3_vfs.xRead method. */
	extern __host_device__ int memfileRead(vsysfile *p, void *buf, int amount, int64_t offset);
	/* Write data to the file. */
	extern __host_device__ int memfileWrite(vsysfile *p, const void *buf, int amount, int64_t offset);
	/* Truncate the file. */
	extern __host_device__ int memfileTruncate(vsysfile *p, int64_t size);
	/* Close the file. */
	extern __host_device__ int memfileClose(vsysfile *p);
	/* Query the size of the file in bytes. */
	extern __host_device__ int memfileFileSize(vsysfile *p, int64_t *size);

	/* Open a journal file. */
	extern __host_device__ int memfileOpen(vsystem *vsys, const char *name, vsysfile *p, int flags, int spill); //: sqlite3JournalOpen
	/* Open an in-memory journal file. */
	extern __host_device__ void memfileMemOpen(vsysfile *p); //: sqlite3MemJournalOpen
#if defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
	extern __host_device__ int memfileCreate(vsysfile *p); //: sqlite3JournalCreate
#endif
	/* The file-handle passed as the only argument is open on a journal file. Return true if this "journal file" is currently stored in heap memory, or false otherwise. */
	extern __host_device__ int memfileIsInMemory(vsysfile *p); //: sqlite3JournalIsInMemory
	/* Return the number of bytes required to store a JournalFile that uses vfs pVfs to create the underlying on-disk files. */
	extern __host_device__ int memfileSize(vsystem *p); //: sqlite3JournalSize
#ifdef  __cplusplus
}
#endif
#endif  /* _EXT_MEMFILE_H */