// tclCkalloc.c -- Interface to malloc and free that provides support for debugging problems involving overwritten, double freeing memory and loss of memory.
//
// Copyright 1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

#ifdef TCL_MEM_DEBUG
#define GUARD_SIZE 8

struct mem_header {
	long length;
	char *file;
	int line;
	struct mem_header *flink;
	struct mem_header *blink;
	unsigned char low_guard[GUARD_SIZE];
	char body[1];
};

static __device__ struct mem_header *_allocHead = NULL;  // List of allocated structures

#define GUARD_VALUE 0341

/* static char high_guard[] = {0x89, 0xab, 0xcd, 0xef}; */

static __device__ int _total_mallocs = 0;
static __device__ int _total_frees = 0;
static __device__ long _current_bytes_malloced = 0;
static __device__ long _maximum_bytes_malloced = 0;
static __device__ int _current_malloc_packets = 0;
static __device__ int _maximum_malloc_packets = 0;
static __device__ int _break_on_malloc = 0;
static __device__ int _trace_on_at_malloc = 0;
static __device__ bool _alloc_tracing = false;
static __device__ bool _init_malloced_bodies = false;
#ifdef MEM_VALIDATE
__device__ bool _validate_memory = true;
#else
__device__ bool _validate_memory = false;
#endif

/*
*----------------------------------------------------------------------
*
* dump_memory_info --
*     Display the global memory management statistics.
*
*----------------------------------------------------------------------
*/
static __device__ void dump_memory_info(FILE *out_) 
{
	fprintf_(out_, "total mallocs             %10d\n", _total_mallocs);
	fprintf_(out_, "total frees               %10d\n", _total_frees);
	fprintf_(out_, "current packets allocated %10d\n", _current_malloc_packets);
	fprintf_(out_, "current bytes allocated   %10ld\n", _current_bytes_malloced);
	fprintf_(out_, "maximum packets allocated %10d\n", _maximum_malloc_packets);
	fprintf_(out_, "maximum bytes allocated   %10ld\n", _maximum_bytes_malloced);
}

/*
*----------------------------------------------------------------------
*
* ValidateMemory --
*     Procedure to validate allocted memory guard zones.
*
*----------------------------------------------------------------------
*/
static __device__ void ValidateMemory(struct mem_header *memHeaderP, char *file, int line, bool nukeGuards)
{
	int idx;
	bool guard_failed = false;
	int byte;
	for (idx = 0; idx < GUARD_SIZE; idx++) {
		byte = *(memHeaderP->low_guard + idx);
		if (byte != GUARD_VALUE) {
			guard_failed = true;
			fflush(stdout);
			byte &= 0xff;
			fprintf_(stderr, "low guard byte %d is 0x%x  \t%c\n", idx, byte, (isprint(byte) ? byte : ' '));
		}
	}
	if (guard_failed) {
		dump_memory_info(stderr);
		fprintf_(stderr, "low guard failed at %lx, %s %d\n", memHeaderP->body, file, line);
		fflush(stderr); // In case name pointer is bad.
		fprintf_(stderr, "%ld bytes allocated at (%s %d)\n", memHeaderP->length, memHeaderP->file, memHeaderP->line);
		panic("Memory validation failure");
	}
	unsigned char *hiPtr = (unsigned char *)memHeaderP->body + memHeaderP->length;
	for (idx = 0; idx < GUARD_SIZE; idx++) {
		byte = *(hiPtr + idx);
		if (byte != GUARD_VALUE) {
			guard_failed = true;
			fflush(stdout);
			byte &= 0xff;
			fprintf_(stderr, "hi guard byte %d is 0x%x  \t%c\n", idx, byte, (isprint(byte) ? byte : ' '));
		}
	}
	if (guard_failed) {
		dump_memory_info(stderr);
		fprintf_(stderr, "high guard failed at %lx, %s %d\n", memHeaderP->body, file, line);
		fflush(stderr); // In case name pointer is bad.
		fprintf_(stderr, "%ld bytes allocated at (%s %d)\n", memHeaderP->length, memHeaderP->file, memHeaderP->line);
		panic("Memory validation failure");
	}
	if (nukeGuards) {
		memset((char *)memHeaderP->low_guard, 0, GUARD_SIZE); 
		memset((char *)hiPtr, 0, GUARD_SIZE); 
	}
}

/*
*----------------------------------------------------------------------
*
* Tcl_ValidateAllMemory --
*     Validates guard regions for all allocated memory.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_ValidateAllMemory(char *file, int line)
{
	struct mem_header *memScanP;
	for (memScanP = _allocHead; memScanP != NULL; memScanP = memScanP->flink)
		ValidateMemory(memScanP, file, line, false);
}

/*
*----------------------------------------------------------------------
*
* Tcl_DumpActiveMemory --
*     Displays all allocated memory to stderr.
*
* Results:
*     Return TCL_ERROR if an error accessing the file occures, `errno' 
*     will have the file error number left in it.
*----------------------------------------------------------------------
*/
__device__ int Tcl_DumpActiveMemory(char *fileName)
{
	FILE *fileP = fopen(fileName, "w");
	if (fileP == NULL)
		return TCL_ERROR;
	struct mem_header *memScanP;
	for (memScanP = _allocHead; memScanP != NULL; memScanP = memScanP->flink) {
		char *address = &memScanP->body[0];
		fprintf_(fileP, "%8lx - %8lx  %7ld @ %s %d", address, address + memScanP->length - 1, memScanP->length, memScanP->file, memScanP->line);
		if (!strcmp(memScanP->file, "tclHash.cu") && memScanP->line == 515) {
			fprintf_(fileP, "\t|%s|", ((Tcl_HashEntry *)address)->key.string);
		}
		fputc('\n', fileP);
	}
	fclose(fileP);
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_MemAlloc - debugging _allocFast
*        Allocate the requested amount of space plus some extra for guard bands at both ends of the request, plus a size, panicing 
*        if there isn't enough space, then write in the guard bands and return the address of the space in the middle that the
*        user asked for.
*
*        The second and third arguments are file and line, these contain the filename and line number corresponding to the caller.
*        These are sent by the _allocFast macro; it uses the preprocessor autodefines __FILE__ and __LINE__.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_MemAlloc(unsigned int size, char *file, int line)
{
	if (_validate_memory)
		Tcl_ValidateAllMemory(file, line);

	struct mem_header *result = (struct mem_header *)malloc((unsigned)size + sizeof(struct mem_header) + GUARD_SIZE);
	if (result == NULL) {
		fflush(stdout);
		dump_memory_info(stderr);
		panic("unable to alloc %d bytes, %s line %d", size, file, line);
	}

	// Fill in guard zones and size.  Link into allocated list.
	result->length = size;
	result->file = file;
	result->line = line;
	memset((char *)result->low_guard, GUARD_VALUE, GUARD_SIZE);
	memset(result->body + size, GUARD_VALUE, GUARD_SIZE);
	result->flink = _allocHead;
	result->blink = NULL;
	if (_allocHead != NULL)
		_allocHead->blink = result;
	_allocHead = result;
	_total_mallocs++;
	if (_trace_on_at_malloc && (_total_mallocs >= _trace_on_at_malloc)) {
		fflush(stdout);
		fprintf_(stderr, "reached malloc trace enable point (%d)\n", _total_mallocs);
		fflush(stderr);
		_alloc_tracing = true;
		_trace_on_at_malloc = 0;
	}
	if (_alloc_tracing)
		fprintf_(stderr,"_allocFast %lx %d %s %d\n", result->body, size, file, line);
	if (_break_on_malloc && (_total_mallocs >= _break_on_malloc)) {
		_break_on_malloc = 0;
		fflush(stdout);
		fprintf_(stderr, "reached malloc break limit (%d)\n", _total_mallocs);
		fprintf_(stderr, "program will now enter C debugger\n");
		fflush(stderr);
		abort();
	}
	_current_malloc_packets++;
	if (_current_malloc_packets > _maximum_malloc_packets)
		_maximum_malloc_packets = _current_malloc_packets;
	_current_bytes_malloced += size;
	if (_current_bytes_malloced > _maximum_bytes_malloced)
		_maximum_bytes_malloced = _current_bytes_malloced;
	if (_init_malloced_bodies)
		memset(result->body, 0xff, (int)size);
	return result->body;
}

/*
*----------------------------------------------------------------------
*
* Tcl_MemFree - debugging _freeFast
*        Verify that the low and high guards are intact, and if so then free the buffer else panic.
*
*        The guards are erased after being checked to catch duplicate frees.
*
*        The second and third arguments are file and line, these contain the filename and line number corresponding to the caller.
*        These are sent by the _freeFast macro; it uses the preprocessor autodefines __FILE__ and __LINE__.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_MemFree(char *ptr, char *file, int line)
{
	struct mem_header *memp = 0; // Must be zero for size calc
	memp = (struct mem_header *)(((char *)ptr) - memp->body); // Since header ptr is zero, body offset will be size
	if (_alloc_tracing)
		fprintf_(stderr, "_freeFast %lx %ld %s %d\n", memp->body, memp->length, file, line);
	if (_validate_memory)
		Tcl_ValidateAllMemory(file, line);
	ValidateMemory(memp, file, line, true);
	_total_frees++;
	_current_malloc_packets--;
	_current_bytes_malloced -= memp->length;
	
	// Delink from allocated list
	if (memp->flink != NULL)
		memp->flink->blink = memp->blink;
	if (memp->blink != NULL)
		memp->blink->flink = memp->flink;
	if (_allocHead == memp)
		_allocHead = memp->flink;
	free((char *)memp);
	return 0;
}

/*
*--------------------------------------------------------------------
*
* Tcl_MemRealloc - debugging _reallocFast
*	Reallocate a chunk of memory by allocating a new one of the right size, copying the old data to the new location, and then
*	freeing the old memory space, using all the memory checking features of this package.
*
*--------------------------------------------------------------------
*/
__device__ char *Tcl_MemRealloc(char *ptr, unsigned int size, char *file, int line)
{
	char *new_ = Tcl_MemAlloc(size, file, line);
	memcpy(new_, ptr, (int)size);
	Tcl_MemFree(ptr, file, line);
	return new_;
}

/*
*----------------------------------------------------------------------
*
* MemoryCmd --
*     Implements the TCL memory command:
*       memory info
*       memory display
*       _break_on_malloc count
*       _trace_on_at_malloc count
*       trace on|off
*       validate on|off
*
* Results:
*     Standard TCL results.
*
*----------------------------------------------------------------------
*/
static __device__ int MemoryCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	char *fileName;
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args:  should be \"", args[0], " option [args..]\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (!strcmp(args[1], "trace")) {
		if (argc != 3) 
			goto bad_suboption;
		_alloc_tracing = (!strcmp(args[2], "on"));
		return TCL_OK;
	}
	if (!strcmp(args[1], "init")) {
		if (argc != 3)
			goto bad_suboption;
		_init_malloced_bodies = (!strcmp(args[2], "on"));
		return TCL_OK;
	}
	if (!strcmp(args[1], "validate")) {
		if (argc != 3)
			goto bad_suboption;
		_validate_memory = (!strcmp(args[2], "on"));
		return TCL_OK;
	}
	if (!strcmp(args[1], "_trace_on_at_malloc")) {
		if (argc != 3) 
			goto argError;
		if (Tcl_GetInt(interp, args[2], &_trace_on_at_malloc) != TCL_OK)
			return TCL_ERROR;
		return TCL_OK;
	}
	if (!strcmp(args[1], "_break_on_malloc")) {
		if (argc != 3) 
			goto argError;
		if (Tcl_GetInt(interp, args[2], &_break_on_malloc) != TCL_OK)
			return TCL_ERROR;
		return TCL_OK;
	}
	if (!strcmp(args[1],"info")) {
		dump_memory_info(stdout);
		return TCL_OK;
	}
	if (!strcmp(args[1], "active")) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args:  should be \"", args[0], " active file", (char *)NULL);
			return TCL_ERROR;
		}
		fileName = (char *)args[2];
		if (fileName[0] == '~')
			if ((fileName = Tcl_TildeSubst(interp, fileName)) == NULL)
				return TCL_ERROR;
		if (Tcl_DumpActiveMemory(fileName) != TCL_OK) {
			Tcl_AppendResult(interp, "error accessing ", args[2], (char *)NULL);
			return TCL_ERROR;
		}
		return TCL_OK;
	}
	Tcl_AppendResult(interp, "bad option \"", args[1], "\":  should be info, init, active, _break_on_malloc, ", "_trace_on_at_malloc, trace, or validate", (char *)NULL);
	return TCL_ERROR;

argError:
	Tcl_AppendResult(interp, "wrong # args:  should be \"", args[0], " ", args[1], "count\"", (char *)NULL);
	return TCL_ERROR;

bad_suboption:
	Tcl_AppendResult(interp, "wrong # args:  should be \"", args[0], " ", args[1], " on|off\"", (char *)NULL);
	return TCL_ERROR;
}

/*
*----------------------------------------------------------------------
*
* Tcl_InitMemory --
*     Initialize the memory command.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_InitMemory(Tcl_Interp *interp)
{
	Tcl_CreateCommand(interp, "memory", MemoryCmd, (ClientData)NULL, (Tcl_CmdDeleteProc *)NULL);
}

#else

///*
//*----------------------------------------------------------------------
//*
//* Tcl_Ckalloc --
//*     Interface to malloc when TCL_MEM_DEBUG is disabled.  It does check that memory was actually allocated.
//*
//*----------------------------------------------------------------------
//*/
//__device__ charVOID *Tcl_MemAlloc(unsigned int size)
//{
//	char *result = malloc(size);
//	if (result == NULL) 
//		panic("unable to alloc %d bytes", size);
//	return result;
//}
//
///*
//*----------------------------------------------------------------------
//*
//* TckCkfree --
//*     Interface to free when TCL_MEM_DEBUG is disabled.  Done here rather in the macro to keep some modules from being compiled with 
//*     TCL_MEM_DEBUG enabled and some with it disabled.
//*
//*----------------------------------------------------------------------
//*/
//__device__ void Tcl_MemFree(charVOID *ptr)
//{
//	free(ptr);
//}

/*
*----------------------------------------------------------------------
*
* Tcl_InitMemory --
*     Dummy initialization for memory command, which is only available if TCL_MEM_DEBUG is on.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_InitMemory(Tcl_Interp *interp)
{
}

#endif
