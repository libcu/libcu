#include <sentinel.h>
#include <sentinel-hostmsg.h>
#if __OS_WIN
#include <windows.h>
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif
#include <stdio.h>

#if HAS_HOSTSENTINEL

//////////////////////
// MUTEX
#pragma region MUTEX
#if 0
#include <ext/mutex.h>
#else
#if __OS_WIN
#define SLEEP(MS) Sleep(MS)
#elif __OS_UNIX
#define SLEEP(MS) sleep(MS)
#endif

/* Mutex with exponential back-off. */
static void mutexSpinLock(void **cancelToken, volatile long *mutex, long cmp = 0, long val = 1) {
	long v; float ms = 32;
#if __OS_WIN
	while ((!cancelToken || *cancelToken) && (v = _InterlockedCompareExchange((volatile long *)mutex, cmp, val)) != cmp) {
#elif __OS_UNIX
	while ((!cancelToken || *cancelToken) && (v = __sync_val_compare_and_swap((long *)mutex, cmp, val)) != cmp) {
#endif
		SLEEP((int)ms);
		if (ms < 256) ms *= 2;
	}
}

/* Mutex set. */
static void mutexSet(volatile long *mutex, long val = 0) {
#if __OS_WIN
	_InterlockedExchange((volatile long *)mutex, val);
#elif __OS_UNIX
	__sync_lock_test_and_set((long *)mutex, val);
#endif
}

#endif
#pragma endregion

#if __OS_WIN
#define AtomicAdd(mutex, val) InterlockedAdd((volatile long *)mutex, val) - val
#elif __OS_UNIX
#define AtomicAdd(mutex, val) __sync_fetch_and_add((volatile long *)mutex, val) - val
#endif

static void executeTrans(char id, sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans);

static char *preparePtrs(sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut, sentinelCommand *cmd, char *data, char *dataEnd, intptr_t offset, sentinelOutPtr *&listOut_, char *&trans) {
	sentinelInPtr *i; sentinelOutPtr *o; char **field; char *ptr = data, *next;
	// PREPARE & TRANSFER
	int transSize = 0;
	sentinelInPtr *listIn = nullptr;
	if (ptrsIn)
		for (i = ptrsIn, field = (char **)i->field; field; i++, field = (char **)i->field) {
			if (!*field) continue;
			int size = i->size != -1 ? i->size : (i->size = (int)strlen(*field) + 1);
			next = ptr + size;
			if (!size) *field = nullptr;
			else if (next <= dataEnd) { i->unknown = ptr; ptr = next; }
			else { i->unknown = listIn; listIn = i; transSize += size; }
		}
	sentinelOutPtr *listOut = nullptr;
	if (ptrsOut) {
		if (ptrsOut[0].field != (char *)-1) ptr = data;
		else ptrsOut++; // { -1 } = append
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			int size = o->size != -1 ? o->size : (o->size = (int)(dataEnd - ptr));
			next = ptr + size;
			if (!size) *field = nullptr;
			else if (next <= dataEnd) { *field = ptr + offset; o->unknown = (void *)-1; ptr = next; }
			else { o->unknown = listOut; listOut = o; transSize += size; continue; }
		}
	}
	listOut_ = listOut;

	// TRANSFER & PACK
	if (transSize)
		executeTrans(0, cmd, transSize, listIn, listOut, offset, trans); // size & transfer-in
	if (ptrsIn)
		for (i = ptrsIn, field = (char **)i->field; field; i++, field = (char **)i->field) {
			if (!*field || !(ptr = (char *)i->unknown)) continue;
			memcpy(ptr, *field, i->size);
			*field = ptr + offset;
		}
	return data;
}

static bool postfixPtrs(sentinelOutPtr *ptrsOut, sentinelCommand *cmd, intptr_t offset, sentinelOutPtr *listOut, char *&trans) {
	sentinelOutPtr *o; char **field, **buf;
	// UNPACK & TRANSFER
	if (ptrsOut) {
		if (ptrsOut[0].field == (char *)-1) ptrsOut++; // { -1 } = append
		for (o = ptrsOut, field = (char **)o->field; field; o++, field = (char **)o->field) {
			if (o->unknown != (void *)-1) continue;
			if (!*field || !(buf = (char **)o->buf)) continue;
			int size = !o->sizeField ? o->size : *(int *)o->sizeField;
			if (size > 0) memcpy(*buf, *field - offset, size);
		}
	}
	if (listOut)
		executeTrans(1, cmd, 0, nullptr, listOut, offset, trans);
	return true;
}

static sentinelMap *_sentinelClientMap = nullptr;
static intptr_t _sentinelClientMapOffset = 0;
void sentinelClientSend(sentinelMessage *msg, int msgLength, sentinelInPtr *ptrsIn, sentinelOutPtr *ptrsOut) {
#ifndef _WIN64
	printf("Sentinel client currently only works in x64.\n"); abort();
#else
	sentinelMap *map = _sentinelClientMap;
	if (!map)
		panic("sentinel: client map not defined. did you start sentinel?\n");

	// ATTACH
	long id = AtomicAdd(&map->setId, 1);
	sentinelCommand *cmd = (sentinelCommand *)&map->cmds[id % SENTINEL_MSGCOUNT];
	if (cmd->magic != SENTINEL_MAGIC)
		panic("host: bad sentinel magic");
	AtomicAdd(&cmd->locks, 1);
	volatile long *control = &cmd->control; intptr_t offset = _sentinelClientMapOffset; char *trans = nullptr;
	mutexSpinLock(nullptr, control, SENTINELCONTROL_NORMAL, SENTINELCONTROL_DEVICE);
	if (cmd->locks != 1)
		panic("host: bad sentinel lock");

	// PREPARE
	char *data = cmd->data + ROUND8_(msgLength), *dataEnd = data + msg->size;
	sentinelOutPtr *listOut = nullptr;
	if (((ptrsIn || ptrsOut) && !(data = preparePtrs(ptrsIn, ptrsOut, cmd, data, dataEnd, offset, listOut, trans))) ||
		(msg->prepare && !msg->prepare(msg, data, dataEnd, offset)))
		panic("host: msg too long");
	if (listOut)
		msg->flow |= SENTINELFLOW_TRAN;
	cmd->length = msgLength; memcpy(cmd->data, msg, msgLength);
	//printf("msg: %d[%d]'", msg->op, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");
	mutexSet(control, SENTINELCONTROL_DEVICERDY);

	// FLOW-WAIT
	if (msg->flow & SENTINELFLOW_WAIT) {
		mutexSpinLock(nullptr, control, SENTINELCONTROL_HOSTRDY, SENTINELCONTROL_DEVICEWAIT);
		cmd->length = msgLength; memcpy(msg, cmd->data, msgLength);
		if ((ptrsOut && !postfixPtrs(ptrsOut, cmd, offset, listOut, trans)) ||
			(msg->postfix && !msg->postfix(msg, offset)))
			panic("host: postfix error");
		mutexSet(control, !listOut ? SENTINELCONTROL_NORMAL : SENTINELCONTROL_DEVICERDY);
	}
	AtomicAdd(&cmd->locks, -1);
#endif
}

static void executeTrans(char id, sentinelCommand *cmd, int size, sentinelInPtr *listIn, sentinelOutPtr *listOut, intptr_t offset, char *&trans) {
	volatile long *control = &cmd->control;
	sentinelInPtr *i; sentinelOutPtr *o; char **field; char *data = cmd->data, *ptr = trans;
	switch (id) {
	case 0:
		*(int *)data = size;
		mutexSet(control, SENTINELCONTROL_TRANSIZE);
		mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRAN);
		ptr = trans = *(char **)data;
		if (listIn)
			for (i = listIn; i; i = (sentinelInPtr *)i->unknown) {
				field = (char **)i->field;
				const char *v = (const char *)*field; int remain = i->size, length = 0;
				while (remain > 0) {
					length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
					memcpy(data, (void *)v, length); remain -= length; v += length;
					mutexSet(control, SENTINELCONTROL_TRANIN);
					mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRAN);
				}
				*field = ptr; ptr += i->size;
				i->unknown = nullptr;
			}
		if (listOut) {
			for (o = listOut; o; o = (sentinelOutPtr *)o->unknown) {
				field = (char **)o->field;
				*field = ptr; ptr += o->size;
			}
		}
		break;
	case 1:
		if (listOut)
			for (o = listOut; o; o = (sentinelOutPtr *)o->unknown) {
				field = (char **)o->buf;
				const char *v = (const char *)*field; int remain = o->size, length = 0;
				while (remain > 0) {
					length = cmd->length = remain > SENTINEL_MSGSIZE ? SENTINEL_MSGSIZE : remain;
					mutexSet(control, SENTINELCONTROL_TRANOUT);
					mutexSpinLock(nullptr, control, SENTINELCONTROL_TRANRDY, SENTINELCONTROL_TRAN);
					memcpy((void *)v, data, length); remain -= length; v += length;
				}
				o->unknown = nullptr;
			}
		break;
	}
}

#if __OS_WIN
static HANDLE _clientMapHandle = NULL;
static int *_clientMap = nullptr;
#elif __OS_UNIX
static void *_clientMap = nullptr;
#endif

void sentinelClientInitialize(char *mapHostName) {
#if __OS_WIN
	_clientMapHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mapHostName);
	if (!_clientMapHandle) {
		printf("(%d) Could not connect to Sentinel host. Please ensure host application is running.\n", GetLastError()); exit(1);
	}
	_clientMap = (int *)MapViewOfFile(_clientMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
	if (!_clientMap) {
		printf("(%d) Could not map view of file.\n", GetLastError());
		CloseHandle(_clientMapHandle); exit(1);
	}
	_sentinelClientMap = (sentinelMap *)ROUNDN_(_clientMap, MEMORY_ALIGNMENT);
	_sentinelClientMapOffset = (intptr_t)((char *)_sentinelClientMap->offset - (char *)_sentinelClientMap);
#elif __OS_UNIX
	struct stat sb;
	int fd = open(mapHostName, O_RDONLY);
	if (fd == -1) { perror("open"); exit(1); }
	if (fstat(fd, &sb) == -1) { perror("fstat"); exit(1); }
	if (!S_ISREG(sb.st_mode)) { fprintf(stderr, "%s is not a file\n", mapHostName); exit(1); }
	_clientMap = mmap(NULL, sizeof(sentinelMap) + MEMORY_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0);
	if (!_clientMap) {
		printf("Could not connect to Sentinel host. Please ensure host application is running.\n"); exit(1);
	}
	if (close(fd) == -1) { perror("close"); exit(1); }
	_sentinelClientMap = (sentinelMap *)ROUNDN_(_clientMap, MEMORY_ALIGNMENT);
	_sentinelClientMapOffset = 0;
#endif
}

void sentinelClientShutdown() {
#if __OS_WIN
	if (_clientMap) { UnmapViewOfFile(_clientMap); _clientMap = nullptr; }
	if (_clientMapHandle) { CloseHandle(_clientMapHandle); _clientMapHandle = NULL; }
#elif __OS_UNIX
	if (_clientMap) { munmap(_clientMap, sizeof(sentinelMap) + MEMORY_ALIGNMENT); _clientMap = nullptr; }
#endif
}

static __forceinline__ int getprocessid_() { host_getprocessid msg; return msg.rc; }

static char *sentinelClientRedirPipelineArgs[] = { (char *)"^0" };
void sentinelClientRedir(pipelineRedir *redir) {
#if __OS_WIN
	HANDLE process = OpenProcess(PROCESS_DUP_HANDLE, FALSE, getprocessid_());
	pipelineCreate(1, sentinelClientRedirPipelineArgs, nullptr, &redir[1].input, &redir[1].output, &redir[1].error, process, redir);
#elif __OS_UNIX
#endif
}

#endif