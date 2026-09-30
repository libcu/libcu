#include <sentinel.h>
#if __OS_WIN
#include <windows.h>
#include <process.h>
#include <io.h>
#define THREADHANDLE HANDLE
#define THREADCALL unsigned int __stdcall
#define AtomicAdd(mutex, val) InterlockedAdd((volatile long *)mutex, val) - val
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <pthread.h>
#define THREADHANDLE pthread_t
#define THREADCALL void *
#define AtomicAdd(mutex, val) __sync_fetch_and_add((volatile long *)mutex, val) - val
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtimecu.h>
#include <ext/mutex.h>

void sentinelCommand::dump() {
	register unsigned char *b = (unsigned char *)&data;
	register int l = length;
	printf("cmd: %d[%d]'", ((sentinelMessage*)data)->op, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

void sentinelMap::dump() {
	register unsigned char *b = (unsigned char *)this;
	register int l = sizeof(sentinelMap);
	printf("map: 0x%p[%d]'", b, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

static sentinelContext _ctx;
static sentinelExecutor _baseHostExecutor = { nullptr, "base", sentinelDefaultHostExecutor, nullptr };
static sentinelExecutor _baseDeviceExecutor = { nullptr, "base", sentinelDefaultDeviceExecutor, nullptr };

static bool executeTrans(void **tag);

// HOSTSENTINEL
#if HAS_HOSTSENTINEL

static THREADHANDLE _threadHostHandle = 0;
static THREADCALL sentinelHostThread(void *data) {
	void *funcTag[3]{ (void *)-1, nullptr, nullptr };
	sentinelMap *map = _ctx.hostMap;
	while (map) {
		if (!_threadHostHandle) break;
		long id = AtomicAdd(&map->getId, 1);
		sentinelCommand *cmd = &map->cmds[id % SENTINEL_MSGCOUNT];
		if (cmd->magic != SENTINEL_MAGIC) {
			printf("host: bad sentinel magic."); exit(1);
		}
		funcTag[1] = cmd;
		volatile long *control = &cmd->control;
		mutexSpinLock((void **)&_threadHostHandle, control,
			SENTINELCONTROL_DEVICERDY, SENTINELCONTROL_HOST, MUTEXPRED_GTE, SENTINELCONTROL_TRAN, executeTrans, funcTag);
		if (!_threadHostHandle) break;
		sentinelMessage *msg = (sentinelMessage *)cmd->data;
		//map->dump();
		//cmd->dump();

		// EXECUTE
		char *(*hostPrepare)(void*, char*, char*, intptr_t) = nullptr;
		for (sentinelExecutor *exec = _ctx.hostList; exec && exec->executor && !exec->executor(exec->tag, msg, cmd->length, &hostPrepare); exec = exec->next) {}

		// FLOW-WAIT
		if (msg->flow & SENTINELFLOW_WAIT) {
			mutexSet(control, SENTINELCONTROL_HOSTRDY);
			if (msg->flow & SENTINELFLOW_TRAN) {
				mutexSpinLock((void **)&_threadHostHandle, control,
					SENTINELCONTROL_DEVICERDY, SENTINELCONTROL_HOSTWAIT, MUTEXPRED_GTE, SENTINELCONTROL_TRAN, executeTrans, funcTag);
				mutexSet(control, SENTINELCONTROL_NORMAL);
			}
		}
		else mutexSet(control, SENTINELCONTROL_NORMAL);
	}
	if (funcTag[2]) free(funcTag[2]);
	return 0;
}

#endif

// DEVICESENTINEL
#if HAS_DEVICESENTINEL

static THREADHANDLE _threadDeviceHandle[SENTINEL_DEVICEMAPS];
static THREADCALL sentinelDeviceThread(void *data) {
	int threadId = (int)(intptr_t)data;
	void *funcTag[3]{ reinterpret_cast<void *>(threadId), nullptr, nullptr };
	sentinelMap *map = _ctx.deviceMap[threadId];
	while (map) {
		if (!_threadDeviceHandle[threadId]) break;
		long id = AtomicAdd(&map->getId, 1);
		sentinelCommand *cmd = &map->cmds[id % SENTINEL_MSGCOUNT];
		if (cmd->magic != SENTINEL_MAGIC) {
			printf("bad sentinel magic."); exit(1);
		}
		funcTag[1] = cmd;
		volatile long *control = &cmd->control;
		mutexSpinLock((void **)&_threadDeviceHandle[threadId], control,
			SENTINELCONTROL_DEVICERDY, SENTINELCONTROL_HOST, MUTEXPRED_GTE, SENTINELCONTROL_TRAN, executeTrans, funcTag);
		if (!_threadDeviceHandle[threadId]) break;
		sentinelMessage *msg = (sentinelMessage *)&cmd->data;
		//map->dump();
		//cmd->dump();

		// EXECUTE
		char *(*hostPrepare)(void*, char*, char*, intptr_t) = nullptr;
		for (sentinelExecutor *exec = _ctx.deviceList; exec && exec->executor && !exec->executor(exec->tag, msg, cmd->length, &hostPrepare); exec = exec->next) {}

		// host-prepare
		char *data = cmd->data + cmd->length, *dataEnd = data + msg->size;
		if (hostPrepare && !hostPrepare(msg, data, dataEnd, map->offset)) {
			printf("msg too long."); exit(0);
		}

		// FLOW-WAIT
		if (msg->flow & SENTINELFLOW_WAIT) {
			mutexSet(control, SENTINELCONTROL_HOSTRDY);
			if (msg->flow & SENTINELFLOW_TRAN) {
				mutexSpinLock((void **)&_threadDeviceHandle[threadId], control,
					SENTINELCONTROL_DEVICERDY, SENTINELCONTROL_HOSTWAIT, MUTEXPRED_GTE, SENTINELCONTROL_TRAN, executeTrans, funcTag);
				mutexSet(control, SENTINELCONTROL_NORMAL);
			}
		}
		else mutexSet(control, SENTINELCONTROL_NORMAL);
	}
	if (funcTag[2]) free(funcTag[2]);
	return 0;
}
#endif

// EXECUTETRANS
#if HAS_HOSTSENTINEL && HAS_DEVICESENTINEL
#define CANCELTOKEN threadId != -1 ? (void **)&_threadDeviceHandle[threadId] : (void **)&_threadHostHandle
#elif HAS_HOSTSENTINEL
#define CANCELTOKEN threadId != -1 ? nullptr : (void **)&_threadHostHandle
#elif HAS_DEVICESENTINEL
#define CANCELTOKEN threadId != -1 ? (void **)&_threadDeviceHandle[threadId] : nullptr
#else
#define CANCELTOKEN nullptr
#endif
static bool executeTrans(void **tag) {
	int threadId = static_cast<int>((intptr_t)tag[0]);
	sentinelCommand *cmd = (sentinelCommand *)tag[1];
	volatile long *control = &cmd->control;
	char *data = cmd->data, *ptr = (char *)tag[2];
	while (*control >= SENTINELCONTROL_TRAN) {
		int length = cmd->length;
		switch (*control) {
		case SENTINELCONTROL_TRANSIZE:
			ptr = (char *)tag[2];
			if (ptr) free(ptr);
			if (!(ptr = (char *)malloc(length))) {
				printf("sentinel transfer out of memory"); exit(0);
			}
			tag[2] = *(char **)data = ptr;
			break;
		case SENTINELCONTROL_TRANIN:
			memcpy(ptr, data, length); ptr += length; break;
		case SENTINELCONTROL_TRANOUT:
			memcpy(data, ptr, length); ptr += length; break;
		default:
#if __OS_WIN
			Sleep(1); continue;
#elif __OS_UNIX
			sleep(1); continue;
#endif
		}
		mutexSet(control, SENTINELCONTROL_TRANRDY);
		mutexSpinLock(CANCELTOKEN, control, SENTINELCONTROL_TRAN, SENTINELCONTROL_TRAN, MUTEXPRED_LT, SENTINELCONTROL_TRAN);
	}
	return true;
}
#undef CANCELTOKEN

#if HAS_HOSTSENTINEL
static sentinelMap *_sentinelHostMap = nullptr;
static intptr_t _sentinelHostMapOffset = 0;
#if __OS_WIN
static HANDLE _hostMapHandle = NULL;
static int *_hostMap = nullptr;
#elif __OS_UNIX
static void *_hostMap = nullptr;
#endif
#endif
#if HAS_DEVICESENTINEL
static bool _sentinelDevice = false;
static int *_deviceMap[SENTINEL_DEVICEMAPS];
#endif
void sentinelServerInitialize(sentinelExecutor *deviceExecutor, char *mapHostName, bool hostSentinel, bool deviceSentinel) {
#if HAS_HOSTSENTINEL
	if (hostSentinel) {
		// create host map
#if __OS_WIN
		_hostMapHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT, mapHostName);
		if (!_hostMapHandle) {
			printf("Could not create file mapping object (%d).\n", GetLastError()); exit(1);
		}
		_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
		if (!_hostMap) {
			printf("Could not map view of file (%d).\n", GetLastError()); CloseHandle(_hostMapHandle); exit(1);
		}
		_sentinelHostMap = _ctx.hostMap = (sentinelMap *)ROUNDN_(_hostMap, MEMORY_ALIGNMENT);
		_sentinelHostMap->offset = (intptr_t)_sentinelHostMap;
#elif __OS_UNIX
		struct stat sb;
		int fd = open(mapHostName, O_RDONLY | O_CREAT);
		if (fd == -1) { perror("open"); exit(1); }
		if (unlink(mapHostName) == -1) { perror("unlink"); exit(1); }
		if (fstat(fd, &sb) == -1) { perror("fstat"); exit(1); }
		if (!S_ISREG(sb.st_mode)) { fprintf(stderr, "%s is not a file\n", mapHostName); exit(1); }
		_hostMap = mmap(NULL, sizeof(sentinelMap) + MEMORY_ALIGNMENT, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, fd, 0);
		if (!_hostMap) {
			printf("Could not map view of file.\n"); exit(1);
		}
		if (close(fd) == -1) { perror("close"); exit(1); }
		_sentinelHostMap = _ctx.hostMap = (sentinelMap *)ROUNDN_(_hostMap, MEMORY_ALIGNMENT);
		_sentinelHostMap->offset = 0;
#endif

		// initialize commands
		for (int j = 0; j < SENTINEL_MSGCOUNT; j++) {
			sentinelCommand *cmd = &_ctx.hostMap->cmds[j];
			cmd->magic = SENTINEL_MAGIC;
			cmd->control = 0;
			cmd->locks = 0;
		}

		// register executor
		sentinelRegisterExecutor(&_baseHostExecutor, true, false);

		// launch threads
#if __OS_WIN
		_threadHostHandle = (HANDLE)_beginthreadex(0, 0, sentinelHostThread, nullptr, 0, 0);
#elif __OS_UNIX
		int err; if ((err = pthread_create(&_threadHostHandle, NULL, &sentinelHostThread, NULL))) {
			printf("Could not create host thread (%s).\n", strerror(err)); exit(1);
		}
#endif
	}
#endif

#if HAS_DEVICESENTINEL
	if (deviceSentinel) {
		// create device maps
		_sentinelDevice = true;
		sentinelMap *d_deviceMap[SENTINEL_DEVICEMAPS];
		for (int i = 0; i < SENTINEL_DEVICEMAPS; i++) {
			cudaErrorCheckF(cudaHostAlloc((void **)&_deviceMap[i], sizeof(sentinelMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
			d_deviceMap[i] = _ctx.deviceMap[i] = (sentinelMap *)_deviceMap[i];
			cudaErrorCheckF(cudaHostGetDevicePointer((void **)&d_deviceMap[i], _ctx.deviceMap[i], 0), goto initialize_error);
#ifndef _WIN64
			_ctx.deviceMap[i]->offset = (intptr_t)((char *)_deviceMap[i] - (char *)d_deviceMap[i]);
			//printf("chk: %x %x [%x]\n", (char *)_deviceMap[i], (char *)d_deviceMap[i], _ctx.deviceMap[i]->Offset);
#else
			_ctx.deviceMap[i]->offset = 0;
#endif
			// initialize commands
			for (int j = 0; j < SENTINEL_MSGCOUNT; j++) {
				sentinelCommand *cmd = &_ctx.deviceMap[i]->cmds[j];
				cmd->magic = SENTINEL_MAGIC;
				cmd->control = 0;
				cmd->locks = 0;
			}
		}
		cudaErrorCheckF(cudaMemcpyToSymbol(_sentinelDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);

		// register executor
		sentinelRegisterExecutor(&_baseDeviceExecutor, true, true);
		if (deviceExecutor)
			sentinelRegisterExecutor(deviceExecutor, true, true);

		// launch threads
		//memset(_threadDeviceHandle, 0, sizeof(_threadDeviceHandle));
#if __OS_WIN
		for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
			_threadDeviceHandle[i] = (HANDLE)_beginthreadex(0, 0, sentinelDeviceThread, (void *)(intptr_t)i, 0, 0);
#elif __OS_UNIX
		int err; for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
			if ((err = pthread_create(&_threadDeviceHandle[i], NULL, &sentinelDeviceThread, (void *)(intptr_t)i))) {
				printf("Could not create device thread (%s).\n", strerror(err)); exit(1);
			}
#endif
	}
#endif
	return;
initialize_error:
	perror("sentinelServerInitialize:Error");
	sentinelServerShutdown();
	exit(1);
}

void sentinelServerShutdown() {
	// close host map
#if HAS_HOSTSENTINEL
#if __OS_WIN
	if (_hostMapHandle) {
		if (_threadHostHandle) { CloseHandle(_threadHostHandle); _threadHostHandle = NULL; }
		if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
		CloseHandle(_hostMapHandle); _hostMapHandle = NULL;
	}
#elif __OS_UNIX
	if (_hostMap) {
		if (_threadHostHandle) { pthread_cancel(_threadHostHandle); _threadHostHandle = 0; }
		munmap(_hostMap, sizeof(sentinelMap) + MEMORY_ALIGNMENT); _hostMap = nullptr;
	}
#endif
#endif
	// close device maps
#if HAS_DEVICESENTINEL
	if (_sentinelDevice) {
		for (int i = 0; i < SENTINEL_DEVICEMAPS; i++) {
#if __OS_WIN
			if (_threadDeviceHandle[i]) { CloseHandle(_threadDeviceHandle[i]); _threadDeviceHandle[i] = NULL; }
#elif __OS_UNIX
			if (_threadDeviceHandle[i]) { pthread_cancel(_threadDeviceHandle[i]); _threadDeviceHandle[i] = 0; }
#endif
			if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
		}
	}
#endif
}

sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice) {
	sentinelExecutor *list = forDevice ? _ctx.deviceList : _ctx.hostList;
	sentinelExecutor *exec = nullptr;
	for (exec = list; exec && name && strcmp(name, exec->name); exec = exec->next) {}
	return exec;
}

static void sentinelUnlinkExecutor(sentinelExecutor *exec, bool forDevice) {
	sentinelExecutor *list = forDevice ? _ctx.deviceList : _ctx.hostList;
	if (!exec) {}
	else if (list == exec)
		if (forDevice) _ctx.deviceList = exec->next;
		else _ctx.hostList = exec->next;
	else if (list) {
		sentinelExecutor *p = list;
		while (p->next && p->next != exec)
			p = p->next;
		if (p->next == exec)
			p->next = exec->next;
	}
}

void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault, bool forDevice) {
	sentinelUnlinkExecutor(exec, forDevice);
	sentinelExecutor *list = forDevice ? _ctx.deviceList : _ctx.hostList;
	if (makeDefault || !list) {
		exec->next = list;
		if (forDevice) _ctx.deviceList = exec;
		else _ctx.hostList = exec;
	}
	else {
		exec->next = list->next;
		list->next = exec;
	}
}

void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice) {
	sentinelUnlinkExecutor(exec, forDevice);
}