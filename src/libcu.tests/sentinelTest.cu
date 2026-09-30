#include <stdiocu.h>
#include <stringcu.h>
#include <sentinel.h>
#include <assert.h>

enum {
	MODULE_SIMPLE = 500,
	MODULE_STRING,
	MODULE_RETURN,
	MODULE_CUSTOM,
	MODULE_COMPLEX,
	MODULE_JUMBOOUT,
};

struct module_simple {
	sentinelMessage base;
	int value;
	__device__ module_simple(bool wait, int value) : base(MODULE_SIMPLE, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE), value(value) { sentinelDeviceSend(&base, sizeof(module_simple)); }
	int rc;
};

struct module_string {
	sentinelMessage base;
	const char *str;
	__device__ module_string(bool wait, const char *str) : base(MODULE_STRING, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), str(str) { sentinelDeviceSend(&base, sizeof(module_string), ptrsIn); }
	int rc;
	sentinelInPtr ptrsIn[2] = {
		{ &str, -1 },
		{ nullptr }
	};
};

struct module_return {
	sentinelMessage base;
	const char *buf; size_t size;
	__device__ module_return(const char *buf, size_t size) : base(MODULE_RETURN, SENTINELFLOW_WAIT, SENTINEL_CHUNK), buf(buf), size(size) { ptrsOut[0].size = size; sentinelDeviceSend(&base, sizeof(module_return), nullptr, ptrsOut); }
	size_t rc;
	void *ptr;
	sentinelOutPtr ptrsOut[2] = {
		{ &ptr, &buf, 0 },
		{ nullptr }
	};
};

struct module_custom {
	static __forceinline__ __device__ char *prepare(module_custom *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = t->str ? (int)strlen(t->str) + 1 : 0;
		char *str = data;
		char *end = data += strLength;
		if (end > dataEnd) return nullptr;
		memcpy(str, t->str, strLength);
		t->str = str + offset;
		t->ptr = str + offset;
		return end;
	}
	static __forceinline__ __device__ bool postfix(module_custom *t, intptr_t offset) {
		char *ptr = (char *)t->ptr - offset;
		//if (t->rc > 0) memcpy(t->buf, ptr, t->rc);
		return true;
	}
	sentinelMessage base;
	const char *str; char *buf;
	__device__ module_custom(bool wait, const char *str) : base(MODULE_CUSTOM, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK, SENTINELPREPARE(prepare), SENTINELPOSTFIX(postfix)), str(str), buf(buf) { sentinelDeviceSend(&base, sizeof(module_custom)); }
	int rc;
	void *ptr;
};

struct module_complex {
	sentinelMessage base;
	const char *str;
	__device__ module_complex(bool wait, const char *str, char *ptr, size_t size) : base(MODULE_COMPLEX, wait ? SENTINELFLOW_WAIT : SENTINELFLOW_NONE, SENTINEL_CHUNK), str(str), ptr(ptr) { ptrsIn[1].size = size; sentinelDeviceSend(&base, sizeof(module_complex), ptrsIn); }
	int rc; int rc2;
	void *ptr;
	sentinelInPtr ptrsIn[3] = {
		{ &str, -1 },
		{ &ptr, 0 },
		{ nullptr }
	};
};

bool sentinelModuleExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	switch (data->op) {
	case MODULE_SIMPLE: { module_simple *msg = (module_simple *)data; msg->rc = msg->value; return true; }
	case MODULE_STRING: { module_string *msg = (module_string *)data; msg->rc = (int)strlen(msg->str); return true; }
	case MODULE_RETURN: { module_return *msg = (module_return *)data; msg->rc = 5; strcpy((char *)msg->ptr, "test"); return true; }
	case MODULE_CUSTOM: { module_custom *msg = (module_custom *)data; msg->rc = (int)strlen(msg->str); return true; }
	case MODULE_COMPLEX: { module_complex *msg = (module_complex *)data; msg->rc = (int)strlen(msg->str); msg->rc2 = (int)strlen((char *)msg->ptr); return true; }
	}
	return false;
}
static sentinelExecutor _moduleExecutor = { nullptr, "module", sentinelModuleExecutor, nullptr };

static __global__ void g_sentinel_test1() {
	//printf("sentinel_test1\n");

	//// SENTINELDEVICESEND ////
	//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
	for (int k = 0; k < 0; k++) {
		printf("%d ", k);
		char buf[100];
		module_simple a0(true, 1); int a0a = a0.rc; assert(a0a == 1);
		module_string a1(true, "test"); int a1a = a1.rc; assert(a1a == 4);
		module_return a2(buf, sizeof(buf)); int a2a = a2.rc; assert(a2a == 5 && !strcmp(buf, "test"));
		module_custom a3(true, "test"); int a3a = a3.rc; assert(a3a == 4);
		char complex[2048]; memset(complex, 1, sizeof(complex));
		module_complex a4(true, "test", complex, sizeof(complex)); int a4a = a4.rc; assert(a4a == 4);
	}

	// JUMBO
#define JUMBOSIZE (4096+4096+1024)
	for (int k = 0; k < 50; k++) {
		printf("J%d ", k);
		char jumbo[JUMBOSIZE]; memset(jumbo, 2, sizeof(jumbo)); jumbo[JUMBOSIZE - 1] = 0;
		//module_complex b0(true, "test", jumbo, sizeof(jumbo)); int b0a = b0.rc; int b0b = b0.rc2; assert(b0a == 4 && b0b == JUMBOSIZE - 1);
		module_return b1(jumbo, sizeof(jumbo)); int b1a = b1.rc; assert(b1a == 5 && !strcmp(jumbo, "test"));
	}
}

cudaError_t sentinel_test1() {
	sentinelRegisterExecutor(&_moduleExecutor);
	g_sentinel_test1<<<1, 1>>>(); return cudaDeviceSynchronize();
}

//// SENTINELDEFAULTEXECUTOR ////
//	extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t));

//// SENTINELSERVERINITIALIZE, SENTINELSERVERSHUTDOWN ////
//	extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
//	extern void sentinelServerShutdown();

//// SENTINELDEVICESEND ////
//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);

//// SENTINELCLIENTINITIALIZE, SENTINELCLIENTSHUTDOWN ////
//	extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
//	extern void sentinelClientShutdown();

//// SENTINELCLIENTSEND ////
//	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);

//// SENTINELFINDEXECUTOR, SENTINELREGISTEREXECUTOR, SENTINELUNREGISTEREXECUTOR ////
//	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
//	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
//	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

//// SENTINELREGISTERFILEUTILS ////
//	extern void sentinelRegisterFileUtils();
