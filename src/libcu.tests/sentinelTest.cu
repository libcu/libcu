#include <stdiocu.h>
#include <stringcu.h>
#include <sentinel.h>
#include <assert.h>

enum {
	MODULE_SIMPLE = 500,
	MODULE_STRING,
};

struct module_simple {
	sentinelMessage Base;
	int Value;
	__device__ module_simple(bool wait, int value) : Base(wait, MODULE_SIMPLE), Value(value) { sentinelDeviceSend(&Base, sizeof(module_simple)); }
	int RC;
};

struct module_string {
	static __forceinline__ __device__ char *Prepare(module_string *t, char *data, char *dataEnd, intptr_t offset) {
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ module_string(bool wait, const char *str) : Base(wait, MODULE_STRING, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(module_string)); }
	int RC;
};

bool sentinelModuleExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*, char*, char*, intptr_t)) {
	switch (data->OP) {
	case MODULE_SIMPLE: { module_simple *msg = (module_simple *)data; msg->RC = msg->Value; return true; }
	case MODULE_STRING: { module_string *msg = (module_string *)data; msg->RC = (int)strlen(msg->Str); return true; }
	}
	return false;
}
static sentinelExecutor _moduleExecutor = { nullptr, "module", sentinelModuleExecutor, nullptr };

static __global__ void g_sentinel_test1() {
	printf("sentinel_test1\n");

	//// SENTINELDEVICESEND ////
	//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
	module_simple a0(true, 1);
	int a0a = a0.RC;
	assert(a0a == 1);
	module_string a1(true, "test");
	int a1a = a1.RC;
	assert(a1a == 4);
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
