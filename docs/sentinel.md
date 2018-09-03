# sentinel

describe sentinel


## Interface

These are the methods to access Sentinel's functionality:
* `sentinelDefaultExecutor` - the built-in default executor auto-registered as base on `sentinelServerInitialize`
* `sentinelServerInitialize` - initializes the server side Sentinel creating its assets and registering the `sentinelDefaultExecutor` and the `executor` if provided.
* `sentinelServerShutdown` - shutsdown the server side Sentinel and its assets
* `sentinelDeviceSend` - used in the message constructor to send message(s) on the device bus
* `sentinelClientInitialize` - initializes the client side Sentinel establishing a connection to the server
* `sentinelClientShutdown` - shutsdown the client side Sentinel
* `sentinelClientSend` - used in the message constructor to send message(s) on the host bus
* `sentinelFindExecutor` - finds the `sentinelExecutor` with the given `name` on the host or device
* `sentinelRegisterExecutor` - registers the `sentinelExecutor` on the host or device
* `sentinelUnregisterExecutor` - un-registers the `sentinelExecutor` on the host or device
* `sentinelRegisterFileUtils` - registers the `sentinelRegisterFileUtils` for use with file-system utilities
```
extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t));
extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
extern void sentinelServerShutdown();
#if HAS_DEVICESENTINEL
	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
#endif
#if HAS_HOSTSENTINEL
	extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
	extern void sentinelClientShutdown();
	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);
#endif
extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

// file-utils
extern void sentinelRegisterFileUtils();
```


## Structure
These defines are used for Sentinel:
* `SENTINEL_NAME` - default name to use for IPC when calling `sentinelServerInitialize`
* `SENTINEL_MAGIC` - a magic value used to ensure message alignment
* `SENTINEL_DEVICEMAPS` - the number of device to host maps, and threads to create
* `SENTINEL_MSGSIZE` - the size of a message structure including its header information
* `SENTINEL_MSGCOUNT` - the number of message structures available in a given map
```
#define SENTINEL_NAME "Sentinel"
#define SENTINEL_MAGIC (unsigned short)0xC811
#define SENTINEL_DEVICEMAPS 1
#define SENTINEL_MSGSIZE 4096
#define SENTINEL_MSGCOUNT 1
```

### SentinelContext
SentinelContext is a singleton which represents the state of Sentinel. Sentinel provides two distinct message buses for device to host, and host to host communication respectivly. The later is used for IPC using named pipes, named `SENTINEL_NAME`, and is extensivly used by the file-system utilities.

`SentinelServerInitialize` execution:
* Creates `SENTINEL_DEVICEMAPS` instances of `sentinelMap`, each with it's own processing thread, and stores them in DeviceMap.
* Sets a single linked list of `sentinelExecutor(s)` for the processesing of all device to host messages in `DeviceList`.
* Creates a single instance of `sentinelMap`, with it's own processing thread, and stores it in HostMap.
* Sets a single linked list of `sentinelExecutor(s)` for the processesing of all host to host messages in `HostList`.
```
sentinelContext
- DeviceMap[SENTINEL_DEVICEMAPS] - sentinelMap(s) used for device
- HostMap - sentinelMap used for host IPC
- HostList - linked list of sentinelExecutor(s) for host processing
- DeviceList - linked list of sentinelExecutor(s) for device processing
```

### SentinelMap
Each `sentinelMap` has a dedicated processing thread and can hold `SENTINEL_MSGCOUNT` messages of size `SENTINEL_MSGSIZE`, this size must include the `sentinelCommand` size.
* `GetId` is a rolling index into the next message to read
* New messages are written to `SetId`, which is marked volatile to by-pass any caching issues
* `Offset(s)` are applied as appropreate to align mapped memory between host and device coordinates
* Data contains all `sentinelCommand(s)` with embedded `sentinelMessage(s)` with a queue depth of `SENTINEL_MSGCOUNT`. `SENTINEL_MSGSIZE` must include the `sentinelCommand` size 
```
sentinelMap
- GetId - current reading location
- SetId:volatile - current writing location, atomicaly incremeted by SENTINEL_MSGSIZE 
- Offset - used for map alignment
- Data[SENTINEL_MSGSIZE*SENTINEL_MSGCOUNT]
```

### SentinelCommand
Each `sentinelCommand` represents a command being passed across the bus, and has an embeded `sentinelMessage` in it's `Data` property
* `Magic` is used to ensure message alignment
* `Control` handles flow control, and is marked volatile to by-pass any caching issues
	* 0 - normal state
	* 1 - device in-progress
	* 2 - client signal that data is ready to process
	* 3 - host in-progress
	* 4 - host signal that results are ready to read
* `Length` and `Data` represent the embeded `sentinelMessage`
```
sentinelCommand
- Magic - magic
- Control:volatile - control flag
- Length - length of data
- Data[...] - data
```

### SentinelMessage
Each `sentinelMessage` is a custom message being passed across the bus
```
sentinelMessage
- Wait - flag to asyc or wait
- OP - operation
- Size - size of message
- Prepare() - method to prepare message for transport
```

### SentinelExecutor
The `sentinelExecutor` is responsible for executing message on host.
```
sentinelExecutor
- Next - linked list pointer
- Name - name of executor
- Executor() - attempts to process messages
- Tag - optional data for executor
```


## Example

The following is an example of creating a custom message, and using it.

### Enum
* Use an `enum` to auto-number the operations in a module
* The developer is responsible for name collisions
* Device and Host messages have seperate namespaces
* Numbers below `500` are reserved for system use
```
enum {
	MODULE_SIMPLE = 500,
	MODULE_STRING,
};
```

### Message
A simple message with a integer value named `Value`, and a integer return code named `RC`.
* `Base` must be first
* `Base` constructor parameters of `size` and `prepare` can be ignored
```
struct module_simple {
	sentinelMessage Base;
	int Value;
	__device__ module_simple(int value)
		: Base(true, MODULE_SIMPLE), Value(value) { sentinelDeviceSend(&Base, sizeof(module_simple)); }
	int RC;
};
```

### Message
Message asset(s) referenced outside of the message payload, like string values, must be coalesced into the message payload. refered values offset(s) must be adjusted to align memory maps.
* `Base` must be first
* `Base` constructor parameters of `size` and `prepare` are required
* `size` should contain enough space to hold the message with it's embeded values, and must be remain under the `SENTINEL_MSGSIZE` plus the `sentinelCommand` overhead size.
* `prepare` must embed referenced values, replacing the original pointers with the emeded one and apply the offset to align memory maps.
```
struct module_string {
	static __forceinline __device__ char *Prepare(module_string *t, char *data, char *dataEnd, intptr_t offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ module_string(const char *str)
		: Base(true, MODULE_STRING, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(module_string)); }
	int RC;
};
```

### Executor
```
bool sentinelExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t))
{
	switch (data->OP) {
	case MODULE_SIMPLE: { module_simple *msg = (module_simple *)data; msg->RC = msg->Value; return true; }
	case MODULE_STRING: { module_string *msg = (module_string *)data; msg->RC = strlen(msg->Str); return true; }
	}
	return false;
}
```

### Calling
to call:
```
module_simple msg(123);
int rc = msg.RC;
```

to call:
```
module_string msg("123");
int rc = msg.RC;
```