## #include <sentinel.h>

## Host Side
Prototype | Description | Tags
--- | --- | :---:
```bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t));``` | xxxx
```void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);``` | xxxx
```void sentinelServerShutdown();``` | xxxx
```void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);``` | xxxx
```void sentinelClientShutdown();``` | xxxx
```void sentinelClientSend(sentinelMessage *msg, int msgLength);``` | xxxx
```sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);``` | xxxx
```void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);``` | xxxx
```void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);``` | xxxx

## Host Side, File Utils
Prototype | Description | Tags
--- | --- | :---:
```void sentinelRegisterFileUtils();``` | xxxx

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);``` | xxxx
