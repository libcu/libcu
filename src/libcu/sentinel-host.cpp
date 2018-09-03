#include <sentinel.h>
#if __OS_WIN
#include <windows.h>
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#endif
#include <stdio.h>

#if HAS_HOSTSENTINEL

sentinelMap *_sentinelHostMap = nullptr;
intptr_t _sentinelHostMapOffset = 0;
void sentinelClientSend(sentinelMessage *msg, int msgLength) {
#ifndef _WIN64
	printf("Sentinel currently only works in x64.\n");
	abort();
#else
	sentinelMap *map = _sentinelHostMap;
	if (!map) {
		printf("sentinel: device map not defined. did you start sentinel?\n");
		exit(0);
	}
#if __OS_WIN
	long id = InterlockedAdd((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#elif __OS_UNIX
	long id = __sync_fetch_and_add((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE;
#endif
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *control = (volatile long *)&cmd->Control;
	//while (InterlockedCompareExchange((long *)control, 1, 0) != 0) { }
	//cmd->Data = (char *)cmd + ROUND8_(sizeof(sentinelCommand));
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg->Prepare && !msg->Prepare(msg, cmd->Data, cmd->Data + ROUND8_(msgLength) + msg->Size, _sentinelHostMapOffset)) {
		printf("msg too long");
		exit(0);
	}
	memcpy(cmd->Data, msg, msgLength);
	//printf("Msg: %d[%d]'", msg->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");

	*control = 2;
	if (msg->Wait) {
#if __OS_WIN
		while (InterlockedCompareExchange((long *)control, 5, 4) != 4) { }
#elif __OS_UNIX
		while (__sync_val_compare_and_swap((long *)control, 5, 4) != 4) { }
#endif
		memcpy(msg, cmd->Data, msgLength);
		*control = 0;
	}
#endif
}

#endif