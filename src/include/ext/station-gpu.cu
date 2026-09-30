#include <crtdefscu.h>
#include <stringcu.h>
#include <ext/station.h>
#include <ext/mutex.h>

__BEGIN_DECLS;

__device__ volatile unsigned int _stationMapId;
__constant__ const stationMap *_stationDeviceMap[STATION_DEVICEMAPS];
__device__ void stationDeviceSend(void *msg, int msgLength) {
	const stationMap *map = _stationDeviceMap[_stationMapId++ % STATION_DEVICEMAPS];
	if (!map)
		panic("station: device map not defined. did you start station?\n");
	long id = atomicAdd((int *)&map->setId, 1);
	stationCommand *cmd = (stationCommand *)&map->cmds[id % STATION_MSGCOUNT];
	if (cmd->magic != STATION_MAGIC)
		panic("bad station magic");
	volatile long *control = (volatile long *)&cmd->control;
	mutexSpinLock(nullptr, control, STATIONCONTROL_NORMAL, STATIONCONTROL_DEVICE);
}

__END_DECLS;
