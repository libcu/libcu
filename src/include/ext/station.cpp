#if __OS_WIN
#include <windows.h>
#elif __OS_UNIX
#include <stdlib.h>
#include <string.h>
#endif
#include <stdio.h>
#include <assert.h>
#include <cuda_runtimecu.h>
#include <ext/station.h>

void stationCommand::dump() {
	register unsigned char *b = (unsigned char *)&data;
	register int l = length;
	printf("Cmd: %d[%d]'", 0, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

static stationContext _ctx;

static bool _stationDevice = false;
static int *_deviceMap[STATION_DEVICEMAPS];
void stationHostInitialize() {
	// create device maps
	_stationDevice = true;
	stationMap *d_deviceMap[STATION_DEVICEMAPS];
	for (int i = 0; i < STATION_DEVICEMAPS; i++) {
		cudaErrorCheckF(cudaHostAlloc((void **)&_deviceMap[i], sizeof(stationMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
		d_deviceMap[i] = _ctx.deviceMap[i] = (stationMap *)_deviceMap[i];
		cudaErrorCheckF(cudaHostGetDevicePointer((void **)&d_deviceMap[i], _ctx.deviceMap[i], 0), goto initialize_error);
#ifndef _WIN64
		_ctx.deviceMap[i]->offset = (intptr_t)((char *)_deviceMap[i] - (char *)d_deviceMap[i]);
		//printf("chk: %x %x [%x]\n", (char *)_deviceMap[i], (char *)d_deviceMap[i], _ctx.deviceMap[i]->Offset);
#else
		_ctx.deviceMap[i]->offset = 0;
#endif
	}
	cudaErrorCheckF(cudaMemcpyToSymbol(_stationDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);
	return;
initialize_error:
	perror("stationHostInitialize:Error");
	stationHostInitialize();
	exit(1);
}

void stationHostShutdown() {
	// close device maps
	for (int i = 0; i < STATION_DEVICEMAPS; i++) {
		if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
	}
}
