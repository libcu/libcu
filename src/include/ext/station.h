/*
station.h - memory transfer framework for device to host functions
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef _STATION_H
#define _STATION_H
#include <crtdefscu.h>
#include <stdio.h>
#if _MSC_VER
#include <fcntl.h>
#include <io.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif

#define STATION_MAGIC (unsigned short)0xC812
#define STATION_DEVICEMAPS 1
#define STATION_MSGSIZE 4096
#define STATION_MSGCOUNT 1

	typedef struct __align__(8) {
		unsigned short magic;
		volatile long control;
#ifndef _WIN64
		int unknown;
#endif
		int length;
		char data[STATION_MSGSIZE];
		void dump();
	} stationCommand;

	typedef struct __align__(8) {
		long getId;
		volatile long setId;
		intptr_t offset;
		stationCommand cmds[STATION_MSGCOUNT];
		void dump();
	} stationMap;

	typedef struct stationContext {
		stationMap *deviceMap[STATION_DEVICEMAPS];
	} stationContext;

	extern __constant__ const stationMap *_stationDeviceMap[STATION_DEVICEMAPS];

	extern void stationHostInitialize();
	extern void stationHostShutdown();
	extern __device__ void stationDeviceSend(void *msg, int msgLength);

#define STATIONCONTROL_NORMAL 0x0
#define STATIONCONTROL_DEVICE 0x1
#define STATIONCONTROL_HOST 0x5
#define STATIONCONTROL_HOSTRDY 0x6

#ifdef __cplusplus
}
#endif

#endif  /* STATION_H */