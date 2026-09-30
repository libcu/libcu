#pragma region License
/*
* jimsh - An interactive shell for Jim
*
* Copyright 2005 Salvatore Sanfilippo <antirez@invece.org>
* Copyright 2009 Steve Bennett <steveb@workware.net.au>
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above
*    copyright notice, this list of conditions and the following
*    disclaimer in the documentation and/or other materials
*    provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE JIM TCL PROJECT ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* JIM TCL PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
* ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation
* are those of the authors and should not be interpreted as representing
* official policies, either expressed or implied, of the Jim Tcl Project.
*/
#pragma endregion

#include <cuda_runtimecu.h>
#include <sentinel.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <stringcu.h>
#include <jim.h>
#include <jimautoconf.h>

#pragma region Name

__device__ static void JimSetArgv(Jim_Interp *interp, int argc, char *const argv[]) {
	Jim_Obj *listObj = Jim_NewListObj(interp, NULL, 0);
	for (int n = 0; n < argc; n++)
		Jim_ListAppendElement(interp, listObj, Jim_NewStringObj(interp, argv[n], -1));
	Jim_SetVariableStr(interp, "argv", listObj);
	Jim_SetVariableStr(interp, "argc", Jim_NewIntObj(interp, argc));
}

__device__ static void JimPrintErrorMessage(Jim_Interp *interp) {
	Jim_MakeErrorMessage(interp);
	printf("%s\n", Jim_String(Jim_GetResult(interp)));
}

// From initjimsh.tcl
extern __device__ int Jim_initjimshInit(Jim_Interp *interp);

#pragma endregion

// SAMPLE COMMAND
#if 0
__device__ int SampleCommand(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const argv[]) {
	if (argc != 2) {
		Jim_WrongNumArgs(interp, 1, argv, "msg");
		return JIM_ERROR;
	}
	printf("%s\n", Jim_String(argv[1]));
	return JIM_OK;
}
#endif

#pragma region Startup + Shutdown

struct PrimaryData {
	Jim_Interp *interp;
	int retcode;
};
struct PrimaryData h_dataP;

// MAIN-INIT
#if __CUDACC__

__device__ struct PrimaryData d_dataP;
void D_DATAP() { cudaErrorCheck(cudaMemcpyToSymbol(d_dataP, &h_dataP, sizeof(h_dataP))); }
void H_DATAP() { cudaErrorCheck(cudaMemcpyFromSymbol(&h_dataP, d_dataP, sizeof(h_dataP))); }

__global__ void g_MainInit(int argc, char *const argv[]);
static int MainInit(int argc, char *const argv[]) {
	memset(&h_dataP, 0, sizeof(h_dataP));
	//cudaErrorCheck(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax));
	cudaErrorCheck(cudaSetDevice(gpuGetMaxGflopsDevice()));
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 5));
	sentinelServerInitialize();
	//
	char **d_argv = cudaDeviceTransferStringArray(argc, argv);
	D_DATAP(); g_MainInit<<<1,1>>>(argc, d_argv); cudaErrorCheck(cudaDeviceSynchronize()); H_DATAP();
	cudaFree(d_argv);
	return h_dataP.retcode;
}

#define _dataP d_dataP
__global__ void g_MainInit(int argc, char *const argv[]) {
#else
#define _dataP h_dataP
static int MainInit(int argc, char *const argv[]) {
	memset(&h_dataP, 0, sizeof(h_dataP));
#endif
	// Create and initialize the interpreter
	Jim_Interp *interp = _dataP.interp = Jim_CreateInterp();
	Jim_RegisterCoreCommands(interp);
	// SAMPLE COMMAND
#if 0
	Jim_CreateCommand(interp, "sample", SampleCommand, nullptr, nullptr);
#endif

	// Register static extensions
	if (Jim_InitStaticExtensions(interp) != JIM_OK)
		JimPrintErrorMessage(interp);
	//
	Jim_SetVariableStrWithStr(interp, "jim::argv0", argv[0]);
	Jim_SetVariableStrWithStr(interp, JIM_INTERACTIVE, argc == 1 ? "1" : "0");
	int retcode = Jim_initjimshInit(interp);
	if (argc == 1) {
		if (retcode == JIM_ERROR)
			JimPrintErrorMessage(interp);
		if (retcode != JIM_EXIT)
			JimSetArgv(interp, 0, NULL);
	}
	else {
		if (argc > 2 && !strcmp(argv[1], "-e")) {
			JimSetArgv(interp, argc - 3, argv + 3);
			retcode = Jim_Eval(interp, argv[2]);
			if (retcode != JIM_ERROR)
				printf("%s\n", Jim_String(Jim_GetResult(interp)));
		}
		else {
			Jim_SetVariableStr(interp, "argv0", Jim_NewStringObj(interp, argv[1], -1));
			JimSetArgv(interp, argc - 2, argv + 2);
			retcode = Jim_EvalFile(interp, argv[1]);
		}
		if (retcode == JIM_ERROR)
			JimPrintErrorMessage(interp);
	}
#if __CUDACC__
	_dataP.retcode = retcode;
#else
	return retcode;
#endif
}

// MAIN-SHUTDOWN
#if __CUDACC__
__global__ void g_MainShutdown(int retcode);
static int MainShutdown(int retcode) {
	D_DATAP(); g_MainShutdown<<<1,1>>>(retcode); cudaErrorCheck(cudaDeviceSynchronize()); H_DATAP();
	cudaDeviceReset();
	sentinelServerShutdown();
	return h_dataP.retcode;
}

__global__ void g_MainShutdown(int retcode) {
#else
static int MainShutdown(int retcode) {
#endif
	Jim_Interp *interp = _dataP.interp;
	if (retcode == JIM_EXIT)
		retcode = Jim_GetExitCode(interp);
	Jim_FreeInterp(interp);
#if __CUDACC__
	_dataP.retcode = retcode;
#else
	return retcode;
#endif
}

#pragma endregion

int main(int argc, char *const argv[]) {
	if (argc > 1 && !strcmp(argv[1], "--version")) {
		printf("%d.%d\n", JIM_VERSION / 100, JIM_VERSION % 100);
		return 0;
	}
	int retcode = MainInit(argc, argv);
	if (argc == 1 && retcode != JIM_EXIT)
		retcode = Jim_InteractivePrompt(h_dataP.interp);
	retcode = MainShutdown(retcode);
	if (retcode == JIM_ERROR)
		retcode = 1;
	else
		retcode = 0;
	return retcode;
}
