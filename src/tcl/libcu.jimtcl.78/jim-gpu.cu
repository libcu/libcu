#include "jim.h"
#include "jim-eventloop.h"

__host_device__ int Jim_gpuInit(Jim_Interp *interp) {
	if (Jim_PackageProvide(interp, "gpu", "1.0", JIM_ERRMSG))
		return JIM_ERR;
	return JIM_OK;
}

__host_device__ int Jim_InitStaticExtensions(Jim_Interp *interp) {
	extern __host_device__ int Jim_bootstrapInit(Jim_Interp *interp);
	extern __host_device__ int Jim_globInit(Jim_Interp *interp);
	extern __host_device__ int Jim_stdlibInit(Jim_Interp *interp);
	extern __host_device__ int Jim_tclcompatInit(Jim_Interp *interp);
	//
	extern __host_device__ int Jim_aioInit(Jim_Interp *interp);
	extern __host_device__ int Jim_arrayInit(Jim_Interp *interp);
	extern __host_device__ int Jim_clockInit(Jim_Interp *interp);
	extern __host_device__ int Jim_execInit(Jim_Interp *interp);
	extern __host_device__ int Jim_fileInit(Jim_Interp *interp);
	extern __host_device__ int Jim_readdirInit(Jim_Interp *interp);
	extern __host_device__ int Jim_regexpInit(Jim_Interp *interp);
	//
#if __CUDACC__
	extern __host_device__ int Jim_gpuInit(Jim_Interp *interp);
#else
	extern __host_device__ int Jim_win32Init(Jim_Interp *interp);
#endif
	extern __host_device__ int Jim_historyInit(Jim_Interp *interp);
	extern __host_device__ int Jim_loadInit(Jim_Interp *interp);
	extern __host_device__ int Jim_namespaceInit(Jim_Interp *interp);
	extern __host_device__ int Jim_packInit(Jim_Interp *interp);
	extern __host_device__ int Jim_packageInit(Jim_Interp *interp);
	//extern __host_device__ int Jim_tclprefixInit(Jim_Interp *interp);

	Jim_bootstrapInit(interp);
	Jim_globInit(interp);
	Jim_stdlibInit(interp);
	//Jim_tclcompatInit(interp);
	//
	Jim_aioInit(interp);
	Jim_arrayInit(interp);
	Jim_clockInit(interp);
	Jim_eventloopInit(interp);
	Jim_execInit(interp);
	Jim_fileInit(interp);
	Jim_readdirInit(interp);
	Jim_regexpInit(interp);
	//
#if __CUDACC__
	Jim_gpuInit(interp);
#else
	//Jim_win32Init(interp);
#endif
#ifndef __CUDACC__
	Jim_historyInit(interp);
	Jim_loadInit(interp);
	Jim_namespaceInit(interp);
	Jim_packInit(interp);
	Jim_packageInit(interp);
	//Jim_tclprefixInit(interp);
#endif
	return JIM_OK;
}
