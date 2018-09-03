// sky:added
#include "tclInt.h"

__device__ void Tcl_WrongNumArgs(Tcl_Interp *interp, int objc, char *objv[], const char *message)
{
}

__device__ Tcl_Obj *Tcl_NewObj(char *value, int length, char *typeName)
{
	return nullptr;
}

//__device__ int Tcl_ObjAppendElement(Tcl_Interp *interp, Tcl_Obj *obj, Tcl_Obj *appendObj)
//{
//}

__device__ void Tcl_IncrRefCount(Tcl_Obj *obj)
{
}

__device__ void Tcl_DecrRefCount(Tcl_Obj *obj)
{
}

// EXTRA
__device__ void Tcl_BackgroundError(Tcl_Interp *interp)
{
}
