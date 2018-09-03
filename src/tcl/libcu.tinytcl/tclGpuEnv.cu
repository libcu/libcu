#include "tclInt.h"
#include "tclGpu.h"

// The structure below is used to keep track of all of the interpereters for which we're managing the "env" array.  It's needed so that they
// can all be updated whenever an environment variable is changed anywhere.
typedef struct EnvInterp {
	Tcl_Interp *interp;			// Interpreter for which we're managing the env array.
	struct EnvInterp *nextPtr;	// Next in list of all such interpreters, or zero.
} EnvInterp;

static __device__ EnvInterp *_firstInterpPtr;
// First in list of all managed interpreters, or NULL if none.

// Declarations for local procedures defined in this file:
static __device__ char *EnvTraceProc(ClientData clientData, Tcl_Interp *interp, char *name1, char *name2, int flags);

/*
*----------------------------------------------------------------------
*
* TclSetupEnv --
*	This procedure is invoked for an interpreter to make environment variables accessible from that interpreter via the "env" associative array.
*
* Results:
*	None.
*
* Side effects:
*	The interpreter is added to a list of interpreters managed by us, so that its view of envariables can be kept consistent
*	with the view in other interpreters.  If this is the first call to Tcl_SetupEnv, then additional initialization happens,
*	such as copying the environment to dynamically-allocated space for ease of management.
*
*----------------------------------------------------------------------
*/
__device__ void TclSetupEnv(Tcl_Interp *interp)
{
	// Next, add the interpreter to the list of those that we manage.
	EnvInterp *eiPtr = (EnvInterp *)_allocFast(sizeof(EnvInterp));
	eiPtr->interp = interp;
	eiPtr->nextPtr = _firstInterpPtr;
	_firstInterpPtr = eiPtr;

	// Store the environment variable values into the interpreter's "env" array, and arrange for us to be notified on future writes and unsets to that array.
	Tcl_UnsetVar2(interp, "env", (char *)NULL, TCLGLOBAL__ONLY);
	for (int i = 0; ; i++) {
		char *p = __environ[i];
		if (!p || !*p ) {
			break;
		}
		char *p2;
		for (p2 = p; *p2 != '='; p2++) { }
		*p2 = 0;
		Tcl_SetVar2(interp, "env", p, p2+1, TCLGLOBAL__ONLY);
		*p2 = '=';
	}
	Tcl_TraceVar2(interp, "env", (char *)NULL, TCLGLOBAL__ONLY | TCL_TRACE_WRITES | TCL_TRACE_UNSETS, EnvTraceProc, (ClientData)NULL);
}

/*
*----------------------------------------------------------------------
*
* EnvTraceProc --
*	This procedure is invoked whenever an environment variable is modified or deleted.  It propagates the change to the
*	"environ" array and to any other interpreters for whom we're managing an "env" array.
*
* Results:
*	Always returns NULL to indicate success.
*
* Side effects:
*	Environment variable changes get propagated.  If the whole "env" array is deleted, then we stop managing things for
*	this interpreter (usually this happens because the whole interpreter is being deleted).
*
*----------------------------------------------------------------------
*/
static __device__ char *EnvTraceProc(ClientData clientData, Tcl_Interp *interp, char *name1, char *name2, int flags)
{
	// First see if the whole "env" variable is being deleted.  If so, just forget about this interpreter.
	if (name2 == NULL) {
		if ((flags & (TCL_TRACE_UNSETS|TCL_TRACE_DESTROYED)) != (TCL_TRACE_UNSETS|TCL_TRACE_DESTROYED)) {
			panic("EnvTraceProc called with confusing arguments");
		}
		register EnvInterp *eiPtr = _firstInterpPtr;
		if (eiPtr->interp == interp) {
			_firstInterpPtr = eiPtr->nextPtr;
		} else {
			register EnvInterp *prevPtr;
			for (prevPtr = eiPtr, eiPtr = eiPtr->nextPtr; ; prevPtr = eiPtr, eiPtr = eiPtr->nextPtr) {
				if (eiPtr == NULL) {
					panic("EnvTraceProc couldn't find interpreter");
				}
				if (eiPtr->interp == interp) {
					prevPtr->nextPtr = eiPtr->nextPtr;
					break;
				}
			}
		}
		_freeFast((char *)eiPtr);
		return NULL;
	}
	// If a value is being set, call setenv to do all of the work.
	if (flags & TCL_TRACE_WRITES) {
		setenv(name2, Tcl_GetVar2(interp, "env", name2, TCLGLOBAL__ONLY), true);
	}
	if (flags & TCL_TRACE_UNSETS) {
		unsetenv(name2);
	}
	return NULL;
}
