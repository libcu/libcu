#include "tclInt.h"
#include "tclGpu.h"

/*
*----------------------------------------------------------------------
*
* Tcl_ErrnoId --
*	Return a textual identifier for the current errno value.
*
* Results:
*	This procedure returns a machine-readable textual identifier that corresponds to the current errno value (e.g. "EPERM").
*	The identifier is the same as the #define name in errno.h.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_ErrnoId()
{
	switch (errno) {
#ifdef ERANGE
	case ERANGE: return "ERANGE";
#endif
	}
	return "unknown error";
}

/*
*----------------------------------------------------------------------
*
* Tcl_SignalId --
*	Return a textual identifier for a signal number.
*
* Results:
*	This procedure returns a machine-readable textual identifier that corresponds to sig.  The identifier is the same as the
*	#define name in signal.h.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_SignalId(int sig)
{
	return "unknown signal";
}

/*
*----------------------------------------------------------------------
*
* Tcl_SignalMsg --
*	Return a human-readable message describing a signal.
*
* Results:
*	This procedure returns a string describing sig that should make sense to a human.  It may not be easy for a machine to parse.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_SignalMsg(int sig)
{
	return "unknown signal";
}