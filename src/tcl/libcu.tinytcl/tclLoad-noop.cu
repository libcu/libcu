// tclLoadDl.c --
//
//	This procedure provides a version of the TclLoadFile that works with the "dlopen" and "dlsym" library procedures for dynamic loading.
//
// Copyright (c) 1995-1997 Sun Microsystems, Inc.
// This file was integrated from tcl to tinytcl by Snapgear.
//
// See the file "license.terms" for information on usage and redistribution
// of this file, and for a DISCLAIMER OF ALL WARRANTIES.

#include "tclLoad.h"

/*
*---------------------------------------------------------------------------
*
* TclpDlopen --
*	Dynamically loads a binary code file into memory and returns a handle to the new code.
*
* Results:
*	A standard Tcl completion code.  If an error occurs, an error message is left in the interp's result.
*
* Side effects:
*	New code suddenly appears in memory.
*
*---------------------------------------------------------------------------
*/
__device__ int TclpDlopen(Tcl_Interp *interp, char *pathPtr, Tcl_LoadHandle *loadHandle, Tcl_FSUnloadFileProc **unloadProcPtr) {
	*unloadProcPtr = NULL;
	*loadHandle = NULL;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* TclpFindSymbol --
*	Looks up a symbol, by name, through a handle associated with a previously loaded piece of code (shared library).
*
* Results:
*	Returns a pointer to the function associated with 'symbol' if it is found.  Otherwise returns NULL and may leave an error message in the interp's result.
*
*----------------------------------------------------------------------
*/
__device__ Tcl_PackageInitProc *TclpFindSymbol(Tcl_Interp *interp, Tcl_LoadHandle loadHandle, const char *symbol) {
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* TclpUnloadFile --
*	Unloads a dynamically loaded binary code file from memory. Code pointers in the formerly loaded file are no longer valid after calling this function.
*
* Results:
*	None.
*
* Side effects:
*	Code removed from memory.
*
*----------------------------------------------------------------------
*/
__device__ void TclpUnloadFile(Tcl_LoadHandle loadHandle) {
}