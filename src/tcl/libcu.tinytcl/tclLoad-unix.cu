// tclLoadDl.c --
//
//	This procedure provides a version of the TclLoadFile that works with the "dlopen" and "dlsym" library procedures for dynamic loading.
//
// Copyright (c) 1995-1997 Sun Microsystems, Inc.
// This file was integrated from tcl to tinytcl by Snapgear.
//
// See the file "license.terms" for information on usage and redistribution
// of this file, and for a DISCLAIMER OF ALL WARRANTIES.

#include "Tcl+Load.h"
#if OS_UNIX
#include <dlfcn.h>
#include <unistd.h>

// In some systems, like SunOS 4.1.3, the RTLD_NOW flag isn't defined and this argument to dlopen must always be 1.  The RTLD_GLOBAL
// flag is needed on some systems (e.g. SCO and UnixWare) but doesn't exist on others;  if it doesn't exist, set it to 0 so it has no effect.
#ifndef RTLD_NOW
#define RTLD_NOW 1
#endif
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0
#endif

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
int TclpDlopen(Tcl_Interp *interp, char *pathPtr, Tcl_LoadHandle *loadHandle, Tcl_FSUnloadFileProc **unloadProcPtr)
{
	// First try the full path the user gave us.  This is particularly important if the cwd is inside a vfs, and we are trying to load using a relative path.
	void *handle = dlopen(pathPtr, RTLD_NOW | RTLD_GLOBAL);
	if (!handle && *pathPtr != '/') {
		char buf[256];
		// Perhaps we couldn't find the file in the LD_LIBRARY_PATH. Look in /lib/tcl
		_snprintf(buf, sizeof(buf), "/lib/tcl/%s", pathPtr);
		handle = dlopen(buf, RTLD_NOW | RTLD_GLOBAL);
		if (!handle) {
			// Still not there, so look in the current directory
			getcwd(buf, sizeof(buf));
			if (strlen(buf) + strlen(pathPtr) + 2 < 256) {
				strcat(buf, "/");
				strcat(buf, pathPtr);
				handle = dlopen(buf, RTLD_NOW | RTLD_GLOBAL);
			}
		}
	}
	if (!handle) {
		Tcl_AppendResult(interp, "couldn't load file \"",  pathPtr, "\": ", dlerror(), (char *)NULL);
		return TCL_ERROR;
	}
	*unloadProcPtr = &TclpUnloadFile;
	*loadHandle = (Tcl_LoadHandle)handle;
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
Tcl_PackageInitProc *TclpFindSymbol(Tcl_Interp *interp, Tcl_LoadHandle loadHandle, const char *symbol)
{
	void *handle = (void *)loadHandle;
	return (Tcl_PackageInitProc *)dlsym(handle, symbol);
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
void TclpUnloadFile(Tcl_LoadHandle loadHandle)
{
	void *handle = (void *)loadHandle;
	dlclose(handle);
}

#endif