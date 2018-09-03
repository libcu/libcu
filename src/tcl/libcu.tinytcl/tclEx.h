// tclExtend.h
//
//    External declarations for the extended Tcl library.
//-----------------------------------------------------------------------------
// Copyright 1992 Karl Lehenbauer and Mark Diekhans.
//
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided
// that the above copyright notice appear in all copies.  Karl Lehenbauer and Mark Diekhans make no representations about the suitability of this
// software for any purpose.  It is provided "as is" without express or implied warranty.

#ifndef __TCLEX_H__
#define __TCLEX_H__

#include <stdiocu.h>
#ifndef __TCL_H__
#include "tcl.h"
#endif

// Version suffix for extended Tcl, this is appended to the standard Tcl version to form the actual extended Tcl version.
#define TCL_EXTD_VERSION_SUFFIX "c"  // 6.1c, 6.2c or 6.3c
typedef void *void_pt;

// Flags for Tcl shell startup.
#define TCLSH_QUICK_STARTUP       1   // Don't process default & init files.
#define TCLSH_ABORT_STARTUP_ERR   2   // Abort on an error.
#define TCLSH_NO_INIT_FILE        4   // Don't process the init file.
#define TCLSH_NO_STACK_DUMP       8   // Don't dump the proc stack on error.

// These globals are used by the infox command.  Should be set by main.
extern __device__ char *tclxVersion;        // Extended Tcl version number.
extern __device__ int tclxPatchlevel;		// Extended Tcl patch level.

extern __device__ char *tclAppName;         // Application name
extern __device__ char *tclAppLongname;     // Long, natural language application name
extern __device__ char *tclAppVersion;      // Version number of the application

// If set to be a pointer to the procedure Tcl_RecordAndEval, will link in history.  Should be set by main.
extern int (*tclShellCmdEvalProc)();

// If non-zero, a signal was received.  Normally signals are handled in Tcl_Eval, but if an application does not return to eval for some period
// of time, then this should be checked and Tcl_CheckForSignal called if this is set.
extern int tclReceivedSignal;

// Exported Extended Tcl functions.
extern __device__ int Tcl_CheckForSignal(Tcl_Interp *interp, int cmdResultCode);
extern __device__ void Tcl_CommandLoop(Tcl_Interp *interp, FILE *inFile, FILE *outFile, int (*evalProc)(), unsigned options);
extern __device__ Tcl_Interp *Tcl_CreateExtendedInterp();
extern __device__ char *Tcl_DeleteKeyedListField(Tcl_Interp *interp, const char *fieldName, const char *keyedList);
extern __device__ char *Tcl_DownShift(char *targetStr, const char *sourceStr);
extern __device__ void Tcl_ErrorAbort(Tcl_Interp *interp, int noStackDump, int exitCode);
extern __device__ char *Tcl_UpShift(char *targetStr, const char *sourceStr);
extern __device__ int Tcl_GetKeyedListField(Tcl_Interp *interp, const char *fieldName, const char *keyedList, char **fieldValuePtr);
__device__ int Tcl_GetKeyedListKeys(Tcl_Interp *interp, const char *subFieldName, const char *keyedList, int *keyesArgcPtr, char ***keyesArgsPtr);
extern __device__ int Tcl_GetLong(Tcl_Interp *interp, const char *string, long *longPtr);
extern __device__ int Tcl_GetUnsigned(Tcl_Interp *interp, const char *string, unsigned *unsignedPtr);
extern __device__ char *Tcl_SetKeyedListField(Tcl_Interp *interp, const char *fieldName, const char *fieldvalue, const char *keyedList);
extern __device__ int Tcl_StrToLong(const char *string, int base, long *longPtr);
extern __device__ int Tcl_StrToInt(const char *string, int base, int *intPtr);
extern __device__ int Tcl_StrToUnsigned(const char *string, int base, unsigned *unsignedPtr);
extern __device__ int Tcl_StrToDouble(const char *string, double *doublePtr);
extern __device__ void_pt Tcl_HandleAlloc(void_pt headerPtr, char *handlePtr);
extern __device__ void Tcl_HandleFree(void_pt headerPtr, void_pt entryPtr);
extern __device__ void_pt Tcl_HandleTblInit(const char *handleBase, int entrySize, int initEntries);
extern __device__ void Tcl_HandleTblRelease(void_pt headerPtr);
extern __device__ int Tcl_HandleTblUseCount(void_pt headerPtr, int amount);
extern __device__ void_pt Tcl_HandleWalk(void_pt headerPtr, int *walkKeyPtr);
extern __device__ void Tcl_WalkKeyToHandle(void_pt headerPtr, int walkKey, char *handlePtr);
extern __device__ void_pt Tcl_HandleXlate(Tcl_Interp *interp, void_pt headerPtr, const char *handle);
extern __device__ int Tcl_MathError(char *functionName, int errorType);
extern __device__ void Tcl_Startup(Tcl_Interp *interp, int argc, const char **args, const char *defaultFile, unsigned options);
extern __device__ int Tcl_ShellEnvInit(Tcl_Interp *interp, unsigned options, const char *programName, int argc, const char **args, int interactive, const char *defaultFile);
extern __device__ int Tcl_System(Tcl_Interp *interp, char *command);

#endif /* __TCLEX_H__ */