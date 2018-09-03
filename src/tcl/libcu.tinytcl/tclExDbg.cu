// tclXdebug.c --
//
// Tcl command execution trace command.
//-----------------------------------------------------------------------------
// Copyright 1992 Karl Lehenbauer and Mark Diekhans.
//
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided
// that the above copyright notice appear in all copies.  Karl Lehenbauer and Mark Diekhans make no representations about the suitability of this
// software for any purpose.  It is provided "as is" without express or implied warranty.

#include "tclExInt.h"
//#include <stdio.h>
//#include <sys/time.h>

// Client data structure for the cmdtrace command.
#define ARG_TRUNCATE_SIZE 40
#define CMD_TRUNCATE_SIZE 60

typedef struct traceInfo_t {
	Tcl_Interp *interp;
	Tcl_Trace   traceHolder;
	int         noEval;
	int         noTruncate;
	int         procCalls;
	int         flush;
	int         depth;
	FILE       *filePtr; // File to output trace to.
} traceInfo_t, *traceInfo_pt;

// Prototypes of internal functions.
static __device__ void PrintStr(FILE *filePtr, char *string, int numChars);
static __device__ void PrintArg(FILE *filePtr, char *argStr, int noTruncate);
static __device__ void TraceCode(traceInfo_pt traceInfoPtr, int level, char *command, int argc, const char *args[]);
static __device__ void CmdTraceRoutine(ClientData clientData, Tcl_Interp *interp, int level, char *command, Tcl_CmdProc *cmdProc, ClientData cmdClientData, int argc, const char *args[]);
static __device__ void CleanUpDebug(ClientData clientData);

/*
*-----------------------------------------------------------------------------
*
* PrintStr --
*     Print an string, truncating it to the specified number of characters. If the string contains newlines, \n is substituted.
*
*-----------------------------------------------------------------------------
*/
static __device__ void PrintStr(FILE *filePtr, char *string, int numChars)
{
	for (int idx = 0; idx < numChars; idx++) {
		if (string[idx] == '\n') {
			fputc('\\', filePtr);
			fputc('n', filePtr);
		} else
			fputc(string[idx], filePtr);
	}
	if (numChars < strlen(string))
		fprintf_(filePtr, "...");
}

/*
*-----------------------------------------------------------------------------
*
* PrintArg --
*     Print an argument string, truncating and adding "..." if its longer then ARG_TRUNCATE_SIZE.  If the string contains white spaces, quote it with angle brackets.
*
*-----------------------------------------------------------------------------
*/
static __device__ void PrintArg(FILE *filePtr, char *argStr, int noTruncate)
{
	int argLen = strlen(argStr);
	int printLen = argLen;
	if (!noTruncate && printLen > ARG_TRUNCATE_SIZE)
		printLen = ARG_TRUNCATE_SIZE;
	bool quote_it = (printLen == 0);
	for (int idx = 0; idx < printLen; idx++)
		if (isspace(argStr[idx])) {
			quote_it = TRUE;
			break;
		}
		if (quote_it) 
			fputc('{', filePtr);
		PrintStr(filePtr, argStr, printLen);
		if (quote_it) 
			fputc('}', filePtr);
}

/*
*-----------------------------------------------------------------------------
*
* TraceCode --
*    Print out a trace of a code line.  Level is used for indenting and marking lines and may be eval or procedure level.
* 
*-----------------------------------------------------------------------------
*/
static __device__ void TraceCode(traceInfo_pt traceInfoPtr, int level, char *command, int argc, const char *args[])
{
#if NOTSUP
	static struct timeval last_time;
	struct timeval this_time;
	gettimeofday(&this_time, 0);
	fprintf_(traceInfoPtr->filePtr, "%2d:", level);
	if (last_time.tv_sec != 0) {
		fprintf_(traceInfoPtr->filePtr, " (%luus)", (this_time.tv_sec - last_time.tv_sec)*1000000 + (this_time.tv_usec - last_time.tv_usec));
	}
	last_time = this_time;
#endif
	if (level > 20)
		level = 20;
	int idx;
	for (idx = 0; idx < level; idx++) 
		fprintf_(traceInfoPtr->filePtr, "  ");
	if (traceInfoPtr->noEval) {
		int printLen = strlen(command);
		if (!traceInfoPtr->noTruncate && printLen > CMD_TRUNCATE_SIZE)
			printLen = CMD_TRUNCATE_SIZE;
		PrintStr(traceInfoPtr->filePtr, command, printLen);
	} else {
		for (idx = 0; idx < argc; idx++) {
			if (idx > 0)
				fputc(' ', traceInfoPtr->filePtr);
			PrintArg(traceInfoPtr->filePtr, (char *)args[idx], traceInfoPtr->noTruncate);
		}
	}
	fputc('\n', traceInfoPtr->filePtr);
	if (traceInfoPtr->flush)
		fflush(traceInfoPtr->filePtr);
}

/*
*-----------------------------------------------------------------------------
*
* CmdTraceRoutine --
*  Routine called by Tcl_Eval to trace a command.
*
*-----------------------------------------------------------------------------
*/
__device__  static void CmdTraceRoutine(ClientData clientData, Tcl_Interp *interp, int level, char *command, Tcl_CmdProc *cmdProc, ClientData cmdClientData, int argc, const char *args[])
{
	Interp *iPtr = (Interp *)interp;
	traceInfo_pt traceInfoPtr = (traceInfo_pt)clientData;
	if (!traceInfoPtr->procCalls) {
		TraceCode(traceInfoPtr, level, command, argc, args);
	} else {
		if (TclFindProc(iPtr, (char *)args[0]) != NULL) {
			int procLevel = (iPtr->varFramePtr == NULL ? 0 : iPtr->varFramePtr->level);
			TraceCode(traceInfoPtr, procLevel, command, argc, args);
		}
	}
}

/*
*-----------------------------------------------------------------------------
*
* Tcl_CmdtraceCmd --
*     Implements the TCL trace command:
*     cmdtrace level|on [noeval] [notruncate] [flush] [procs] [filehdl]
*     cmdtrace off
*     cmdtrace depth
*
* Results:
*  Standard TCL results.
*
*-----------------------------------------------------------------------------
*/
static __device__ int Tcl_CmdtraceCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	//Interp *iPtr = (Interp *)interp;
	traceInfo_pt infoPtr = (traceInfo_pt)clientData;
	int idx;
	char *fileHandle;
	if (argc < 2)
		goto argumentError;

	// Handle `depth' sub-command.
	if (STREQU(args[1], "depth")) {
		if (argc != 2)
			goto argumentError;
		sprintf(interp->result, "%d", infoPtr->depth);
		return TCL_OK;
	}

	// If a trace is in progress, delete it now.
	if (infoPtr->traceHolder != NULL) {
		Tcl_DeleteTrace(interp, infoPtr->traceHolder);
		infoPtr->depth = 0;
		infoPtr->traceHolder = NULL;
	}

	// Handle off sub-command.
	if (STREQU(args[1], "off")) {
		if (argc != 2)
			goto argumentError;
		return TCL_OK;
	}

	infoPtr->noEval     = FALSE;
	infoPtr->noTruncate = FALSE;
	infoPtr->procCalls  = FALSE;
	infoPtr->flush      = FALSE;
	infoPtr->filePtr    = stdout;
	fileHandle          = NULL;

	for (idx = 2; idx < argc; idx++) {
		if (STREQU(args[idx], "notruncate")) {
			if (infoPtr->noTruncate)
				goto argumentError;
			infoPtr->noTruncate = TRUE;
			continue;
		}
		if (STREQU(args[idx], "noeval")) {
			if (infoPtr->noEval)
				goto argumentError;
			infoPtr->noEval = TRUE;
			continue;
		}
		if (STREQU(args[idx], "flush")) {
			if (infoPtr->flush)
				goto argumentError;
			infoPtr->flush = TRUE;
			continue;
		}
		if (STREQU(args[idx], "procs")) {
			if (infoPtr->procCalls)
				goto argumentError;
			infoPtr->procCalls = TRUE;
			continue;
		}
		if (STRNEQU(args[idx], "std", 3) || STRNEQU(args[idx], "file", 4)) {
			if (fileHandle != NULL)
				goto argumentError;
			fileHandle = (char *)args[idx];
			continue;
		}
		goto invalidOption;
	}

	if (STREQU(args[1], "on")) {
		infoPtr->depth = MAXINT;
	} else {
		if (Tcl_GetInt(interp, args[1], &(infoPtr->depth)) != TCL_OK)
			return TCL_ERROR;
	}
	if (fileHandle != NULL) {
		OpenFile_ *tclFilePtr;
		if (TclGetOpenFile(interp, fileHandle, &tclFilePtr) != TCL_OK)
			return TCL_ERROR;
		if (!tclFilePtr->writable) {
			Tcl_AppendResult(interp, "file not writable: ", fileHandle, (char *)NULL);
			return TCL_ERROR;
		}
		infoPtr->filePtr = tclFilePtr->f;
	}

	infoPtr->traceHolder = Tcl_CreateTrace(interp, infoPtr->depth, CmdTraceRoutine, (ClientData)infoPtr);
	return TCL_OK;

argumentError:
	Tcl_AppendResult (interp, "wrong # args: ", args[0], " level | on [noeval] [notruncate] [flush] [procs]", "[handle] | off | depth", (char *)NULL);
	return TCL_ERROR;

invalidOption:
	Tcl_AppendResult (interp, "invalid option: expected ", "one of \"noeval\", \"notruncate\", \"procs\", ", "\"flush\" or a file handle", (char *)NULL);
	return TCL_ERROR;
}

/*
*-----------------------------------------------------------------------------
*
*  CleanUpDebug --
*
*  Release the client data area when the trace command is deleted.
*
*-----------------------------------------------------------------------------
*/
static __device__ void CleanUpDebug(ClientData clientData)
{
	traceInfo_pt infoPtr = (traceInfo_pt)clientData;
	if (infoPtr->traceHolder != NULL)
		Tcl_DeleteTrace(infoPtr->interp, infoPtr->traceHolder);
	_freeFast((char *)infoPtr);
}

/*
*-----------------------------------------------------------------------------
*
*  Tcl_InitDebug --
*
*  Initialize the TCL debugging commands.
*
*-----------------------------------------------------------------------------
*/
__device__ void TclEx_InitDebug(Tcl_Interp *interp)
{
	traceInfo_pt infoPtr;
	infoPtr = (traceInfo_pt)_allocFast(sizeof(traceInfo_t));
	infoPtr->interp      = interp;
	infoPtr->traceHolder = NULL;
	infoPtr->noEval      = FALSE;
	infoPtr->noTruncate  = FALSE;
	infoPtr->procCalls   = FALSE;
	infoPtr->flush       = FALSE;
	infoPtr->depth       = 0;
	Tcl_CreateCommand(interp, "cmdtrace", Tcl_CmdtraceCmd, (ClientData)infoPtr, CleanUpDebug);
}
