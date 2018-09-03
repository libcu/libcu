// tclTest.c --
//
//	Test driver for TCL.
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include <stdio.h>
#include <stdlib.h>
#include "tcl.h"
#include "tclEx+Int.h"
#ifdef DEBUGGER
#include "tclEx+Dbg.h"
#endif

#if 1

// From generated load_extensions.c
__device__ void Tcl_InitExtensions(Tcl_Interp *interp);

Tcl_Interp *_interp;
Tcl_CmdBuf _buffer;
bool _quitFlag = false;
//char _initCmd[] = "puts stdout \"Embedded Tcl 6.8.0\n\"";
//char _initCmd[] = "source tcl_sys/autoinit.tcl";
//char _initCmd[] = "cd tests; source all";
char _initCmd[] = "cd tests; source file.test";

#ifdef TCL_MEM_DEBUG
char _dumpFile[100];
int cmdCheckmem(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " fileName\"", (char *)NULL);
		return TCL_ERROR;
	}
	strcpy(_dumpFile, args[1]);
	_quitFlag = true;
	return TCL_OK;
}
#endif

int main(int argc, const char *args[]) {
	_interp = Tcl_CreateInterp();
#ifdef TCL_MEM_DEBUG
	Tcl_InitMemory(_interp);
#endif
	TclEx_InitDebug(_interp);
	TclEx_InitGeneral(_interp);
#ifdef DEBUGGER
	TclEx_InitDebug(_interp);
#endif

	// Init any static extensions
	TclEx_InitExtensions(_interp);
#ifdef TCL_MEM_DEBUG
	Tcl_CreateCommand(_interp, "checkmem", cmdCheckmem, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
	_buffer = Tcl_CreateCmdBuf();
	int result;
	FILE *in;
	FILE *out;
	if (argc > 1 && strcmp(args[1], "-"))
	{
		char *filename = (char *)args[1]+1;

		// Before we eval the file, create an args global containing the remaining arguments
		char *args2 = Tcl_Merge(argc - 2, args + 2);
		Tcl_SetVar(_interp, "argv", args2, TCLGLOBAL__ONLY);
		_freeFast(args2);

		result = Tcl_EvalFile(_interp, filename);
		if (result != TCL_OK)
		{
			// And make sure we print an informative error if something goes wrong
			Tcl_AddErrorInfo(_interp, "");
			printf("%s\n", Tcl_GetVar(_interp, "errorInfo", TCL_LEAVE_ERR_MSG));
			exit(1);
		}
		exit(0);
	}
	else
	{
		// Are we in interactive mode or script from stdin mode?
		int noninteractive = (argc > 1);
		in = stdin;
		out = stdout;

#ifndef TCL_GENERIC_ONLY
		if (!noninteractive)
		{
			result = Tcl_Eval(_interp, _initCmd, 0, (char **)NULL);
			if (result != TCL_OK)
			{
				printf("%s\n", _interp->result);
				exit(1);
			}
		}
#endif
		bool gotPartial = false;
		while (true)
		{
			clearerr(in);
			if (!gotPartial)
			{
				if (!noninteractive) fputs("% ", out);
				fflush(out);
			}
			char line[1000];
			if (fgets(line, 1000, in) == NULL)
			{
				if (!gotPartial) exit(0);
				line[0] = 0;
			}
			char *cmd = Tcl_AssembleCmd(_buffer, line);
			if (cmd == NULL)
			{
				gotPartial = true;
				continue;
			}

			gotPartial = false;
#ifdef TCL_NO_HISTORY
			result = Tcl_Eval(_interp, cmd, 0, (char **)NULL);
#else
			result = Tcl_RecordAndEval(_interp, cmd, 0);
#endif
			if (result == TCL_OK)
			{
				if (*_interp->result != 0 && !noninteractive) printf("%s\n", _interp->result);
				if (_quitFlag)
				{
					Tcl_DeleteInterp(_interp);
					Tcl_DeleteCmdBuf(_buffer);
#ifdef TCL_MEM_DEBUG
					Tcl_DumpActiveMemory(_dumpFile);
#endif
					exit(0);
				}
			}
			else
			{
				if (result == TCL_ERROR) printf("Error");
				else printf("Error %d", result);
				if (*_interp->result != 0) printf(": %s\n", _interp->result);
				else printf("\n");
			}
		}
	}
}

#endif