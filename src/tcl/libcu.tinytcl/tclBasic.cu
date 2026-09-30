// tclBasic.c --
//
//	Contains the basic facilities for TCL command interpretation, including interpreter creation and deletion, command creation
//	and deletion, and command parsing and execution.
//
// Copyright 1987-1992 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

// The following structure defines all of the commands in the Tcl core, and the C procedures that execute them.
typedef struct {
	const char *name;		// Name of command.
	Tcl_CmdProc *proc;		// Procedure that executes command.
} CmdInfo;

// Built-in commands, and the procedures associated with them:
static __constant__ CmdInfo _builtInCmds[] = {
	// Commands in the generic core:
	{"append",		Tcl_AppendCmd},
	{"array",		Tcl_ArrayCmd},
	{"break",		Tcl_BreakCmd},
	{"case",		Tcl_CaseCmd},
	{"catch",		Tcl_CatchCmd},
	{"concat",		Tcl_ConcatCmd},
	{"continue",	Tcl_ContinueCmd},
	{"error",		Tcl_ErrorCmd},
	{"eval",		Tcl_EvalCmd},
	{"expr",		Tcl_ExprCmd},
	{"for",			Tcl_ForCmd},
	{"foreach",		Tcl_ForeachCmd},
	{"format",		Tcl_FormatCmd},
	{"global",		TclGLOBAL_Cmd},
	{"glob",		Tcl_GlobCmd},
#ifndef TCL_NO_HISTORY
	{"h",			Tcl_HistoryCmd},
	{"history",		Tcl_HistoryCmd},
#endif
	{"if",			Tcl_IfCmd},
	{"incr",		Tcl_IncrCmd},
	{"info",		Tcl_InfoCmd},
	{"join",		Tcl_JoinCmd},
	{"lappend",		Tcl_LappendCmd},
	{"lindex",		Tcl_LindexCmd},
	{"linsert",		Tcl_LinsertCmd},
	{"list",		Tcl_ListCmd},
	{"llength",		Tcl_LlengthCmd},
	{"lrange",		Tcl_LrangeCmd},
	{"lreplace",	Tcl_LreplaceCmd},
	{"lsearch",		Tcl_LsearchCmd},
	{"lsort",		Tcl_LsortCmd},
	{"proc",		Tcl_ProcCmd},
	{"regexp",		Tcl_RegexpCmd},
	{"regsub",		Tcl_RegsubCmd},
	{"rename",		Tcl_RenameCmd},
	{"return",		Tcl_ReturnCmd},
	{"scan",		Tcl_ScanCmd},
	{"set",			Tcl_SetCmd},
	{"split",		Tcl_SplitCmd},
	{"string",		Tcl_StringCmd},
	{"trace",		Tcl_TraceCmd},
	{"unset",		Tcl_UnsetCmd},
	{"uplevel",		Tcl_UplevelCmd},
	{"upvar",		Tcl_UpvarCmd},
	{"while",		Tcl_WhileCmd},

	// Commands in the UNIX core:
	{"exec",		Tcl_ExecCmd},
	{"time",		Tcl_TimeCmd},
	{"pid",			Tcl_PidCmd},
#ifdef HAVE_TCL_LOAD
	{"load",		Tcl_LoadCmd},
#endif

#ifndef TCL_GENERIC_ONLY
	{"cd",			Tcl_CdCmd},
	{"close",		Tcl_CloseCmd},
	{"eof",			Tcl_EofCmd},
	{"exit",		Tcl_ExitCmd},
	{"file",		Tcl_FileCmd},
	{"flush",		Tcl_FlushCmd},
	{"gets",		Tcl_GetsCmd},
	{"open",		Tcl_OpenCmd},
	{"puts",		Tcl_PutsCmd},
	{"pwd",			Tcl_PwdCmd},
	{"read",		Tcl_ReadCmd},
	{"seek",		Tcl_SeekCmd},
	{"source",		Tcl_SourceCmd},
	{"tell",		Tcl_TellCmd},
#endif /* TCL_GENERIC_ONLY */
	{NULL, (Tcl_CmdProc *)NULL}
};

/*
*----------------------------------------------------------------------
*
* Tcl_CreateInterp --
*	Create a new TCL command interpreter.
*
* Results:
*	The return value is a token for the interpreter, which may be used in calls to procedures like Tcl_CreateCmd, Tcl_Eval, or
*	Tcl_DeleteInterp.
*
* Side effects:
*	The command interpreter is initialized with an empty variable table and the built-in commands.
*
*----------------------------------------------------------------------
*/
__device__ Tcl_Interp *Tcl_CreateInterp() {
	register Interp *iPtr = (Interp *)_allocFast(sizeof(Interp));
	memset(iPtr, 0, sizeof(*iPtr));
	iPtr->result = iPtr->resultSpace;
	iPtr->freeProc = 0;
	iPtr->errorLine = 0;
	Tcl_InitHashTable(&iPtr->commandTable, TCL_STRING_KEYS);
	Tcl_InitHashTable(&iPtr->globalTable, TCL_STRING_KEYS);
	iPtr->numLevels = 0;
	iPtr->framePtr = NULL;
	iPtr->varFramePtr = NULL;
	iPtr->activeTracePtr = NULL;
	iPtr->numEvents = 0;
	iPtr->events = NULL;
	iPtr->curEvent = 0;
	iPtr->curEventNum = 0;
	iPtr->revPtr = NULL;
	iPtr->historyFirst = NULL;
	iPtr->revDisables = 1;
	iPtr->evalFirst = iPtr->evalLast = NULL;
	iPtr->appendResult = NULL;
	iPtr->appendAvl = 0;
	iPtr->appendUsed = 0;
	iPtr->numFiles = 0;
	iPtr->filePtrArray = NULL;
	iPtr->num_regexps = DEFAULT_NUM_REGEXPS;
	iPtr->regexps = (CompiledRegexp *)_allocFast(sizeof(CompiledRegexp) * iPtr->num_regexps);
	int i;
	for (i = 0; i < iPtr->num_regexps; i++) {
		iPtr->regexps[i].pattern = NULL;
		iPtr->regexps[i].length = -1;
		iPtr->regexps[i].regexp = NULL;
	}
	iPtr->cmdCount = 0;
	iPtr->noEval = 0;
	iPtr->signal = 0;
	iPtr->catch_level = 0;
	iPtr->scriptFile = NULL;
	iPtr->flags = 0;
	iPtr->tracePtr = NULL;
	iPtr->resultSpace[0] = 0;

	// Create the built-in commands.  Do it here, rather than calling Tcl_CreateCommand, because it's faster (there's no need to check for a pre-existing command by the same name).
	for (register CmdInfo *cmdInfoPtr = _builtInCmds; cmdInfoPtr->name != NULL; cmdInfoPtr++) {
		int new_;
		Tcl_HashEntry *hPtr = Tcl_CreateHashEntry(&iPtr->commandTable, cmdInfoPtr->name, &new_);
		if (new_) {
			register Command *cmdPtr = (Command *)_allocFast(sizeof(Command));
			cmdPtr->proc = cmdInfoPtr->proc;
			cmdPtr->clientData = (ClientData)NULL;
			cmdPtr->deleteProc = NULL;
			Tcl_SetHashValue(hPtr, cmdPtr);
		}
	}
#ifndef TCL_GENERIC_ONLY
	TclSetupEnv((Tcl_Interp *)iPtr);
#endif
	return (Tcl_Interp *)iPtr;
}

/*
*----------------------------------------------------------------------
*
* Tcl_DeleteInterp --
*	Delete an interpreter and free up all of the resources associated with it.
*
* Results:
*	None.
*
* Side effects:
*	The interpreter is destroyed.  The caller should never again use the interp token.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_DeleteInterp(Tcl_Interp *interp) {
	Interp *iPtr = (Interp *)interp;

	// If the interpreter is in use, delay the deletion until later.
	iPtr->flags |= DELETED;
	if (iPtr->numLevels != 0) {
		return;
	}

	// Free up any remaining resources associated with the interpreter.
	Tcl_HashSearch search;
	for (Tcl_HashEntry *hPtr = Tcl_FirstHashEntry(&iPtr->commandTable, &search); hPtr != NULL; hPtr = Tcl_NextHashEntry(&search)) {
		register Command *cmdPtr = (Command *)Tcl_GetHashValue(hPtr);
		if (cmdPtr->deleteProc != NULL) {
			(*cmdPtr->deleteProc)(cmdPtr->clientData);
		}
		_freeFast((char *)cmdPtr);
	}
	Tcl_DeleteHashTable(&iPtr->commandTable);
	TclDeleteVars(iPtr, &iPtr->globalTable);
	int i;
	if (iPtr->events != NULL) {
		for (i = 0; i < iPtr->numEvents; i++) {
			_freeFast(iPtr->events[i].command);
		}
		_freeFast((char *)iPtr->events);
	}
	while (iPtr->revPtr != NULL) {
		HistoryRev *nextPtr = iPtr->revPtr->nextPtr;
		_freeFast((char *)iPtr->revPtr);
		iPtr->revPtr = nextPtr;
	}
	if (iPtr->appendResult != NULL) {
		_freeFast(iPtr->appendResult);
	}
#ifndef TCL_GENERIC_ONLY
	if (iPtr->numFiles > 0) {
		for (i = 0; i < iPtr->numFiles; i++) {
			OpenFile_ *filePtr;
			filePtr = iPtr->filePtrArray[i];
			if (filePtr == NULL) {
				continue;
			}
			if (i >= 3) {
				fclose(filePtr->f);
				if (filePtr->f2 != NULL) {
					fclose(filePtr->f2);
				}
				if (filePtr->numPids > 0) {
					//Tcl_DetachPids(filePtr->numPids, filePtr->pidPtr);
					_freeFast((char *)filePtr->pidPtr);
				}
			}
			_freeFast((char *)filePtr);
		}
		_freeFast((char *)iPtr->filePtrArray);
	}
#endif
	for (i = 0; i < iPtr->num_regexps; i++) {
		if (iPtr->regexps[i].pattern == NULL) {
			break;
		}
		_freeFast(iPtr->regexps[i].pattern);
		regfree(iPtr->regexps[i].regexp);
		_freeFast((char *)iPtr->regexps[i].regexp);
	}
	_freeFast((char *)iPtr->regexps);
	while (iPtr->tracePtr != NULL) {
		Trace *nextPtr = iPtr->tracePtr->nextPtr;
		_freeFast((char *)iPtr->tracePtr);
		iPtr->tracePtr = nextPtr;
	}
	_freeFast((char *)iPtr);
}

/*
*----------------------------------------------------------------------
*
* Tcl_CreateCommand --
*	Define a new command in a command table.
*
* Results:
*	None.
*
* Side effects:
*	If a command named cmdName already exists for interp, it is deleted.  In the future, when cmdName is seen as the name of
*	a command by Tcl_Eval, proc will be called.  When the command is deleted from the table, deleteProc will be called.  See the
*	manual entry for details on the calling sequence.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_CreateCommand(Tcl_Interp *interp, const char *cmdName, Tcl_CmdProc *proc, ClientData clientData, Tcl_CmdDeleteProc *deleteProc) {
	Interp *iPtr = (Interp *)interp;
	register Command *cmdPtr;
	int new_;
	Tcl_HashEntry *hPtr = Tcl_CreateHashEntry(&iPtr->commandTable, cmdName, &new_);
	if (!new_) {
		// Command already exists:  delete the old one.
		cmdPtr = (Command *)Tcl_GetHashValue(hPtr);
		if (cmdPtr->deleteProc != NULL) {
			(*cmdPtr->deleteProc)(cmdPtr->clientData);
		}
	}
	else {
		cmdPtr = (Command *)_allocFast(sizeof(Command));
		Tcl_SetHashValue(hPtr, cmdPtr);
	}
	cmdPtr->proc = proc;
	cmdPtr->clientData = clientData;
	cmdPtr->deleteProc = deleteProc;
}

/*
*----------------------------------------------------------------------
*
* Tcl_DeleteCommand --
*	Remove the given command from the given interpreter.
*
* Results:
*	0 is returned if the command was deleted successfully.
*	-1 is returned if there didn't exist a command by that name.
*
* Side effects:
*	CmdName will no longer be recognized as a valid command for interp.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_DeleteCommand(Tcl_Interp *interp, char *cmdName) {
	Interp *iPtr = (Interp *)interp;
	Tcl_HashEntry *hPtr = Tcl_FindHashEntry(&iPtr->commandTable, cmdName);
	if (hPtr == NULL) {
		return -1;
	}
	Command *cmdPtr = (Command *)Tcl_GetHashValue(hPtr);
	if (cmdPtr->deleteProc != NULL) {
		(*cmdPtr->deleteProc)(cmdPtr->clientData);
	}
	_freeFast((char *)cmdPtr);
	Tcl_DeleteHashEntry(hPtr);
	return 0;
}

/*
*-----------------------------------------------------------------
*
* Tcl_Eval --
*	Parse and execute a command in the Tcl language.
*
* Results:
*	The return value is one of the return codes defined in tcl.hd (such as TCL_OK), and interp->result contains a string value
*	to supplement the return code.  The value of interp->result will persist only until the next call to Tcl_Eval:  copy it or
*	lose it! *TermPtr is filled in with the character just after the last one that was part of the command (usually a NULL
*	character or a closing bracket).
*
* Side effects:
*	Almost certainly;  depends on the command.
*
*-----------------------------------------------------------------
*/
__device__ int Tcl_Eval(Tcl_Interp *interp, char *cmd, int flags, char **termPtr) {
#define NUM_CHARS 200
#define NUM_ARGS 10
	register Interp *iPtr = (Interp *)interp;

	// Initialize the result to an empty string and clear out any error information.  This makes sure that we return an empty
	// result if there are no commands in the command string.
	Tcl_FreeResult((Tcl_Interp *)iPtr);
	iPtr->result = iPtr->resultSpace;
	iPtr->resultSpace[0] = 0;
	int result = TCL_OK; // Return value.

	// Check depth of nested calls to Tcl_Eval:  if this gets too large, it's probably because of an infinite loop somewhere.
	iPtr->numLevels++;
	if (iPtr->numLevels > MAX_NESTING_DEPTH) {
		iPtr->numLevels--;
		iPtr->result = (char *)"too many nested calls to Tcl_Eval (infinite loop?)";
		return TCL_ERROR;
	}

	// Initialize the area in which command copies will be assembled.
	char copyStorage[NUM_CHARS];
	ParseValue pv;
	pv.buffer = copyStorage;
	pv.end = copyStorage + NUM_CHARS - 1;
	pv.expandProc = TclExpandParseValue;
	pv.clientData = (ClientData)NULL;

	register char *src = cmd; // Points to current character in cmd.
	char termChar; // Return when this character is found (either ']' or '\0').  Zero means that newlines terminate commands.
	if (flags & TCL_BRACKET_TERM) {
		termChar = ']';
	}
	else {
		termChar = 0;
	}
	char *dummy; // Make termPtr point here if it was originally NULL.
	if (termPtr == NULL) {
		termPtr = &dummy;
	}
	*termPtr = src;
	char *cmdStart = src; // Points to first non-blank char. in command (used in calling trace procedures).

	if (flags & TCL_CATCH_SIGNAL) {
		iPtr->catch_level++;
	}

	// The storage immediately below is used to generate a copy of the command, after all argument substitutions.  Pv will contain the args values passed to the command procedure.
	char *oldBuffer;

	// There can be many sub-commands (separated by semi-colons or newlines) in one command string.  This outer loop iterates over individual commands.
	int i;
	const char *argStorage[NUM_ARGS]; // This procedure generates an (args, argc) array for the command, It starts out with stack-allocated space but uses dynamically- allocated storage to increase it if needed.
	const char **args = argStorage;
	int argSize = NUM_ARGS;
	const char *ellipsis = ""; // Used in setting errorInfo variable; set to "..." to indicate that not all of offending command is included in errorInfo.  "" means that the command is all there.
	while (*src != termChar) {
		if (iPtr->catch_level && iPtr->signal) {
			break;
		}

		iPtr->flags &= ~(ERR_IN_PROGRESS | ERROR_CODE_SET);

		// Skim off leading white space and semi-colons, and skip comments.
		while (true) {
			register int c = *src;
			if (CHAR_TYPE(c) != TCL_SPACE && c != ';' && c != '\n') {
				break;
			}
			src += 1;
		}
		if (*src == '#') {
			for (src++; *src != 0; src++) {
				if (*src == '\n' && src[-1] != '\\') {
					src++;
					break;
				}
			}
			continue;
		}
		cmdStart = src;

		// Parse the words of the command, generating the argc and args for the command procedure.  May have to call
		// TclParseWords several times, expanding the args array between calls.
		pv.next = oldBuffer = pv.buffer;
		int argc = 0;
		while (true) {
			// Note:  the "- 2" below guarantees that we won't use the last two args slots here.  One is for a NULL pointer to
			// mark the end of the list, and the other is to leave room for inserting the command name "unknown" as the first argument (see below).
			int newArgs, maxArgs = argSize - argc - 2;
			result = TclParseWords((Tcl_Interp *)iPtr, src, flags, maxArgs, termPtr, &newArgs, &args[argc], &pv);
			src = *termPtr;
			if (result != TCL_OK) {
				ellipsis = (char *)"...";
				goto done;
			}

			// Careful!  Buffer space may have gotten reallocated while parsing words.  If this happened, be sure to update all of the older args pointers to refer to the new space.
			if (oldBuffer != pv.buffer) {
				for (i = 0; i < argc; i++) {
					args[i] = pv.buffer + (args[i] - oldBuffer);
				}
				oldBuffer = pv.buffer;
			}
			argc += newArgs;
			if (newArgs < maxArgs) {
				args[argc] = (char *)NULL;
				break;
			}

			// Args didn't all fit in the current array.  Make it bigger.
			argSize *= 2;
			const char **newArgs2 = (const char **)_allocFast((unsigned)argSize * sizeof(char *));
			for (i = 0; i < argc; i++) {
				newArgs2[i] = args[i];
			}
			if (args != argStorage) {
				_freeFast((char *)args);
			}
			args = newArgs2;
		}

		// If this is an empty command (or if we're just parsing commands without evaluating them), then just skip to the next command.
		if (argc == 0 || iPtr->noEval) {
			continue;
		}
		args[argc] = NULL;

		// Save information for the history module, if needed.
		if (flags & TCL_RECORD_BOUNDS) {
			iPtr->evalFirst = cmdStart;
			iPtr->evalLast = src - 1;
		}

		// Find the procedure to execute this command.  If there isn't one, then see if there is a command "unknown".  If so,
		// invoke it instead, passing it the words of the original command as arguments.
		Tcl_HashEntry *hPtr = Tcl_FindHashEntry(&iPtr->commandTable, (char *)args[0]);
		if (hPtr == NULL) {
			hPtr = Tcl_FindHashEntry(&iPtr->commandTable, "unknown");
			if (hPtr == NULL) {
				Tcl_ResetResult(interp);
				Tcl_AppendResult(interp, "invalid command name: \"", args[0], "\"", (char *)NULL);
				result = TCL_ERROR;
				goto done;
			}
			for (i = argc; i >= 0; i--) {
				args[i + 1] = args[i];
			}
			args[0] = "unknown";
			argc++;
		}
		Command *cmdPtr = (Command *)Tcl_GetHashValue(hPtr);

		// Call trace procedures, if any.
		for (register Trace *tracePtr = iPtr->tracePtr; tracePtr != NULL; tracePtr = tracePtr->nextPtr) {
			if (tracePtr->level < iPtr->numLevels) {
				continue;
			}
			char saved = *src;
			*src = 0;
			(*tracePtr->proc)(tracePtr->clientData, interp, iPtr->numLevels, cmdStart, cmdPtr->proc, cmdPtr->clientData, argc, args);
			*src = saved;
		}

		// At long last, invoke the command procedure.  Reset the result to its default empty value first (it could have
		// gotten changed by earlier commands in the same command string).
		iPtr->cmdCount++;
		Tcl_FreeResult(iPtr);
		iPtr->result = iPtr->resultSpace;
		iPtr->resultSpace[0] = 0;
		result = (*cmdPtr->proc)(cmdPtr->clientData, interp, argc, args);
		if (result != TCL_OK) {
			break;
		}
	}

	if (iPtr->catch_level && iPtr->signal) {
		// Got a signal in a catch so throw the signal
		result = TCL_SIGNAL;
		Tcl_SetResult(interp, Tcl_SignalId(iPtr->signal), TCL_STATIC);
		iPtr->signal = 0;
	}

	// Free up any extra resources that were allocated.
done:
	if (pv.buffer != copyStorage) {
		_freeFast((char *)pv.buffer);
	}
	if (args != argStorage) {
		_freeFast((char *)args);
	}
	iPtr->numLevels--;
	if (iPtr->numLevels == 0) {
		if (result == TCL_RETURN) {
			result = TCL_OK;
		}
		if (result == TCL_SIGNAL) {
			result = TCL_ERROR;
		}
		if (result != TCL_OK && result != TCL_ERROR) {
			Tcl_ResetResult(interp);
			if (result == TCL_BREAK) {
				iPtr->result = (char *)"invoked \"break\" outside of a loop";
			}
			else if (result == TCL_CONTINUE) {
				iPtr->result = (char *)"invoked \"continue\" outside of a loop";
			}
			else {
				iPtr->result = iPtr->resultSpace;
				sprintf(iPtr->resultSpace, "command returned bad code: %d", result);
			}
			result = TCL_ERROR;
		}
		if (iPtr->flags & DELETED) {
			Tcl_DeleteInterp(interp);
		}
	}

	if (flags & TCL_CATCH_SIGNAL) {
		iPtr->catch_level--;
	}

	// If an error occurred, record information about what was being executed when the error occurred.
	if (result == TCL_ERROR && !(iPtr->flags & ERR_ALREADY_LOGGED)) {
		// Compute the line number where the error occurred.
		iPtr->errorLine = 1;
		register char *p;
		for (p = cmd; p != cmdStart; p++) {
			if (*p == '\n') {
				iPtr->errorLine++;
			}
		}
		for (; isspace(*p) || *p == ';'; p++) {
			if (*p == '\n') {
				iPtr->errorLine++;
			}
		}
		// Figure out how much of the command to print in the error message (up to a certain number of characters, or up to the first new-line).
		int numChars = (int)(src - cmdStart);
		if (numChars > (NUM_CHARS - 50)) {
			numChars = NUM_CHARS - 50;
			ellipsis = " ...";
		}
		if (!(iPtr->flags & ERR_IN_PROGRESS)) {
			sprintf(copyStorage, "\n    while executing\n\"%.*s%s\"", numChars, cmdStart, ellipsis);
		}
		else {
			sprintf(copyStorage, "\n    invoked from within\n\"%.*s%s\"", numChars, cmdStart, ellipsis);
		}
		Tcl_AddErrorInfo(interp, copyStorage);
		iPtr->flags &= ~ERR_ALREADY_LOGGED;
	}
	else {
		iPtr->flags &= ~ERR_ALREADY_LOGGED;
	}

	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_CreateTrace --
*	Arrange for a procedure to be called to trace command execution.
*
* Results:
*	The return value is a token for the trace, which may be passed to Tcl_DeleteTrace to eliminate the trace.
*
* Side effects:
*	From now on, proc will be called just before a command procedure is called to execute a Tcl command.  Calls to proc will have the following form:
*
*	void proc(ClientData clientData, Tcl_Interp *interp, int level, char *command, int (*cmdProc)(), ClientData cmdClientData, int argc, const char *args[])
*	{
*	}
*
*	The clientData and interp arguments to proc will be the same as the corresponding arguments to this procedure.  Level gives
*	the nesting level of command interpretation for this interpreter (0 corresponds to top level).  Command gives the ASCII text of
*	the raw command, cmdProc and cmdClientData give the procedure that will be called to process the command and the ClientData value it
*	will receive, and argc and args give the arguments to the command, after any argument parsing and substitution.  Proc
*	does not return a value.
*
*----------------------------------------------------------------------
*/
__device__ Tcl_Trace Tcl_CreateTrace(Tcl_Interp *interp, int level, Tcl_CmdTraceProc *proc, ClientData clientData) {
	register Interp *iPtr = (Interp *)interp;
	register Trace *tracePtr = (Trace *)_allocFast(sizeof(Trace));
	tracePtr->level = level;
	tracePtr->proc = proc;
	tracePtr->clientData = clientData;
	tracePtr->nextPtr = iPtr->tracePtr;
	iPtr->tracePtr = tracePtr;
	return (Tcl_Trace)tracePtr;
}

/*
*----------------------------------------------------------------------
*
* Tcl_DeleteTrace --
*	Remove a trace.
*
* Results:
*	None.
*
* Side effects:
*	From now on there will be no more calls to the procedure given in trace.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_DeleteTrace(Tcl_Interp *interp, Tcl_Trace trace) {
	register Interp *iPtr = (Interp *)interp;
	register Trace *tracePtr = (Trace *)trace;
	if (iPtr->tracePtr == tracePtr) {
		iPtr->tracePtr = tracePtr->nextPtr;
		_freeFast((char *)tracePtr);
	}
	else {
		for (register Trace *tracePtr2 = iPtr->tracePtr; tracePtr2 != NULL; tracePtr2 = tracePtr2->nextPtr) {
			if (tracePtr2->nextPtr == tracePtr) {
				tracePtr2->nextPtr = tracePtr->nextPtr;
				_freeFast((char *)tracePtr);
				return;
			}
		}
	}
}

/*
*----------------------------------------------------------------------
*
* Tcl_AddErrorInfo --
*	Add information to a message being accumulated that describes the current error.
*
* Results:
*	None.
*
* Side effects:
*	The contents of message are added to the "errorInfo" variable. If Tcl_Eval has been called since the current value of errorInfo
*	was set, errorInfo is cleared before adding the new message.
*
*----------------------------------------------------------------------
*/
__device__ void Tcl_AddErrorInfo(Tcl_Interp *interp, char *message) {
	register Interp *iPtr = (Interp *)interp;
	// If an error is already being logged, then the new errorInfo is the concatenation of the old info and the new message.
	// If this is the first piece of info for the error, then the new errorInfo is the concatenation of the message in interp->result and the new message.
	if (!(iPtr->flags & ERR_IN_PROGRESS)) {
		Tcl_SetVar2(interp, (char *)"errorInfo", (char *)NULL, interp->result, TCLGLOBAL__ONLY);
		iPtr->flags |= ERR_IN_PROGRESS;

		// If the errorCode variable wasn't set by the code that generated the error, set it to "NONE".
		if (!(iPtr->flags & ERROR_CODE_SET)) {
			Tcl_SetVar2(interp, (char *)"errorCode", (char *)NULL, (char *)"NONE", TCLGLOBAL__ONLY);
		}
	}
	Tcl_SetVar2(interp, (char *)"errorInfo", (char *)NULL, message, TCLGLOBAL__ONLY | TCL_APPEND_VALUE);
}

/*
*----------------------------------------------------------------------
*
* Tcl_VarEval --
*	Given a variable number of string arguments, concatenate them all together and execute the result as a Tcl command.
*
* Results:
*	A standard Tcl return result.  An error message or other result may be left in interp->result.
*
* Side effects:
*	Depends on what was done by the command.
*
*----------------------------------------------------------------------
*/
__device__ int _Tcl_VarEval(Tcl_Interp *interp, va_list va) {
#define FIXED_SIZE 200
	// Copy the strings one after the other into a single larger string.  Use stack-allocated space for small commands, but if
	// the commands gets too large than call _allocFast to create the space.
	int spaceAvl = FIXED_SIZE;
	int spaceUsed = 0;
	char fixedSpace[FIXED_SIZE + 1];
	char *cmd = fixedSpace;
	while (true) {
		char *string = va_arg(va, char *);
		if (string == NULL) {
			break;
		}
		int length = strlen(string);
		if ((spaceUsed + length) > spaceAvl) {
			spaceAvl = spaceUsed + length;
			spaceAvl += spaceAvl / 2;
			char *new_ = (char *)_allocFast((unsigned)spaceAvl);
			memcpy(new_, cmd, spaceUsed);
			if (cmd != fixedSpace) {
				_freeFast(cmd);
			}
			cmd = new_;
		}
		strcpy(cmd + spaceUsed, string);
		spaceUsed += length;
	}
	cmd[spaceUsed] = '\0';

	int result = Tcl_Eval(interp, cmd, 0, (char **)NULL);
	if (cmd != fixedSpace) {
		_freeFast(cmd);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* TclGLOBAL_Eval --
*	Evaluate a command at global level in an interpreter.
*
* Results:
*	A standard Tcl result is returned, and interp->result is modified accordingly.
*
* Side effects:
*	The command string is executed in interp, and the execution is carried out in the variable context of global level (no
*	procedures active), just as if an "uplevel #0" command were being executed.
*
*----------------------------------------------------------------------
*/
__device__ int TclGLOBAL_Eval(Tcl_Interp *interp, char *command) {
	register Interp *iPtr = (Interp *)interp;
	CallFrame *savedVarFramePtr = iPtr->varFramePtr;
	iPtr->varFramePtr = NULL;
	int result = Tcl_Eval(interp, command, 0, (char **)NULL);
	iPtr->varFramePtr = savedVarFramePtr;
	return result;
}