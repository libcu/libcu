// tclProc.c --
//
//	This file contains routines that implement Tcl procedures, including the "proc" and "uplevel" commands.
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

// Forward references to procedures defined later in this file:
static __device__ int InterpProc(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[]);
static __device__ void ProcDeleteProc(ClientData clientData);

/*
*----------------------------------------------------------------------
*
* Tcl_ProcCmd --
*	This procedure is invoked to process the "proc" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result value.
*
* Side effects:
*	A new procedure gets created.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ProcCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " name args body\"", (char *)NULL);
		return TCL_ERROR;
	}
	register Arg *argPtr = NULL; // Initialization not needed, but prevents compiler warning.
	register Proc *procPtr = (Proc *)_allocFast(sizeof(Proc));
	procPtr->command = (char *)_allocFast((unsigned)strlen(args[3]) + 1);
	strcpy(procPtr->command, args[3]);
	procPtr->argPtr = NULL;
	procPtr->uses = 1; // 1 for initial definition

	// Break up the argument list into argument specifiers, then process each argument specifier.
	int argCount;
	const char **argArray = NULL;
	int result = Tcl_SplitList(interp, (char *)args[2], &argCount, &argArray);
	if (result != TCL_OK) {
		goto procError;
	}
	Arg *lastArgPtr = NULL;
	for (int i = 0; i < argCount; i++) {
		// Now divide the specifier up into name and default.
		int fieldCount;
		const char **fieldValues;
		result = Tcl_SplitList(interp, (char *)argArray[i], &fieldCount, &fieldValues);
		if (result != TCL_OK) {
			goto procError;
		}
		if (fieldCount > 2) {
			_freeFast((char *)fieldValues);
			Tcl_AppendResult(interp, "too many fields in argument specifier \"", argArray[i], "\"", (char *)NULL);
			result = TCL_ERROR;
			goto procError;
		}
		if (fieldCount == 0 || *fieldValues[0] == 0) {
			_freeFast((char *) fieldValues);
			Tcl_AppendResult(interp, "procedure \"", args[1], "\" has argument with no name", (char *)NULL);
			result = TCL_ERROR;
			goto procError;
		}
		int nameLength = strlen(fieldValues[0]) + 1;
		int valueLength;
		if (fieldCount == 2) {
			valueLength = strlen(fieldValues[1]) + 1;
		} else {
			valueLength = 0;
		}
		argPtr = (Arg *)_allocFast((unsigned)(sizeof(Arg) - sizeof(argPtr->name) + nameLength + valueLength));
		if (lastArgPtr == NULL) {
			procPtr->argPtr = argPtr;
		} else {
			lastArgPtr->nextPtr = argPtr;
		}
		lastArgPtr = argPtr;
		argPtr->nextPtr = NULL;
		strcpy(argPtr->name, fieldValues[0]);
		if (fieldCount == 2) {
			argPtr->defValue = argPtr->name + nameLength;
			strcpy(argPtr->defValue, fieldValues[1]);
		} else {
			argPtr->defValue = NULL;
		}
		_freeFast((char *)fieldValues);
	}
	Tcl_CreateCommand(interp, (char *)args[1], InterpProc, (ClientData)procPtr, ProcDeleteProc);
	_freeFast((char *)argArray);
	return TCL_OK;

procError:
	_freeFast(procPtr->command);
	while (procPtr->argPtr != NULL) {
		argPtr = procPtr->argPtr;
		procPtr->argPtr = argPtr->nextPtr;
		_freeFast((char *)argPtr);
	}
	_freeFast((char *)procPtr);
	if (argArray != NULL) {
		_freeFast((char *)argArray);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* TclGetFrame --
*	Given a description of a procedure frame, such as the first argument to an "uplevel" or "upvar" command, locate the
*	call frame for the appropriate level of procedure.
*
* Results:
*	The return value is -1 if an error occurred in finding the frame (in this case an error message is left in interp->result).
*	1 is returned if string was either a number or a number preceded by "#" and it specified a valid frame.  0 is returned if string
*	isn't one of the two things above (in this case, the lookup acts as if string were "1").  The variable pointed to by
*	framePtrPtr is filled in with the address of the desired frame (unless an error occurs, in which case it isn't modified).
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int TclGetFrame(Tcl_Interp *interp, char *string, CallFrame **framePtrPtr)
{
	register Interp *iPtr = (Interp *)interp;
	if (iPtr->varFramePtr == NULL) {
		iPtr->result = "already at top level";
		return -1;
	}
	// Parse string to figure out which level number to go to.
	int level;
	int result = 1;
	if (*string == '#') {
		if (Tcl_GetInt(interp, string+1, &level) != TCL_OK) {
			return -1;
		}
		if (level < 0) {
levelError:
			Tcl_AppendResult(interp, "bad level \"", string, "\"", (char *)NULL);
			return -1;
		}
	} else if (isdigit(*string)) {
		if (Tcl_GetInt(interp, string, &level) != TCL_OK) {
			return -1;
		}
		level = iPtr->varFramePtr->level - level;
	} else {
		level = iPtr->varFramePtr->level - 1;
		result = 0;
	}
	// Figure out which frame to use, and modify the interpreter so its variables come from that frame.
	CallFrame *framePtr;
	if (level == 0) {
		framePtr = NULL;
	} else {
		for (framePtr = iPtr->varFramePtr; framePtr != NULL; framePtr = framePtr->callerVarPtr) {
			if (framePtr->level == level) {
				break;
			}
		}
		if (framePtr == NULL) {
			goto levelError;
		}
	}
	*framePtrPtr = framePtr;
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_UplevelCmd --
*	This procedure is invoked to process the "uplevel" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result value.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_UplevelCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[])
{
	register Interp *iPtr = (Interp *)interp;
	if (argc < 2) {
uplevelSyntax:
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " ?level? command ?arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}

	// Find the level to use for executing the command.
	CallFrame *framePtr;
	int result = TclGetFrame(interp, (char *)args[1], &framePtr);
	if (result == -1) {
		return TCL_ERROR;
	}
	argc -= (result+1);
	if (argc == 0) {
		goto uplevelSyntax;
	}
	args += (result+1);

	// Modify the interpreter state to execute in the given frame.
	CallFrame *savedVarFramePtr = iPtr->varFramePtr;
	iPtr->varFramePtr = framePtr;

	// Execute the residual arguments as a command.
	if (argc == 1) {
		result = Tcl_Eval(interp, (char *)args[0], 0, (char **)NULL);
	} else {
		char *cmd = Tcl_Concat(argc, args);
		result = Tcl_Eval(interp, cmd, 0, (char **)NULL);
		_freeFast(cmd);
	}
	if (result == TCL_ERROR) {
		char msg[60];
		sprintf(msg, "\n    (\"uplevel\" body line %d)", interp->errorLine);
		Tcl_AddErrorInfo(interp, msg);
	}

	// Restore the variable frame, and return.
	iPtr->varFramePtr = savedVarFramePtr;
	return result;
}

/*
*----------------------------------------------------------------------
*
* TclFindProc --
*	Given the name of a procedure, return a pointer to the record describing the procedure.
*
* Results:
*	NULL is returned if the name doesn't correspond to any procedure.  Otherwise the return value is a pointer to the procedure's record.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ Proc *TclFindProc(Interp *iPtr, char *procName)
{
	Tcl_HashEntry *hPtr = Tcl_FindHashEntry(&iPtr->commandTable, procName);
	if (hPtr == NULL) {
		return NULL;
	}
	Command *cmdPtr = (Command *)Tcl_GetHashValue(hPtr);
	if (cmdPtr->proc != InterpProc) {
		return NULL;
	}
	return (Proc *)cmdPtr->clientData;
}

/*
*----------------------------------------------------------------------
*
* TclIsProc --
*	Tells whether a command is a Tcl procedure or not.
*
* Results:
*	If the given command is actuall a Tcl procedure, the return value is the address of the record describing
*	the procedure.  Otherwise the return value is 0.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ Proc *TclIsProc(Command *cmdPtr)
{
	if (cmdPtr->proc == InterpProc) {
		return (Proc *)cmdPtr->clientData;
	}
	return (Proc *)0;
}

/*
*----------------------------------------------------------------------
*
* InterpProc --
*	When a Tcl procedure gets invoked, this routine gets invoked to interpret the procedure.
*
* Results:
*	A standard Tcl result value, usually TCL_OK.
*
* Side effects:
*	Depends on the commands in the procedure.
*
*----------------------------------------------------------------------
*/
static __device__ int InterpProc(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	register Proc *procPtr = (Proc *)clientData;
	register Interp *iPtr = (Interp *)interp;
	int result;

	// Set up a call frame for the new procedure invocation.
	CallFrame frame;
	Tcl_InitHashTable(&frame.varTable, TCL_STRING_KEYS);
	if (iPtr->varFramePtr != NULL) {
		frame.level = iPtr->varFramePtr->level + 1;
	} else {
		frame.level = 1;
	}
	frame.argc = argc;
	frame.args = args;
	frame.callerPtr = iPtr->framePtr;
	frame.callerVarPtr = iPtr->varFramePtr;
	iPtr->framePtr = &frame;
	iPtr->varFramePtr = &frame;

	// Match the actual arguments against the procedure's formal parameters to compute local variables.
	register Arg *argPtr;
	const char **args2;
	for (argPtr = procPtr->argPtr, args2 = args+1, argc -= 1; argPtr != NULL; argPtr = argPtr->nextPtr, args2++, argc--) {
		// Handle the special case of the last formal being "args".  When it occurs, assign it a list consisting of all the remaining actual arguments.
		char *value;
		if (argPtr->nextPtr == NULL && !strcmp(argPtr->name, "args")) {
			if (argc < 0) {
				argc = 0;
			}
			value = Tcl_Merge(argc, args2);
			Tcl_SetVar(interp, argPtr->name, value, 0);
			_freeFast(value);
			argc = 0;
			break;
		} else if (argc > 0) {
			value = (char *)*args2;
		} else if (argPtr->defValue != NULL) {
			value = argPtr->defValue;
		} else {
			Tcl_AppendResult(interp, "no value given for parameter \"", argPtr->name, "\" to \"", args[0], "\"", (char *)NULL);
			result = TCL_ERROR;
			goto procDone;
		}
		Tcl_SetVar(interp, argPtr->name, value, 0);
	}
	if (argc > 0) {
		Tcl_AppendResult(interp, "called \"", args[0], "\" with too many arguments", (char *)NULL);
		result = TCL_ERROR;
		goto procDone;
	}

	// Increment the usage count
	procPtr->uses++;

	// Invoke the commands in the procedure's body.
	char *end;
	result = Tcl_Eval(interp, procPtr->command, 0, &end);

	if (procPtr->uses == 1) {
		// Now we "delete" the proc. This will decrement the reference count, and only delete the proc if the usage count reaches 0.
		// This might happen if the proc was renamed/deleted while it was executing. We can't reference procPtr after this point.
		ProcDeleteProc((ClientData)procPtr);
	} else {
		procPtr->uses--;
	}

	if (result == TCL_RETURN) {
		result = TCL_OK;
	} else if (result == TCL_ERROR) {
		// Record information telling where the error occurred.
		char msg[100];
		sprintf(msg, "\n    (procedure \"%.50s\" line %d)", args[0], iPtr->errorLine);
		Tcl_AddErrorInfo(interp, msg);
	} else if (result == TCL_BREAK) {
		iPtr->result = "invoked \"break\" outside of a loop";
		result = TCL_ERROR;
	} else if (result == TCL_CONTINUE) {
		iPtr->result = "invoked \"continue\" outside of a loop";
		result = TCL_ERROR;
	}

	// Delete the call frame for this procedure invocation (it's important to remove the call frame from the interpreter
	// before deleting it, so that traces invoked during the deletion don't see the partially-deleted frame).
procDone:
	iPtr->framePtr = frame.callerPtr;
	iPtr->varFramePtr = frame.callerVarPtr;
	TclDeleteVars(iPtr, &frame.varTable);
	return result;
}

/*
*----------------------------------------------------------------------
*
* ProcDeleteProc --
*	This procedure is invoked just before a command procedure is removed from an interpreter.  Its job is to release all the
*	resources allocated to the procedure. 'uses' is decremented, but if it doesn't go to zero, it is not deleted.
*
* Results:
*	None.
*
* Side effects:
*	Memory gets freed.
*
*----------------------------------------------------------------------
*/
static __device__ void ProcDeleteProc(ClientData clientData)
{
	register Proc *procPtr = (Proc *)clientData;
	if (--procPtr->uses <= 0) {
		_freeFast((char *)procPtr->command);
		for (register Arg *argPtr = procPtr->argPtr; argPtr != NULL;) {
			Arg *nextPtr = argPtr->nextPtr;
			_freeFast((char *)argPtr);
			argPtr = nextPtr;
		}
		_freeFast((char *)procPtr);
	}
}
