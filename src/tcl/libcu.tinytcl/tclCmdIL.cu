// tclCmdIL.c --
//
//	This file contains the top-level command routines for most of the Tcl built-in commands whose names begin with the letters
//	I through L.  It contains only commands in the generic core (i.e. those that don't depend much upon UNIX facilities).
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

// Forward declarations for procedures defined in this file:
static __device__ int SortCompareProc(const char *first, const char *second);

/*
*----------------------------------------------------------------------
*
* Tcl_IfCmd --
*	This procedure is invoked to process the "if" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_IfCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	int i = 1;
	while (true) {
		// At this point in the loop, args and argc refer to an expression to test, either for the main expression or an expression
		// following an "elseif".  The arguments after the expression must be "then" (optional) and a script to execute if the expression is true.
		if (i >= argc) {
			Tcl_AppendResult(interp, "wrong # args: no expression after \"", args[i - 1], "\" argument", (char *)NULL);
			return TCL_ERROR;
		}
		int value;
		int result = Tcl_ExprBoolean(interp, (char *)args[i], &value);
		if (result != TCL_OK) {
			return result;
		}
		i++;
		if (i < argc && !strcmp(args[i], "then")) {
			i++;
		}
		if (i >= argc) {
			Tcl_AppendResult(interp, "wrong # args: no script following \"", args[i - 1], "\" argument", (char *)NULL);
			return TCL_ERROR;
		}
		if (value) {
			return Tcl_Eval(interp, (char *)args[i], 0, (char **)NULL);
		}

		// The expression evaluated to false.  Skip the command, then see if there is an "else" or "elseif" clause.
		i++;
		if (i >= argc) {
			return TCL_OK;
		}
		if (args[i][0] == 'e' && !strcmp(args[i], "elseif")) {
			i++;
			continue;
		}
		break;
	}

	// Couldn't find a "then" or "elseif" clause to execute.  Check now for an "else" clause.  We know that there's at least one more argument when we get here.
	if (!strcmp(args[i], "else")) {
		i++;
		if (i >= argc) {
			Tcl_AppendResult(interp, "wrong # args: no script following \"else\" argument", (char *)NULL);
			return TCL_ERROR;
		}
	}
	return Tcl_Eval(interp, (char *)args[i], 0, (char **)NULL);
}

/*
*----------------------------------------------------------------------
*
* Tcl_IncrCmd --
*	This procedure is invoked to process the "incr" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_IncrCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2 && argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " varName ?increment?\"", (char *)NULL);
		return TCL_ERROR;
	}
	char *oldString = Tcl_GetVar(interp, (char *)args[1], TCL_LEAVE_ERR_MSG);
	if (oldString == NULL) {
		return TCL_ERROR;
	}
	int value;
	if (Tcl_GetInt(interp, oldString, &value) != TCL_OK) {
		Tcl_AddErrorInfo(interp, (char *)"\n    (reading value of variable to increment)");
		return TCL_ERROR;
	}
	if (argc == 2) {
		value += 1;
	}
	else {
		int increment;
		if (Tcl_GetInt(interp, args[2], &increment) != TCL_OK) {
			Tcl_AddErrorInfo(interp, (char *)"\n    (reading increment)");
			return TCL_ERROR;
		}
		value += increment;
	}
	char newString[30];
	sprintf(newString, "%d", value);
	char *result = Tcl_SetVar(interp, (char *)args[1], newString, TCL_LEAVE_ERR_MSG);
	if (result == NULL) {
		return TCL_ERROR;
	}
	interp->result = result;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_InfoCmd --
*	This procedure is invoked to process the "info" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_InfoCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	register Interp *iPtr = (Interp *)interp;
	int length;
	char c;
	Arg *argPtr;
	Proc *procPtr;
	Var *varPtr;
	Command *cmdPtr;
	Tcl_HashEntry *hPtr;
	Tcl_HashSearch search;

	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " option ?arg arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	c = args[1][0];
	length = strlen(args[1]);
	if (c == 'a' && !strncmp(args[1], (char *)"args", length)) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " args procname\"", (char *)NULL);
			return TCL_ERROR;
		}
		procPtr = TclFindProc(iPtr, (char *)args[2]);
		if (procPtr == NULL) {
		infoNoSuchProc:
			Tcl_AppendResult(interp, "\"", args[2], "\" isn't a procedure", (char *)NULL);
			return TCL_ERROR;
		}
		for (argPtr = procPtr->argPtr; argPtr != NULL; argPtr = argPtr->nextPtr) {
			Tcl_AppendElement(interp, argPtr->name, 0);
		}
		return TCL_OK;
	}
	else if (c == 'b' && !strncmp(args[1], "body", length)) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " body procname\"", (char *)NULL);
			return TCL_ERROR;
		}
		procPtr = TclFindProc(iPtr, (char *)args[2]);
		if (procPtr == NULL) {
			goto infoNoSuchProc;
		}
		iPtr->result = procPtr->command;
		return TCL_OK;
	}
	else if (c == 'c' && !strncmp(args[1], "cmdcount", length) && length >= 2) {
		if (argc != 2) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " cmdcount\"", (char *)NULL);
			return TCL_ERROR;
		}
		sprintf(iPtr->result, "%d", iPtr->cmdCount);
		return TCL_OK;
	}
	else if (c == 'c' && !strncmp(args[1], "commands", length) && length >= 4) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " commands [pattern]\"", (char *)NULL);
			return TCL_ERROR;
		}
		for (hPtr = Tcl_FirstHashEntry(&iPtr->commandTable, &search); hPtr != NULL; hPtr = Tcl_NextHashEntry(&search)) {
			char *name = Tcl_GetHashKey(&iPtr->commandTable, hPtr);
			if (argc == 3 && !Tcl_StringMatch(name, (char *)args[2])) {
				continue;
			}
			Tcl_AppendElement(interp, name, 0);
		}
		return TCL_OK;
	}
	else if (c == 'c' && !strncmp(args[1], "complete", length) && length >= 4) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " complete command\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (Tcl_CommandComplete((char *)args[2])) {
			interp->result = (char *)"1";
		}
		else {
			interp->result = (char *)"0";
		}
		return TCL_OK;
	}
	else if (c == 'd' && !strncmp(args[1], "default", length)) {
		if (argc != 5) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " default procname arg varname\"", (char *)NULL);
			return TCL_ERROR;
		}
		procPtr = TclFindProc(iPtr, (char *)args[2]);
		if (procPtr == NULL) {
			goto infoNoSuchProc;
		}
		for (argPtr = procPtr->argPtr; ; argPtr = argPtr->nextPtr) {
			if (argPtr == NULL) {
				Tcl_AppendResult(interp, "procedure \"", args[2], "\" doesn't have an argument \"", args[3], "\"", (char *)NULL);
				return TCL_ERROR;
			}
			if (strcmp(args[3], argPtr->name) == 0) {
				if (argPtr->defValue != NULL) {
					if (Tcl_SetVar((Tcl_Interp *)iPtr, (char *)args[4], argPtr->defValue, 0) == NULL) {
					defStoreError:
						Tcl_AppendResult(interp, "couldn't store default value in variable \"", args[4], "\"", (char *)NULL);
						return TCL_ERROR;
					}
					iPtr->result = (char *)"1";
				}
				else {
					if (Tcl_SetVar((Tcl_Interp *)iPtr, (char *)args[4], (char *)"", 0) == NULL) {
						goto defStoreError;
					}
					iPtr->result = (char *)"0";
				}
				return TCL_OK;
			}
		}
	}
	else if (c == 'e' && !strncmp(args[1], "exists", length)) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " exists varName\"", (char *)NULL);
			return TCL_ERROR;
		}
		char *p = Tcl_GetVar((Tcl_Interp *)iPtr, (char *)args[2], 0);
		// The code below handles the special case where the name is for an array:  Tcl_GetVar will reject this since you can't read an array variable without an index.
		if (p == NULL) {
			if (strchr(args[2], '(') != NULL) {
			noVar:
				iPtr->result = (char *)"0";
				return TCL_OK;
			}
			Tcl_HashEntry *hPtr;
			if (iPtr->varFramePtr == NULL) {
				hPtr = Tcl_FindHashEntry(&iPtr->globalTable, (char *)args[2]);
			}
			else {
				hPtr = Tcl_FindHashEntry(&iPtr->varFramePtr->varTable, (char *)args[2]);
			}
			if (hPtr == NULL) {
				goto noVar;
			}
			Var *varPtr = (Var *)Tcl_GetHashValue(hPtr);
			if (varPtr->flags & VAR_UPVAR) {
				varPtr = (Var *)Tcl_GetHashValue(varPtr->value.upvarPtr);
			}
			if (!(varPtr->flags & VAR_ARRAY)) {
				goto noVar;
			}
		}
		iPtr->result = (char *)"1";
		return TCL_OK;
	}
	else if (c == 'g' && !strncmp(args[1], "globals", length)) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " globals [pattern]\"", (char *)NULL);
			return TCL_ERROR;
		}
		for (hPtr = Tcl_FirstHashEntry(&iPtr->globalTable, &search); hPtr != NULL; hPtr = Tcl_NextHashEntry(&search)) {
			varPtr = (Var *)Tcl_GetHashValue(hPtr);
			if (varPtr->flags & VAR_UNDEFINED) {
				continue;
			}
			char *name = Tcl_GetHashKey(&iPtr->globalTable, hPtr);
			if (argc == 3 && !Tcl_StringMatch(name, (char *)args[2])) {
				continue;
			}
			Tcl_AppendElement(interp, name, 0);
		}
		return TCL_OK;
	}
#ifdef HAVE_GETHOSTNAME
	else if (c == 'h' && !strncmp(args[1], "hostname", length)) {
		if (argc != 2) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " hostname\"", (char *)NULL);
			return TCL_ERROR;
		}
		static int hostnameInited = 0;
		if (!hostnameInited) {
			static char hostname[128];
			gethostname(hostname, sizeof(hostname));
			hostnameInited = 1;
		}
		iPtr->result = hostname;
		return TCL_OK;
	}
#endif
	else if (c == 'l' && !strncmp(args[1], "level", length) && length >= 2) {
		if (argc == 2) {
			if (iPtr->varFramePtr == NULL) {
				iPtr->result = (char *)"0";
			}
			else {
				sprintf(iPtr->result, "%d", iPtr->varFramePtr->level);
			}
			return TCL_OK;
		}
		else if (argc == 3) {
			int level;
			if (Tcl_GetInt(interp, args[2], &level) != TCL_OK) {
				return TCL_ERROR;
			}
			if (level <= 0) {
				if (iPtr->varFramePtr == NULL) {
				levelError:
					Tcl_AppendResult(interp, "bad level \"", args[2], "\"", (char *)NULL);
					return TCL_ERROR;
				}
				level += iPtr->varFramePtr->level;
			}
			CallFrame *framePtr;
			for (framePtr = iPtr->varFramePtr; framePtr != NULL; framePtr = framePtr->callerVarPtr) {
				if (framePtr->level == level) {
					break;
				}
			}
			if (framePtr == NULL) {
				goto levelError;
			}
			iPtr->result = Tcl_Merge(framePtr->argc, framePtr->args);
			iPtr->freeProc = (Tcl_FreeProc *)free;
			return TCL_OK;
		}
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " level [number]\"", (char *)NULL);
		return TCL_ERROR;
	}
	else if (c == 'l' && !strncmp(args[1], "library", length) && length >= 2) {
		if (argc != 2) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " library\"", (char *)NULL);
			return TCL_ERROR;
		}
		interp->result = getenv("TCL_LIBRARY");
		if (interp->result == NULL) {
#ifdef TCL_LIBRARY
			interp->result = (char *)TCL_LIBRARY;
#else
			interp->result = (char *)"there is no Tcl library at this installation";
			return TCL_ERROR;
#endif
		}
		return TCL_OK;
	}
	else if (c == 'l' && !strncmp(args[1], "locals", length) && length >= 2) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " locals [pattern]\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (iPtr->varFramePtr == NULL) {
			return TCL_OK;
		}
		for (hPtr = Tcl_FirstHashEntry(&iPtr->varFramePtr->varTable, &search); hPtr != NULL; hPtr = Tcl_NextHashEntry(&search)) {
			varPtr = (Var *)Tcl_GetHashValue(hPtr);
			if (varPtr->flags & (VAR_UNDEFINED | VAR_UPVAR)) {
				continue;
			}
			char *name = Tcl_GetHashKey(&iPtr->varFramePtr->varTable, hPtr);
			if (argc == 3 && !Tcl_StringMatch(name, (char *)args[2])) {
				continue;
			}
			Tcl_AppendElement(interp, name, 0);
		}
		return TCL_OK;
	}
	else if (c == 'p' && !strncmp(args[1], "procs", length)) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " procs [pattern]\"", (char *)NULL);
			return TCL_ERROR;
		}
		for (hPtr = Tcl_FirstHashEntry(&iPtr->commandTable, &search); hPtr != NULL; hPtr = Tcl_NextHashEntry(&search)) {
			char *name = Tcl_GetHashKey(&iPtr->commandTable, hPtr);
			cmdPtr = (Command *)Tcl_GetHashValue(hPtr);
			if (!TclIsProc(cmdPtr)) {
				continue;
			}
			if (argc == 3 && !Tcl_StringMatch(name, (char *)args[2])) {
				continue;
			}
			Tcl_AppendElement(interp, name, 0);
		}
		return TCL_OK;
	}
	else if (c == 's' && !strncmp(args[1], "script", length)) {
		if (argc != 2) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " script\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (iPtr->scriptFile != NULL) {
			interp->result = iPtr->scriptFile;
		}
		return TCL_OK;
	}
	else if (c == 't' && !strncmp(args[1], "tclversion", length)) {
		if (argc != 2) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " tclversion\"", (char *)NULL);
			return TCL_ERROR;
		}
		// Note:  TCL_VERSION below is expected to be set with a "-D" switch in the Makefile.
		iPtr->result = (char *)TCL_VERSION;
		return TCL_OK;
	}
	else if (c == 'v' && !strncmp(args[1], "vars", length)) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " vars [pattern]\"", (char *)NULL);
			return TCL_ERROR;
		}
		Tcl_HashTable *tablePtr;
		if (iPtr->varFramePtr == NULL) {
			tablePtr = &iPtr->globalTable;
		}
		else {
			tablePtr = &iPtr->varFramePtr->varTable;
		}
		for (hPtr = Tcl_FirstHashEntry(tablePtr, &search); hPtr != NULL; hPtr = Tcl_NextHashEntry(&search)) {
			varPtr = (Var *)Tcl_GetHashValue(hPtr);
			if (varPtr->flags & VAR_UNDEFINED) {
				continue;
			}
			char *name = Tcl_GetHashKey(tablePtr, hPtr);
			if (argc == 3 && !Tcl_StringMatch(name, (char *)args[2])) {
				continue;
			}
			Tcl_AppendElement(interp, name, 0);
		}
		return TCL_OK;
	}
	else {
		Tcl_AppendResult(interp, "bad option \"", args[1], "\": should be args, body, cmdcount, commands, ", "complete, default, ", "exists, globals, level, library, locals, procs, ", "script, tclversion, or vars", (char *)NULL);
		return TCL_ERROR;
	}
}

/*
*----------------------------------------------------------------------
*
* Tcl_JoinCmd --
*	This procedure is invoked to process the "join" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_JoinCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	const char *joinString;
	if (argc == 2) {
		joinString = " ";
	}
	else if (argc == 3) {
		joinString = args[2];
	}
	else {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list ?joinString?\"", (char *)NULL);
		return TCL_ERROR;
	}
	int listArgc;
	const char **listArgs;
	if (Tcl_SplitList(interp, (char *)args[1], &listArgc, &listArgs) != TCL_OK) {
		return TCL_ERROR;
	}
	for (int i = 0; i < listArgc; i++) {
		if (i == 0) {
			Tcl_AppendResult(interp, listArgs[0], (char *)NULL);
		}
		else {
			Tcl_AppendResult(interp, joinString, listArgs[i], (char *)NULL);
		}
	}
	_freeFast((char *)listArgs);
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LindexCmd --
*	This procedure is invoked to process the "lindex" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_LindexCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	char *p, *element;
	int index, result;
	if (argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list index\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (!strcmp(args[2], "end")) {
		// Find the length of the list
		for (index = 0, p = (char *)args[1]; *p != 0; index++) {
			result = TclFindElement(interp, p, &element, &p, (int *)NULL, (int *)NULL);
			if (result != TCL_OK) {
				return result;
			}
			if (*element == 0) {
				break;
			}
		}
		Tcl_ResetResult(interp);
		// and subtract 1
		index--;
	}
	else {
		if (Tcl_GetInt(interp, args[2], &index) != TCL_OK) {
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "expected integer or \"end\" but got \"", args[2], "\"", (char *)NULL);
			return TCL_ERROR;
		}
	}
	if (index < 0) {
		return TCL_OK;
	}

	int size, parenthesized;
	for (p = (char *)args[1]; index >= 0; index--) {
		result = TclFindElement(interp, p, &element, &p, &size, &parenthesized);
		if (result != TCL_OK) {
			return result;
		}
	}
	if (size == 0) {
		return TCL_OK;
	}
	if (size >= TCL_RESULT_SIZE) {
		interp->result = (char *)_allocFast((unsigned)size + 1);
		interp->freeProc = (Tcl_FreeProc *)free;
	}
	if (parenthesized) {
		memcpy(interp->result, element, size);
		interp->result[size] = 0;
	}
	else {
		TclCopyAndCollapse(size, element, interp->result);
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LinsertCmd --
*	This procedure is invoked to process the "linsert" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_LinsertCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc < 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list index element ?element ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	int index;
	if (Tcl_GetInt(interp, args[2], &index) != TCL_OK) {
		return TCL_ERROR;
	}

	// Skip over the first "index" elements of the list, then add all of those elements to the result.
	int size = 0;
	char *element = (char *)args[1];
	char *p;
	int count;
	for (count = 0, p = (char *)args[1]; count < index && *p != 0; count++) {
		int result = TclFindElement(interp, p, &element, &p, &size, (int *)NULL);
		if (result != TCL_OK) {
			return result;
		}
	}
	if (*p == 0) {
		Tcl_AppendResult(interp, args[1], (char *)NULL);
	}
	else {
		char *end = element + size;
		if (element != args[1]) {
			while (*end != 0 && !isspace(*end)) {
				end++;
			}
		}
		char savedChar = *end;
		*end = 0;
		Tcl_AppendResult(interp, args[1], (char *)NULL);
		*end = savedChar;
	}

	// Add the new list elements.
	for (int i = 3; i < argc; i++) {
		Tcl_AppendElement(interp, args[i], 0);
	}

	// Append the remainder of the original list.
	if (*p != 0) {
		Tcl_AppendResult(interp, " ", p, (char *)NULL);
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ListCmd --
*	This procedure is invoked to process the "list" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ListCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc >= 2) {
		interp->result = Tcl_Merge(argc - 1, args + 1);
		interp->freeProc = (Tcl_FreeProc *)free;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LlengthCmd --
*	This procedure is invoked to process the "llength" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_LlengthCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list\"", (char *)NULL);
		return TCL_ERROR;
	}
	int count;
	char *p;
	for (count = 0, p = (char *)args[1]; *p != 0; count++) {
		char *element;
		int result = TclFindElement(interp, p, &element, &p, (int *)NULL, (int *)NULL);
		if (result != TCL_OK) {
			return result;
		}
		if (*element == 0) {
			break;
		}
	}
	sprintf(interp->result, "%d", count);
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LrangeCmd --
*	This procedure is invoked to process the "lrange" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_LrangeCmd(ClientData notUsed, Tcl_Interp *interp, int argc, const char *args[]) {
	int first, last, result;
	char *begin, *end, c, *dummy;
	if (argc != 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list first last\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (Tcl_GetInt(interp, args[2], &first) != TCL_OK) {
		return TCL_ERROR;
	}
	if (first < 0) {
		first = 0;
	}
	if (*args[3] == 'e' && !strncmp(args[3], "end", strlen(args[3]))) {
		last = 1000000;
	}
	else {
		if (Tcl_GetInt(interp, args[3], &last) != TCL_OK) {
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "expected integer or \"end\" but got \"", args[3], "\"", (char *)NULL);
			return TCL_ERROR;
		}
	}
	if (first > last) {
		return TCL_OK;
	}

	// Extract a range of fields.
	int count;
	for (count = 0, begin = (char *)args[1]; count < first; count++) {
		result = TclFindElement(interp, begin, &dummy, &begin, (int *)NULL, (int *)NULL);
		if (result != TCL_OK) {
			return result;
		}
		if (*begin == 0) {
			break;
		}
	}
	for (count = first, end = begin; count <= last && *end != 0; count++) {
		result = TclFindElement(interp, end, &dummy, &end, (int *)NULL, (int *)NULL);
		if (result != TCL_OK) {
			return result;
		}
	}

	// Chop off trailing spaces.
	while (isspace(end[-1])) {
		end--;
	}
	c = *end;
	*end = 0;
	Tcl_SetResult(interp, begin, TCL_VOLATILE);
	*end = c;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LreplaceCmd --
*	This procedure is invoked to process the "lreplace" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_LreplaceCmd(ClientData notUsed, Tcl_Interp *interp, int argc, const char *args[]) {
	char *p1, *p2, *dummy;
	int i, first, last, count, result;
	if (argc < 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list first last ?element element ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (Tcl_GetInt(interp, args[2], &first) != TCL_OK) {
		return TCL_ERROR;
	}
	if (TclGetListIndex(interp, (char *)args[3], &last) != TCL_OK) {
		return TCL_ERROR;
	}
	if (first < 0) {
		first = 0;
	}
	if (last < 0) {
		last = 0;
	}
	if (first > last) {
		Tcl_AppendResult(interp, "first index must not be greater than second", (char *)NULL);
		return TCL_ERROR;
	}

	// Skip over the elements of the list before "first".
	int size = 0;
	char *element = (char *)args[1];
	for (count = 0, p1 = (char *)args[1]; count < first && *p1 != 0; count++) {
		result = TclFindElement(interp, p1, &element, &p1, &size, (int *)NULL);
		if (result != TCL_OK) {
			return result;
		}
	}
	if (*p1 == 0) {
		Tcl_AppendResult(interp, "list doesn't contain element ", args[2], (char *)NULL);
		return TCL_ERROR;
	}

	// Skip over the elements of the list up through "last".
	for (p2 = p1; count <= last && *p2 != 0; count++) {
		result = TclFindElement(interp, p2, &dummy, &p2, (int *)NULL, (int *)NULL);
		if (result != TCL_OK) {
			return result;
		}
	}

	// Add the elements before "first" to the result.  Be sure to include quote or brace characters that might terminate the last of these elements.
	p1 = element + size;
	if (element != args[1]) {
		while (*p1 != 0 && !isspace(*p1)) {
			p1++;
		}
	}
	char savedChar = *p1;
	*p1 = 0;
	Tcl_AppendResult(interp, args[1], (char *)NULL);
	*p1 = savedChar;

	// Add the new list elements.
	for (i = 4; i < argc; i++) {
		Tcl_AppendElement(interp, args[i], 0);
	}

	// Append the remainder of the original list.
	if (*p2 != 0) {
		if (*interp->result == 0) {
			Tcl_SetResult(interp, p2, TCL_VOLATILE);
		}
		else {
			Tcl_AppendResult(interp, " ", p2, (char *)NULL);
		}
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LsearchCmd --
*	This procedure is invoked to process the "lsearch" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_LsearchCmd(ClientData notUsed, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " list pattern\"", (char *)NULL);
		return TCL_ERROR;
	}
	int listArgc;
	const char **listArgs;
	if (Tcl_SplitList(interp, (char *)args[1], &listArgc, &listArgs) != TCL_OK) {
		return TCL_ERROR;
	}
	int match = -1;
	for (int i = 0; i < listArgc; i++) {
		if (Tcl_StringMatch((char *)listArgs[i], (char *)args[2])) {
			match = i;
			break;
		}
	}
	sprintf(interp->result, "%d", match);
	_freeFast((char *)listArgs);
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_LsortCmd --
*	This procedure is invoked to process the "lsort" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/

// The procedure below is called back by qsort to determine the proper ordering between two elements.
static __device__ int SortCompareProc(const char *first, const char *second) {
	return strcmp(*((char **)first), *((char **)second));
}

static __device__ int IntegerSortCompareProc(const char *first, const char *second) {
	int firstint = atoi(*((char **)first));
	int secondint = atoi(*((char **)second));
	return (firstint < secondint ? -1 : (firstint == secondint ? 0 : 1));
}

// Why doesn't qsort allow a user arg!!!
static __device__ char *_sort_command = 0;
static __device__ int _sort_result = TCL_OK;
static __device__ Tcl_Interp *_sort_interp = 0;

static __device__ int CommandSortCompareProc(const char *first, const char *second) {
	// We have already had an error and we need to return something, so fallback to strcmp
	if (_sort_result != TCL_OK) {
		return strcmp(*((char **)first), *((char **)second));
	}
	const char *cmdargs[4];
	cmdargs[0] = _sort_command;
	cmdargs[1] = *((char **)first);
	cmdargs[2] = *((char **)second);
	cmdargs[3] = *((char **)second);
	char *compare_cmd = Tcl_Merge(3, cmdargs);
	_sort_result = Tcl_Eval(_sort_interp, compare_cmd, 0, 0);
	_freeFast(compare_cmd);
	if (_sort_result != TCL_OK) {
		// We need to return something, so fallback to strcmp
		return strcmp(cmdargs[1], cmdargs[2]);
	}
	return atoi(_sort_interp->result);
}

__device__ int Tcl_LsortCmd(ClientData notUsed, Tcl_Interp *interp, int argc, const char *args[]) {
	typedef int (compare_function_type)(const void *, const void *);
	compare_function_type *compare = (compare_function_type *)SortCompareProc;
	_sort_result = TCL_OK;
	while (argc > 2) {
		argc--;
		args++;
		if (!strcmp(args[0], "-integer")) {
			compare = (compare_function_type *)IntegerSortCompareProc;
			break;
		}
		if (!strcmp(args[0], "-command")) {
			compare = (compare_function_type *)CommandSortCompareProc;
			_sort_command = (char *)args[1];
			_sort_interp = interp;
			argc--;
			args++;
			break;
		}
	}
	char *cmd = (char *)args[0];
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", cmd, " ?-integer|-command cmd? list\"", (char *)NULL);
		return TCL_ERROR;
	}
	int listArgc;
	const char **listArgs;
	if (Tcl_SplitList(interp, (char *)args[1], &listArgc, &listArgs) != TCL_OK) {
		return TCL_ERROR;
	}
	qsort(listArgs, listArgc, sizeof(char *), compare);
	if (_sort_result != TCL_OK) {
		return _sort_result;
	}
	interp->result = Tcl_Merge(listArgc, listArgs);
	interp->freeProc = (Tcl_FreeProc *)free;
	_freeFast((char *)listArgs);
	return TCL_OK;
}
