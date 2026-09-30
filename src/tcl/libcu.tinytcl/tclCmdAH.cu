// tclCmdAH.c --
//
//	This file contains the top-level command routines for most of the Tcl built-in commands whose names begin with the letters A to H.
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

/*
*----------------------------------------------------------------------
*
* Tcl_BreakCmd --
*	This procedure is invoked to process the "break" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_BreakCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_BREAK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_CaseCmd --
*	This procedure is invoked to process the "case" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_CaseCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	int i, result;
	if (argc < 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " string ?in? patList body ... ?default body?\"", (char *)NULL);
		return TCL_ERROR;
	}
	char *string = (char *)args[1];
	int body = -1;
	if (!strcmp(args[2], "in")) {
		i = 3;
	}
	else {
		i = 2;
	}
	int caseArgc = argc - i;
	const char **caseArgs = args + i;

	// If all of the pattern/command pairs are lumped into a single argument, split them out again.
	int splitArgs = 0;
	if (caseArgc == 1) {
		result = Tcl_SplitList(interp, (char *)caseArgs[0], &caseArgc, &caseArgs);
		if (result != TCL_OK) {
			return result;
		}
		splitArgs = 1;
	}

	for (i = 0; i < caseArgc; i += 2) {
		int patArgc, j;
		const char **patArgs;

		if (i == (caseArgc - 1)) {
			interp->result = (char *)"extra case pattern with no body";
			result = TCL_ERROR;
			goto cleanup;
		}

		// Check for special case of single pattern (no list) with no backslash sequences.
		register char *p;
		for (p = (char *)caseArgs[i]; *p != 0; p++) {
			if (isspace(*p) || *p == '\\') {
				break;
			}
		}
		if (*p == 0) {
			if (*caseArgs[i] == 'd' && !strcmp(caseArgs[i], "default")) {
				body = i + 1;
			}
			if (Tcl_StringMatch(string, (char *)caseArgs[i])) {
				body = i + 1;
				goto match;
			}
			continue;
		}

		// Break up pattern lists, then check each of the patterns in the list.
		result = Tcl_SplitList(interp, (char *)caseArgs[i], &patArgc, &patArgs);
		if (result != TCL_OK) {
			goto cleanup;
		}
		for (j = 0; j < patArgc; j++) {
			if (Tcl_StringMatch(string, (char *)patArgs[j])) {
				body = i + 1;
				break;
			}
		}
		_freeFast((char *)patArgs);
		if (j < patArgc) {
			break;
		}
	}

match:
	if (body != -1) {
		result = Tcl_Eval(interp, (char *)caseArgs[body], 0, (char **)NULL);
		if (result == TCL_ERROR) {
			char msg[100];
			sprintf(msg, "\n    (\"%.50s\" arm line %d)", caseArgs[body - 1], interp->errorLine);
			Tcl_AddErrorInfo(interp, msg);
		}
		goto cleanup;
	}

	// Nothing matched:  return nothing.
	result = TCL_OK;

cleanup:
	if (splitArgs) {
		_freeFast((char *)caseArgs);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_CatchCmd --
*	This procedure is invoked to process the "catch" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_CatchCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2 && argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " command ?varName?\"", (char *)NULL);
		return TCL_ERROR;
	}
	int result = Tcl_Eval(interp, (char *)args[1], TCL_CATCH_SIGNAL, (char **)NULL);
	if (argc == 3) {
		if (Tcl_SetVar(interp, (char *)args[2], interp->result, 0) == NULL) {
			Tcl_SetResult(interp, (char *)"couldn't save command result in variable", TCL_STATIC);
			return TCL_ERROR;
		}
	}
	Tcl_ResetResult(interp);
	sprintf(interp->result, "%d", result);
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ConcatCmd --
*	This procedure is invoked to process the "concat" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ConcatCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " arg ?arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (argc >= 2) {
		interp->result = Tcl_Concat(argc - 1, args + 1);
		interp->freeProc = (Tcl_FreeProc *)free;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ContinueCmd --
*	This procedure is invoked to process the "continue" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ContinueCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_CONTINUE;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ErrorCmd --
*	This procedure is invoked to process the "error" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ErrorCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	Interp *iPtr = (Interp *)interp;
	if (argc < 2 || argc > 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " message ?errorInfo? ?errorCode?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (argc >= 3 && args[2][0] != 0) {
		Tcl_AddErrorInfo(interp, (char *)args[2]);
		iPtr->flags |= ERR_ALREADY_LOGGED;
	}
	if (argc == 4) {
		Tcl_SetVar2(interp, (char *)"errorCode", (char *)NULL, (char *)args[3], TCLGLOBAL__ONLY);
		iPtr->flags |= ERROR_CODE_SET;
	}
	Tcl_SetResult(interp, (char *)args[1], TCL_VOLATILE);
	return TCL_ERROR;
}

/*
*----------------------------------------------------------------------
*
* Tcl_EvalCmd --
*	This procedure is invoked to process the "eval" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_EvalCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " arg ?arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	int result;
	if (argc == 2) {
		result = Tcl_Eval(interp, (char *)args[1], 0, (char **)NULL);
	}
	else {
		// More than one argument:  concatenate them together with spaces between, then evaluate the result.
		char *cmd = Tcl_Concat(argc - 1, args + 1);
		result = Tcl_Eval(interp, cmd, 0, (char **)NULL);
		_freeFast(cmd);
	}
	if (result == TCL_ERROR) {
		char msg[60];
		sprintf(msg, "\n    (\"eval\" body line %d)", interp->errorLine);
		Tcl_AddErrorInfo(interp, msg);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ExprCmd --
*	This procedure is invoked to process the "expr" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ExprCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " arg ?arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (argc == 2) {
		return Tcl_ExprString(interp, (char *)args[1]);
	}
	else {
		char *buf = Tcl_Concat(argc - 1, args + 1);
		int result = Tcl_ExprString(interp, buf);
		_freeFast(buf);
		return result;
	}
}

/*
*----------------------------------------------------------------------
*
* Tcl_ForCmd --
*	This procedure is invoked to process the "for" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ForCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 5) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " start test next command\"", (char *)NULL);
		return TCL_ERROR;
	}
	int result = Tcl_Eval(interp, (char *)args[1], 0, (char **)NULL);
	if (result != TCL_OK) {
		if (result == TCL_ERROR) {
			Tcl_AddErrorInfo(interp, (char *)"\n    (\"for\" initial command)");
		}
		return result;
	}
	while (true) {
		int value;
		result = Tcl_ExprBoolean(interp, (char *)args[2], &value);
		if (result != TCL_OK) {
			return result;
		}
		if (!value) {
			break;
		}
		result = Tcl_Eval(interp, (char *)args[4], 0, (char **)NULL);
		if (result != TCL_OK && result != TCL_CONTINUE) {
			if (result == TCL_ERROR) {
				char msg[60];
				sprintf(msg, "\n    (\"for\" body line %d)", interp->errorLine);
				Tcl_AddErrorInfo(interp, msg);
			}
			break;
		}
		result = Tcl_Eval(interp, (char *)args[3], 0, (char **)NULL);
		if (result == TCL_BREAK) {
			break;
		}
		else if (result != TCL_OK) {
			if (result == TCL_ERROR) {
				Tcl_AddErrorInfo(interp, (char *)"\n    (\"for\" loop-end command)");
			}
			return result;
		}
	}
	if (result == TCL_BREAK) {
		result = TCL_OK;
	}
	if (result == TCL_OK) {
		Tcl_ResetResult(interp);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ForeachCmd --
*	This procedure is invoked to process the "foreach" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_ForeachCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " varName list command\"", (char *)NULL);
		return TCL_ERROR;
	}
	// Break the list up into elements, and execute the command once for each value of the element.
	int listArgc;
	const char **listArgs;
	int result = Tcl_SplitList(interp, (char *)args[2], &listArgc, &listArgs);
	if (result != TCL_OK) {
		return result;
	}
	for (int i = 0; i < listArgc; i++) {
		if (Tcl_SetVar(interp, (char *)args[1], (char *)listArgs[i], 0) == NULL) {
			Tcl_SetResult(interp, (char *)"couldn't set loop variable", TCL_STATIC);
			result = TCL_ERROR;
			break;
		}
		result = Tcl_Eval(interp, (char *)args[3], 0, (char **)NULL);
		if (result != TCL_OK) {
			if (result == TCL_CONTINUE) {
				result = TCL_OK;
			}
			else if (result == TCL_BREAK) {
				result = TCL_OK;
				break;
			}
			else if (result == TCL_ERROR) {
				char msg[100];
				sprintf(msg, "\n    (\"foreach\" body line %d)", interp->errorLine);
				Tcl_AddErrorInfo(interp, msg);
				break;
			}
			else {
				break;
			}
		}
	}
	_freeFast((char *)listArgs);
	if (result == TCL_OK) {
		Tcl_ResetResult(interp);
	}
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_FormatCmd --
*	This procedure is invoked to process the "format" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_FormatCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	register char *format;	// Used to read characters from the format string.
	char newFormat[40];		// A new format specifier is generated here.
	int width;			// Field width from field specifier, or 0 if no width given.
	int precision;		// Field precision from field specifier, or 0 if no precision given.
	int size;			// Number of bytes needed for result of conversion, based on type of conversion ("e", "s", etc.) and width from above.
	char *oneWordValue = NULL;	// Used to hold value to pass to sprintf, if it's a one-word value.
	int intValue;		// value is an int
	double twoWordValue;	// Used to hold value to pass to sprintf if it's a two-word value.
	int useTwoWords;		// 0 means use oneWordValue, 1 means use twoWordValue.
	char *dst = interp->result;	// Where result is stored.  Starts off at interp->resultSpace, but may get dynamically re-allocated if this isn't enough.
	int dstSize = 0;		// Number of non-null characters currently stored at dst.
	int dstSpace = TCL_RESULT_SIZE;
	// Total amount of storage space available in dst (not including null terminator.
	int noPercent;		// Special case for speed:  indicates there's no field specifier, just a string to copy.
	int valSize;		// size of value if num (short/int/long).

	// This procedure is a bit nasty.  The goal is to use sprintf to do most of the dirty work.  There are several problems:
	// 1. this procedure can't trust its arguments.
	// 2. we must be able to provide a large enough result area to hold whatever's generated.  This is hard to estimate.
	// 2. there's no way to move the arguments from args to the call to sprintf in a reasonable way.  This is particularly nasty because some of the arguments may be two-word values (doubles).
	// So, what happens here is to scan the format string one % group at a time, making many individual calls to sprintf.
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " formatString ?arg arg ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	const char **curArg = args + 2; // Remainder of args array.
	argc -= 2;
	for (format = (char *)args[1]; *format != 0; ) {
		register char *newPtr = newFormat;
		width = precision = useTwoWords = noPercent = valSize = 0;

		// Get rid of any characters before the next field specifier. Collapse backslash sequences found along the way.
		if (*format != '%') {
			register char *p;
			oneWordValue = p = format;
			while (*format != '%' && *format != 0) {
				if (*format == '\\') {
					int bsSize;
					*p = Tcl_Backslash(format, &bsSize);
					if (*p != 0) {
						p++;
					}
					format += bsSize;
				}
				else {
					*p = *format;
					p++;
					format++;
				}
			}
			size = (int)(p - oneWordValue);
			noPercent = 1;
			goto doField;
		}
		if (format[1] == '%') {
			oneWordValue = format;
			size = 1;
			noPercent = 1;
			format += 2;
			goto doField;
		}

		// Parse off a field specifier, compute how many characters will be needed to store the result, and substitute for "*" size specifiers.
		*newPtr = '%';
		newPtr++;
		format++;
		while (*format == '-' || *format == '#' || *format == '0' || *format == ' ' || *format == '+') {
			*newPtr = *format;
			newPtr++;
			format++;
		}
		if (isdigit(*format)) {
			width = atoi(format);
			do {
				format++;
			} while (isdigit(*format));
		}
		else if (*format == '*') {
			if (argc <= 0) {
				goto notEnoughArgs;
			}
			if (Tcl_GetInt(interp, *curArg, &width) != TCL_OK) {
				goto fmtError;
			}
			argc--;
			curArg++;
			format++;
		}
		if (width != 0) {
			sprintf(newPtr, "%d", width);
			while (*newPtr != 0) {
				newPtr++;
			}
		}
		if (*format == '.') {
			*newPtr = '.';
			newPtr++;
			format++;
		}
		if (isdigit(*format)) {
			precision = atoi(format);
			do {
				format++;
			} while (isdigit(*format));
		}
		else if (*format == '*') {
			if (argc <= 0) {
				goto notEnoughArgs;
			}
			if (Tcl_GetInt(interp, *curArg, &precision) != TCL_OK) {
				goto fmtError;
			}
			argc--;
			curArg++;
			format++;
		}
		if (precision != 0) {
			sprintf(newPtr, "%d", precision);
			while (*newPtr != 0) {
				newPtr++;
			}
		}
		if (*format == 'l') {
			valSize = sizeof(long);
			format++;
		}
		else if (*format == 'h') {
			valSize = sizeof(short);
			*newPtr = 'h';
			newPtr++;
			format++;
		}
		*newPtr = *format;
		newPtr++;
		*newPtr = 0;
		if (argc <= 0) {
			goto notEnoughArgs;
		}
		switch (*format) {
		case 'D':
		case 'O':
		case 'U':
			if (valSize != sizeof(int)) {
				newPtr++;
			}
			else {
				valSize = sizeof(int);
			}
			newPtr[-1] = _tolower(*format);
			newPtr[-2] = 'l';
			*newPtr = 0;
		case 'd':
		case 'o':
		case 'u':
		case 'x':
		case 'X':
			if (Tcl_GetInt(interp, *curArg, &intValue) != TCL_OK) {
				goto fmtError;
			}
			size = 40;
			if (valSize == 0)
				valSize = sizeof(int);
			break;
		case 's':
			oneWordValue = (char *)*curArg;
			size = strlen(*curArg);
			valSize = 0;
			break;
		case 'c':
			if (Tcl_GetInt(interp, *curArg, &intValue) != TCL_OK) {
				goto fmtError;
			}
			size = 1;
			if (valSize == 0)
				valSize = sizeof(int);
			break;
		case 'F':
			newPtr[-1] = _tolower(newPtr[-1]);
		case 'e':
		case 'E':
		case 'f':
		case 'g':
		case 'G':
			if (Tcl_GetDouble(interp, *curArg, &twoWordValue) != TCL_OK) {
				goto fmtError;
			}
			useTwoWords = 1;
			size = 320;
			if (precision > 10) {
				size += precision;
			}
			break;
		case 0:
			interp->result = (char *)"format string ended in middle of field specifier";
			goto fmtError;
		default:
			sprintf(interp->result, "bad field specifier \"%c\"", *format);
			goto fmtError;
		}
		argc--;
		curArg++;
		format++;

		// Make sure that there's enough space to hold the formatted result, then format it.
	doField:
		if (width > size) {
			size = width;
		}
		if ((dstSize + size) > dstSpace) {
			int newSpace = 2 * (dstSize + size);
			char *newDst = (char *)_allocFast((unsigned)newSpace + 1);
			if (dstSize != 0) {
				memcpy(newDst, dst, dstSize);
			}
			if (dstSpace != TCL_RESULT_SIZE) {
				_freeFast(dst);
			}
			dst = newDst;
			dstSpace = newSpace;
		}
		if (noPercent) {
			memcpy((dst + dstSize), oneWordValue, size);
			dstSize += size;
			dst[dstSize] = 0;
		}
		else {
			if (useTwoWords) {
				sprintf(dst + dstSize, newFormat, twoWordValue);
			}
			else if (valSize == sizeof(short)) {
				// The double cast below is needed for a few machines (e.g. Pyramids as of 1/93) that don't like casts directly from pointers to shorts.
				sprintf(dst + dstSize, newFormat, (short)intValue);
			}
			else if (valSize == sizeof(int)) {
				sprintf(dst + dstSize, newFormat, intValue);
			}
			else if (valSize == sizeof(long)) {
				sprintf(dst + dstSize, newFormat, (long)intValue);
			}
			else if (valSize != 0) {
				sprintf(dst + dstSize, newFormat, (char *)(long)intValue);
			}
			else {
				sprintf(dst + dstSize, newFormat, (char *)oneWordValue);
			}
			dstSize += strlen(dst + dstSize);
		}
	}

	interp->result = dst;
	if (dstSpace != TCL_RESULT_SIZE) {
		interp->freeProc = (Tcl_FreeProc *)free;
	}
	else {
		interp->freeProc = 0;
	}
	return TCL_OK;

notEnoughArgs:
	interp->result = (char *)"not enough arguments for all format specifiers";
fmtError:
	if (dstSpace != TCL_RESULT_SIZE) {
		_freeFast(dst);
	}
	return TCL_ERROR;
}
