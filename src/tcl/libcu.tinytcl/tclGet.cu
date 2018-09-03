// tclGet.c --
//
//	This file contains procedures to convert strings into other forms, like integers or floating-point numbers or
//	booleans, doing syntax checking along the way.
//
// Copyright 1990-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

__device__ int Tcl_GetIndex(Tcl_Interp *interp, const char *string, const char *table[], char *msg, int flags, int *indexPtr, bool insensitive)
{
	panic("Not Implemented");
	return TCL_OK;
}

__device__ int Tcl_GetIndex2(Tcl_Interp *interp, const char *string, const void *structTable[], int offset, char *msg, int flags, int *indexPtr, bool insensitive)
{
	panic("Not Implemented");
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetInt --
*	Given a string, produce the corresponding integer value.
*
* Results:
*	The return value is normally TCL_OK;  in this case *intPtr will be set to the integer value equivalent to string.  If
*	string is improperly formed then TCL_ERROR is returned and an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_GetInt(Tcl_Interp *interp, const char *string, int *intPtr)
{
	char *end;
	long i = strtol(string, &end, 0);
	while (*end != '\0' && isspace(*end)) {
		end++;
	}
	if (end == string || *end != 0) {
		Tcl_AppendResult(interp, "expected integer but got \"", string, "\"", (char *)NULL);
		return TCL_ERROR;
	}
	*intPtr = (int)i;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetWideInt --
*	Given a string, produce the corresponding long integer value.
*
* Results:
*	The return value is normally TCL_OK;  in this case *intPtr will be set to the long integer value equivalent to string.  If
*	string is improperly formed then TCL_ERROR is returned and an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_GetWideInt(Tcl_Interp *interp, const char *string, int64_t *intPtr)
{
	char *end;
	int64_t i = strtoll(string, &end, 0);
	while (*end != '\0' && isspace(*end)) {
		end++;
	}
	if (end == string || *end != 0) {
		Tcl_AppendResult(interp, "expected integer but got \"", string, "\"", (char *)NULL);
		return TCL_ERROR;
	}
	*intPtr = i;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetDouble --
*	Given a string, produce the corresponding double-precision floating-point value.
*
* Results:
*	The return value is normally TCL_OK;  in this case *doublePtr will be set to the double-precision value equivalent to string.
*	If string is improperly formed then TCL_ERROR is returned and an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_GetDouble(Tcl_Interp *interp, const char *string, double *doublePtr)
{
	char *end;
	double d = strtod(string, &end);
	while (*end != '\0' && isspace(*end)) {
		end++;
	}
	if (end == string || *end != 0) {
		Tcl_AppendResult(interp, "expected floating-point number but got \"", string, "\"", (char *)NULL);
		return TCL_ERROR;
	}
	*doublePtr = d;
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetBoolean --
*	Given a string, return a 0/1 boolean value corresponding to the string.
*
* Results:
*	The return value is normally TCL_OK;  in this case *boolPtr will be set to the 0/1 value equivalent to string.  If
*	string is improperly formed then TCL_ERROR is returned and an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_GetBoolean(Tcl_Interp *interp, const char *string, bool *boolPtr)
{
	char c, lowerCase[10]; // Convert the input string to all lower-case.
	int i;
	for (i = 0; i < 9; i++) {
		c = string[i];
		if (c == 0) {
			break;
		}
		if (c >= 'A' && c <= 'Z') {
			c += 'a' - 'A';
		}
		lowerCase[i] = c;
	}
	lowerCase[i] = 0;

	int length = strlen(lowerCase);
	c = lowerCase[0];
	if (c == '0' && lowerCase[1] == '\0') {
		*boolPtr = false;
	} else if (c == '1' && lowerCase[1] == '\0') {
		*boolPtr = true;
	} else if (c == 'y' && !strncmp(lowerCase, "yes", length)) {
		*boolPtr = true;
	} else if (c == 'n' && !strncmp(lowerCase, "no", length)) {
		*boolPtr = false;
	} else if (c == 't' && !strncmp(lowerCase, "true", length)) {
		*boolPtr = true;
	} else if (c == 'f' && !strncmp(lowerCase, "false", length)) {
		*boolPtr = false;
	} else if (c == 'o' && length >= 2) {
		if (!strncmp(lowerCase, "on", length)) {
			*boolPtr = true;
		} else if (!strncmp(lowerCase, "off", length)) {
			*boolPtr = false;
		}
	} else {
		Tcl_AppendResult(interp, "expected boolean value but got \"", string, "\"", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_GetByteArray --
*	Given a string, return a byte array corresponding to the string.
*
* Results:
*	The return value is normally TCL_OK;  in this case *arrayLength will be set to the length and byte array will be returned.  If
*	string is improperly formed then TCL_ERROR is returned and an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_GetByteArray(Tcl_Interp *interp, const char *string, int *arrayLength)
{
	return TCL_OK;
}
