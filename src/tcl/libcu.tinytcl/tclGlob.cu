// tclGlob.c --
//
//	This file provides procedures and commands for file name manipulation, such as tilde expansion and globbing.
//
// Copyright 1990-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"
#include "tclGpu.h"

// The structure below is used to keep track of a globbing result being built up (i.e. a partial list of file names).  The list grows dynamically to be as big as needed.
typedef struct {
	char *result;		// Pointer to result area.
	int totalSpace;		// Total number of characters allocated for result.
	int spaceUsed;		// Number of characters currently in use to hold the partial result (not including the terminating NULL). */
	int dynamic;		// 0 means result is static space, 1 means it's dynamic.
} GlobResult;

// Declarations for procedures local to this file:
static __device__ void AppendResult(Tcl_Interp *interp, char *dir, char *separator, char *name, int nameLength);
static __device__ int DoGlob(Tcl_Interp *interp, char *dir, char *rem);

/*
*----------------------------------------------------------------------
*
* AppendResult --
*	Given two parts of a file name (directory and element within directory), concatenate the two together and append them to
*	the result building up in interp.
*
* Results:
*	There is no return value.
*
* Side effects:
*	Interp->result gets extended.
*
*----------------------------------------------------------------------
*/
static __device__ void AppendResult(Tcl_Interp *interp, char *dir, char *separator, char *name, int nameLength) {
	// Next, see if we can put together a valid list element from dir and name by calling Tcl_AppendResult.
	int dirFlags;
	if (*dir == 0) {
		dirFlags = 0;
	}
	else {
		Tcl_ScanElement(dir, &dirFlags);
	}
	char saved = name[nameLength];
	name[nameLength] = 0;
	int nameFlags;
	Tcl_ScanElement(name, &nameFlags);
	if (!dirFlags && !nameFlags) {
		if (*interp->result != 0) {
			Tcl_AppendResult(interp, " ", dir, separator, name, (char *)NULL);
		}
		else {
			Tcl_AppendResult(interp, dir, separator, name, (char *)NULL);
		}
		name[nameLength] = saved;
		return;
	}

	// This name has weird characters in it, so we have to convert it to a list element.  To do that, we have to merge the characters
	// into a single name.  To do that, malloc a buffer to hold everything.
	char *p = (char *)_allocFast((unsigned)(strlen(dir) + strlen(separator) + nameLength + 1));
	sprintf(p, "%s%s%s", dir, separator, name);
	name[nameLength] = saved;
	Tcl_AppendElement(interp, p, 0);
	_freeFast(p);
}

/*
*----------------------------------------------------------------------
*
* DoGlob --
*	This recursive procedure forms the heart of the globbing code.  It performs a depth-first traversal of the tree
*	given by the path name to be globbed.
*
* Results:
*	The return value is a standard Tcl result indicating whether an error occurred in globbing.  After a normal return the
*	result in interp will be set to hold all of the file names given by the dir and rem arguments.  After an error the
*	result in interp will hold an error message.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
#undef STATIC_SIZE
static __device__ int DoGlob(Tcl_Interp *interp, char *dir, char *rem) {
#define STATIC_SIZE 200 // When generating information for the next lower call, use static areas if the name is short, and malloc if the name is longer.
	// When this procedure is entered, the name to be globbed may already have been partly expanded by ancestor invocations of
	// DoGlob.  The part that's already been expanded is in "dir" (this may initially be empty), and the part still to expand
	// is in "rem".  This procedure expands "rem" one level, making recursive calls to itself if there's still more stuff left
	// in the remainder.
	register char *p;

	// Figure out whether we'll need to add a slash between the directory name and file names within the directory when concatenating them together.
	char *separator;
	if (dir[0] == 0 || (dir[0] == '/' && dir[1] == 0) ? "" : "/") {
		separator = (char *)"";
	}
	else {
		separator = (char *)"/";
	}

	// First, find the end of the next element in rem, checking along the way for special globbing characters.
	bool gotSpecial = false;
	char *openBrace = NULL, *closeBrace = NULL;
	for (p = rem; ; p++) {
		register char c = *p;
		if (c == '\0' || c == '/') {
			break;
		}
		if (c == '{' && !openBrace) {
			openBrace = p;
		}
		if (c == '}' && !closeBrace) {
			closeBrace = p;
		}
		if (c == '*' || c == '[' || c == '\\' || c == '?') {
			gotSpecial = true;
		}
	}

	// If there is an open brace in the argument, then make a recursive call for each element between the braces.  In this case, the
	// recursive call to DoGlob uses the same "dir" that we got. If there are several brace-pairs in a single name, we just handle
	// one here, and the others will be handled in recursive calls.
	if (openBrace) {
		char static1[STATIC_SIZE];
		if (!closeBrace) {
			Tcl_ResetResult(interp);
			interp->result = (char *)"unmatched open-brace in file name";
			return TCL_ERROR;
		}
		int remLength = strlen(rem) + 1;
		char *newRem;
		if (remLength <= STATIC_SIZE) {
			newRem = static1;
		}
		else {
			newRem = (char *)_allocFast((unsigned)remLength);
		}
		int l1 = (int)(openBrace - rem);
		strncpy(newRem, rem, l1);
		for (p = openBrace; *p != '}'; ) {
			char *element = p + 1;
			for (p = element; ((*p != '}') && (*p != ',')); p++) {} /* Empty loop body:  just find end of this element. */
			int l2 = (int)(p - element);
			strncpy(newRem + l1, element, l2);
			strcpy(newRem + l1 + l2, closeBrace + 1);
			if (DoGlob(interp, dir, newRem) != TCL_OK) {
				return TCL_ERROR;
			}
		}
		if (remLength > STATIC_SIZE) {
			_freeFast(newRem);
		}
		return TCL_OK;
	}

	// If there were any pattern-matching characters, then scan through the directory to find all the matching names.
	int result;
	if (gotSpecial) {
		// Be careful not to do any actual file system operations on a directory named "";  instead, use ".".  This is needed because
		// some versions of UNIX don't treat "" like "." automatically.
		char *dirName;
		if (*dir == '\0') {
			dirName = (char *)".";
		}
		else {
			dirName = dir;
		}
		struct stat statBuf;
		if (stat(dirName, &statBuf) != 0 || !S_ISDIR(statBuf.st_mode)) {
			return TCL_OK;
		}
		DIR *d = opendir(dirName);
		if (d == NULL) {
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "couldn't read directory \"", dirName, "\": ", Tcl_OSError(interp), (char *)NULL);
			return TCL_ERROR;
		}
		int l1 = strlen(dir);
		int l2 = (int)(p - rem);
		char static2[STATIC_SIZE];
		char *pattern;
		if (l2 < STATIC_SIZE) {
			pattern = static2;
		}
		else {
			pattern = (char *)_allocFast((unsigned)(l2 + 1));
		}
		strncpy(pattern, rem, l2);
		pattern[l2] = '\0';
		result = TCL_OK;
		while (true) {
			struct dirent *entryPtr = readdir(d);
			if (entryPtr == NULL) {
				break;
			}
			// Don't match names starting with "." unless the "." is present in the pattern.
			if (*entryPtr->d_name == '.' && *pattern != '.') {
				continue;
			}
			if (Tcl_StringMatch(entryPtr->d_name, pattern)) {
				int nameLength = strlen(entryPtr->d_name);
				if (*p == 0) {
					AppendResult(interp, dir, separator, entryPtr->d_name, nameLength);
				}
				else {
					char static1[STATIC_SIZE];
					char *newDir;
					if ((l1 + nameLength + 2) <= STATIC_SIZE) {
						newDir = static1;
					}
					else {
						newDir = (char *)_allocFast((unsigned)(l1 + nameLength + 2));
					}
					sprintf(newDir, "%s%s%s", dir, separator, entryPtr->d_name);
					result = DoGlob(interp, newDir, p + 1);
					if (newDir != static1) {
						_freeFast(newDir);
					}
					if (result != TCL_OK) {
						break;
					}
				}
			}
		}
		closedir(d);
		if (pattern != static2) {
			_freeFast(pattern);
		}
		return result;
	}

	// This is the simplest case:  just another path element.  Move it to the dir side and recurse (or just add the name to the
	// list, if we're at the end of the path).
	if (*p == 0) {
		AppendResult(interp, dir, separator, rem, (int)(p - rem));
	}
	else {
		int l1 = strlen(dir);
		int l2 = l1 + (int)(p - rem) + 2;
		char static1[STATIC_SIZE];
		char *newDir;
		if (l2 <= STATIC_SIZE) {
			newDir = static1;
		}
		else {
			newDir = (char *)_allocFast((unsigned)l2);
		}
		sprintf(newDir, "%s%s%.*s", dir, separator, (int)(p - rem), rem);
		result = DoGlob(interp, newDir, p + 1);
		if (newDir != static1) {
			_freeFast(newDir);
		}
		if (result != TCL_OK) {
			return TCL_ERROR;
		}
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* Tcl_TildeSubst --
*	Given a name starting with a tilde, produce a name where the tilde and following characters have been replaced by
*	the home directory location for the named user.
*
* Results:
*	The result is a pointer to a static string containing the new name.  This name will only persist until the next
*	call to Tcl_TildeSubst;  save it if you care about it for the long term.  If there was an error in processing the
*	tilde, then an error message is left in interp->result and the return value is NULL.
*
* Side effects:
*	None that the caller needs to worry about.
*
*----------------------------------------------------------------------
*/
#if TCL_GETWD
__device__ char *Tcl_TildeSubst(Tcl_Interp *interp, char *name) {
#define STATIC_BUF_SIZE 50
	static char staticBuf[STATIC_BUF_SIZE];
	static int curSize = STATIC_BUF_SIZE;
	static char *curBuf = staticBuf;
	int length;
	register char *p;

	if (name[0] != '~') {
		return name;
	}

	// First, find the directory name corresponding to the tilde entry.
	bool fromPw = false;
	char *dir;
	if (name[1] == '/' || name[1] == '\0') {
		dir = getenv("HOME");
		if (dir == NULL) {
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "couldn't find HOME environment ", "variable to expand \"", name, "\"", (char *)NULL);
			return NULL;
		}
		p = name + 1;
	}
	else {
		struct passwd *pwPtr;
		for (p = &name[1]; *p != 0 && *p != '/'; p++) {} // Null body;  just find end of name.
		length = p - &name[1];
		if (length >= curSize) {
			length = curSize - 1;
		}
		memcpy(curBuf, (name + 1), length);
		curBuf[length] = '\0';
		pwPtr = getpwnam(curBuf);
		if (pwPtr == NULL) {
			endpwent();
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "user \"", curBuf, "\" doesn't exist", (char *)NULL);
			return NULL;
		}
		dir = pwPtr->pw_dir;
		fromPw = true;
	}

	// Grow the buffer if necessary to make enough space for the full file name.
	length = strlen(dir) + strlen(p);
	if (length >= curSize) {
		if (curBuf != staticBuf) {
			_freeFast(curBuf);
		}
		curSize = length + 1;
		curBuf = (char *)_allocFast((unsigned)curSize);
	}

	// Finally, concatenate the directory name with the remainder of the path in the buffer.
	strcpy(curBuf, dir);
	strcat(curBuf, p);
	if (fromPw) {
		endpwent();
	}
	return curBuf;
}
#else
__device__ char *Tcl_TildeSubst(Tcl_Interp *interp, char *name) {
	return name;
}
#endif

/*
*----------------------------------------------------------------------
*
* Tcl_GlobCmd --
*	This procedure is invoked to process the "glob" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_GlobCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc < 2) {
	notEnoughArgs:
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " ?-nocomplain? name ?name ...?\"", (char *)NULL);
		return TCL_ERROR;
	}
	bool noComplain = false;
	if (args[1][0] == '-' && !strcmp(args[1], "-nocomplain")) {
		if (argc < 3) {
			goto notEnoughArgs;
		}
		noComplain = true;
	}

	int i;
	for (i = 1 + noComplain; i < argc; i++) {
		// Do special checks for names starting at the root and for names beginning with ~.  Then let DoGlob do the rest.
		char *thisName = (char *)args[i];
#if TCL_GETWD
		if (*thisName == '~') {
			thisName = Tcl_TildeSubst(interp, thisName);
			if (thisName == NULL) {
				return TCL_ERROR;
			}
		}
#endif
		int result;
		if (*thisName == '/') {
			result = DoGlob(interp, (char *)"/", thisName + 1);
		}
		else {
			result = DoGlob(interp, (char *)"", thisName);
		}
		if (result != TCL_OK) {
			return result;
		}
	}
	if (!*interp->result && !noComplain) {
		const char *sep = "";
		Tcl_AppendResult(interp, "no files matched glob pattern", (argc == 2 ? " \"" : "s \""), (char *)NULL);
		for (i = 1; i < argc; i++) {
			Tcl_AppendResult(interp, sep, args[i], (char *)NULL);
			sep = " ";
		}
		Tcl_AppendResult(interp, "\"", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}
