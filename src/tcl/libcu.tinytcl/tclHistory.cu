// tclHistory.c --
//
//	This module implements history as an optional addition to Tcl. It can be called to record commands ("events") before they are
//	executed, and it provides a command that may be used to perform history substitutions.
//
// Copyright (c) 1990-1993 The Regents of the University of California.
// Copyright (c) 1994-1995 Sun Microsystems, Inc.
// This file was integrated from tcl to tinytcl by Snapgear.
//
// See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.

#include "tclInt.h"

/*
* This history stuff is mostly straightforward, except for one thing that makes everything very complicated.  Suppose that the following
* commands get executed:
*	echo foo
*	history redo
* It's important that the history event recorded for the second command be "echo foo", not "history redo".  Otherwise, if another "history redo"
* command is typed, it will result in infinite recursions on the "history redo" command.  Thus, the actual recorded history must be
*	echo foo
*	echo foo
* To do this, the history command revises recorded history as part of its execution.  In the example above, when "history redo" starts
* execution, the current event is "history redo", but the history command arranges for the current event to be changed to "echo foo".
*
* There are three additional complications.  The first is that history substitution may only be part of a command, as in the following
* command sequence:
*	echo foo bar
*	echo [history word 3]
* In this case, the second event should be recorded as "echo bar".  Only part of the recorded event is to be modified.  Fortunately, Tcl_Eval
* helps with this by recording (in the evalFirst and evalLast fields of the intepreter) the location of the command being executed, so the
* history module can replace exactly the range of bytes corresponding to the history substitution command.
*
* The second complication is that there are two ways to revise history: replace a command, and replace the result of a command.  Consider the
* two examples below:
*	format {result is %d} $num	   |	format {result is %d} $num
*	print [history redo]		   |	print [history word 3]
* Recorded history for these two cases should be as follows:
*	format {result is %d} $num	   |	format {result is %d} $num
*	print [format {result is %d} $num] |	print $num
* In the left case, the history command was replaced with another command to be executed (the brackets were retained), but in the case on the
* right the result of executing the history command was replaced (i.e. brackets were replaced too).
*
* The third complication is that there could potentially be many history substitutions within a single command, as in:
*	echo [history word 3] [history word 2]
* There could even be nested history substitutions, as in:
*	history subs abc [history word 2]
* If history revisions were made immediately during each "history" command invocations, it would be very difficult to produce the correct cumulative
* effect from several substitutions in the same command.  To get around this problem, the actual history revision isn't made during the execution
* of the "history" command.  Information about the changes is just recorded, in xxx records, and the actual changes are made during the next call to
* Tcl_RecordHistory (when we know that execution of the previous command has finished).
*/

// Default space allocation for command strings:
#define INITIAL_CMD_SIZE 40

// Forward declarations for procedures defined later in this file:
static __device__ void DoRevs(Interp *iPtr);
static __device__ HistoryEvent *GetEvent(Interp *iPtr, char *string);
static __device__ char *GetWords(Interp *iPtr, char *command, char *words);
static __device__ void InitHistory(Interp *iPtr);
static __device__ void InsertRev(Interp *iPtr, HistoryRev *revPtr);
static __device__ void MakeSpace(HistoryEvent *hPtr, int size);
static __device__ void RevCommand(Interp *iPtr, char *string);
static __device__ void RevResult(Interp *iPtr, char *string);
static __device__ int SubsAndEval(Interp *iPtr, char *cmd, char *old, char *new_);

/*
*----------------------------------------------------------------------
*
* InitHistory --
*	Initialize history-related state in an interpreter.
*
* Results:
*	None.
*
* Side effects:
*	History info is initialized in iPtr.
*
*----------------------------------------------------------------------
*/
static __device__ void InitHistory(register Interp *iPtr)
{
	if (iPtr->numEvents != 0) {
		return;
	}
	iPtr->numEvents = 20;
	iPtr->events = (HistoryEvent *)_allocFast((unsigned)(iPtr->numEvents * sizeof(HistoryEvent)));
	for (int i = 0; i < iPtr->numEvents; i++) {
		iPtr->events[i].command = (char *)_allocFast(INITIAL_CMD_SIZE);
		*iPtr->events[i].command = 0;
		iPtr->events[i].bytesAvl = INITIAL_CMD_SIZE;
	}
	iPtr->curEvent = 0;
	iPtr->curEventNum = 0;
}

/*
*----------------------------------------------------------------------
*
* Tcl_RecordAndEval --
*	This procedure adds its command argument to the current list of recorded events and then executes the command by calling Tcl_Eval.
*
* Results:
*	The return value is a standard Tcl return value, the result of executing cmd.
*
* Side effects:
*	The command is recorded and executed.  In addition, pending history revisions are carried out, and information is set up to enable
*	Tcl_Eval to identify history command ranges.  This procedure also initializes history information for the interpreter, if it hasn't
*	already been initialized.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_RecordAndEval(Tcl_Interp *interp, char *cmd, int flags)
{
	register Interp *iPtr = (Interp *)interp;
	if (iPtr->numEvents == 0) {
		InitHistory(iPtr);
	}
	DoRevs(iPtr);

	// Don't record empty commands.
	while (isspace(*cmd)) {
		cmd++;
	}
	if (*cmd == '\0') {
		Tcl_ResetResult(interp);
		return TCL_OK;
	}
	iPtr->curEventNum++;
	iPtr->curEvent++;
	if (iPtr->curEvent >= iPtr->numEvents) {
		iPtr->curEvent = 0;
	}
	register HistoryEvent *eventPtr = &iPtr->events[iPtr->curEvent];

	// Chop off trailing newlines before recording the command.
	int length = strlen(cmd);
	while (cmd[length-1] == '\n') {
		length--;
	}
	MakeSpace(eventPtr, length + 1);
	strncpy(eventPtr->command, cmd, (size_t)length);
	eventPtr->command[length] = 0;

	// Execute the command.  Note: history revision isn't possible after a nested call to this procedure, because the event at the top of
	// the history list no longer corresponds to what's going on when a nested call here returns.  Thus, must leave history revision disabled when we return.
	int result = TCL_OK;
	if (!(flags & TCL_NO_EVAL)) {
		iPtr->historyFirst = cmd;
		iPtr->revDisables = 0;
		result = Tcl_Eval(interp, cmd, flags | TCL_RECORD_BOUNDS, 0);
	}
	iPtr->revDisables = 1;
	return result;
}

/*
*----------------------------------------------------------------------
*
* Tcl_HistoryCmd --
*	This procedure is invoked to process the "history" Tcl command. See the user documentation for details on what it does.
*
* Results:
*	A standard Tcl result.
*
* Side effects:
*	See the user documentation.
*
*----------------------------------------------------------------------
*/
__device__ int Tcl_HistoryCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[])
{
	register Interp *iPtr = (Interp *)interp;
	register HistoryEvent *eventPtr;
	if (iPtr->numEvents == 0) {
		InitHistory(iPtr);
	}
	// If no arguments, treat the same as "history info".
	if (argc == 1) {
		goto infoCmd;
	}
	int c = args[1][0];
	int length = strlen(args[1]);
	if (c == 'a' && !strncmp(args[1], "add", length)) {
		if (argc != 3 && argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " add event ?exec?\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (argc == 4) {
			if (strncmp(args[3], "exec", strlen(args[3]))) {
				Tcl_AppendResult(interp, "bad argument \"", args[3], "\": should be \"exec\"", (char *)NULL);
				return TCL_ERROR;
			}
			return Tcl_RecordAndEval(interp, (char *)args[2], 0);
		}
		return Tcl_RecordAndEval(interp, (char *)args[2], TCL_NO_EVAL);
	} else if (c == 'c' && !strncmp(args[1], "change", length)) {
		if (argc != 3 && argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " change newValue ?event?\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (argc == 3) {
			eventPtr = &iPtr->events[iPtr->curEvent];
			iPtr->revDisables += 1;
			while (iPtr->revPtr != NULL) {
				_freeFast(iPtr->revPtr->newBytes);
				HistoryRev *nextPtr = iPtr->revPtr->nextPtr;
				_freeFast((char *)iPtr->revPtr);
				iPtr->revPtr = nextPtr;
			}
		} else {
			eventPtr = GetEvent(iPtr, (char *)args[3]);
			if (eventPtr == NULL) {
				return TCL_ERROR;
			}
		}
		MakeSpace(eventPtr, (int)strlen(args[2]) + 1);
		strcpy(eventPtr->command, args[2]);
		return TCL_OK;
	} else if (c == 'e' && !strncmp(args[1], "event", length)) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " event ?event?\"", (char *)NULL);
			return TCL_ERROR;
		}
		eventPtr = GetEvent(iPtr, (char *)(argc==2 ? "-1" : args[2]));
		if (eventPtr == NULL) {
			return TCL_ERROR;
		}
		RevResult(iPtr, eventPtr->command);
		Tcl_SetResult(interp, eventPtr->command, TCL_VOLATILE);
		return TCL_OK;
	} else if (c == 'i' && !strncmp(args[1], "info", length)) {
		if (argc != 2 && argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " info ?count?\"", (char *)NULL);
			return TCL_ERROR;
		}
infoCmd:
		int count;
		if (argc == 3) {
			if (Tcl_GetInt(interp, args[2], &count) != TCL_OK) {
				return TCL_ERROR;
			}
			if (count > iPtr->numEvents) {
				count = iPtr->numEvents;
			}
		} else {
			count = iPtr->numEvents;
		}
		char *newline = "";
		int indx, i;
		for (i = 0, indx = iPtr->curEvent + 1 + iPtr->numEvents - count; i < count; i++, indx++) {
			if (indx >= iPtr->numEvents) {
				indx -= iPtr->numEvents;
			}
			char *cur = iPtr->events[indx].command;
			if (*cur == '\0') {
				continue; // No command recorded here.
			}
			char serial[20];
			sprintf(serial, "%6d  ", iPtr->curEventNum + 1 - (count - i));
			Tcl_AppendResult(interp, newline, serial, (char *)NULL);
			newline = "\n";
			// Tricky formatting here:  for multi-line commands, indent the continuation lines.
			while (true) {
				char *next = (char *)strchr(cur, '\n');
				if (next == NULL) {
					break;
				}
				next++;
				char savedChar = *next;
				*next = 0;
				Tcl_AppendResult(interp, cur, "\t", (char *)NULL);
				*next = savedChar;
				cur = next;
			}
			Tcl_AppendResult(interp, cur, (char *)NULL);
		}
		return TCL_OK;
	} else if (c == 'k' && !strncmp(args[1], "keep", length)) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " keep number\"", (char *)NULL);
			return TCL_ERROR;
		}
		int count;
		if (Tcl_GetInt(interp, args[2], &count) != TCL_OK) {
			return TCL_ERROR;
		}
		if (count <= 0 || count > 1000) {
			Tcl_AppendResult(interp, "illegal keep count \"", args[2], "\"", (char *)NULL);
			return TCL_ERROR;
		}
		// Create a new history array and copy as much existing history as possible from the old array.
		HistoryEvent *events = (HistoryEvent *)_allocFast((unsigned)(count * sizeof(HistoryEvent)));
		int src;
		if (count < iPtr->numEvents) {
			src = iPtr->curEvent + 1 - count;
			if (src < 0) {
				src += iPtr->numEvents;
			}
		} else {
			src = iPtr->curEvent + 1;
		}
		int i;
		for (i = 0; i < count; i++, src++) {
			if (src >= iPtr->numEvents) {
				src = 0;
			}
			if (i < iPtr->numEvents) {
				events[i] = iPtr->events[src];
				iPtr->events[src].command = NULL;
			} else {
				events[i].command = (char *)_allocFast(INITIAL_CMD_SIZE);
				events[i].command[0] = 0;
				events[i].bytesAvl = INITIAL_CMD_SIZE;
			}
		}
		// Throw away everything left in the old history array, and substitute the new one for the old one.
		for (i = 0; i < iPtr->numEvents; i++) {
			if (iPtr->events[i].command != NULL) {
				_freeFast(iPtr->events[i].command);
			}
		}
		_freeFast((char *)iPtr->events);
		iPtr->events = events;
		if (count < iPtr->numEvents) {
			iPtr->curEvent = count-1;
		} else {
			iPtr->curEvent = iPtr->numEvents-1;
		}
		iPtr->numEvents = count;
		return TCL_OK;
	} else if (c == 'n' && !strncmp(args[1], "nextid", length)) {
		if (argc != 2) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " nextid\"", (char *)NULL);
			return TCL_ERROR;
		}
		sprintf(iPtr->result, "%d", iPtr->curEventNum+1);
		return TCL_OK;
	} else if (c == 'r' && !strncmp(args[1], "redo", length)) {
		if (argc > 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " redo ?event?\"", (char *)NULL);
			return TCL_ERROR;
		}
		eventPtr = GetEvent(iPtr, (char *)(argc == 2 ? "-1" : args[2]));
		if (eventPtr == NULL) {
			return TCL_ERROR;
		}
		RevCommand(iPtr, eventPtr->command);
		return Tcl_Eval(interp, eventPtr->command, 0, 0);
	} else if (c == 's' && !strncmp(args[1], "substitute", length)) {
		if (argc > 5 || argc < 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " substitute old new ?event?\"", (char *)NULL);
			return TCL_ERROR;
		}
		eventPtr = GetEvent(iPtr, (char *)(argc == 4 ? "-1" : args[4]));
		if (eventPtr == NULL) {
			return TCL_ERROR;
		}
		return SubsAndEval(iPtr, eventPtr->command, (char *)args[2], (char *)args[3]);
	} else if (c == 'w' && !strncmp(args[1], "words", length)) {
		if (argc != 3 && argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " words num-num/pat ?event?\"", (char *)NULL);
			return TCL_ERROR;
		}
		eventPtr = GetEvent(iPtr, (char *)(argc == 3 ? "-1" : args[3]));
		if (eventPtr == NULL) {
			return TCL_ERROR;
		}
		char *words = GetWords(iPtr, eventPtr->command, (char *)args[2]);
		if (words == NULL) {
			return TCL_ERROR;
		}
		RevResult(iPtr, words);
		iPtr->result = words;
		iPtr->freeProc = TCL_DYNAMIC;
		return TCL_OK;
	}
	Tcl_AppendResult(interp, "bad option \"", args[1], "\": must be add, change, event, info, keep, nextid, ", "redo, substitute, or words", (char *)NULL);
	return TCL_ERROR;
}

/*
*----------------------------------------------------------------------
*
* MakeSpace --
*	Given a history event, make sure it has enough space for a string of a given length (enlarge the string area if necessary).
*
* Results:
*	None.
*
* Side effects:
*	More memory may get allocated.
*
*----------------------------------------------------------------------
*/
static __device__ void MakeSpace(HistoryEvent *hPtr, int size)
{
	if (hPtr->bytesAvl < size) {
		_freeFast(hPtr->command);
		hPtr->command = (char *)_allocFast((unsigned) size);
		hPtr->bytesAvl = size;
	}
}

/*
*----------------------------------------------------------------------
*
* InsertRev --
*	Add a new revision to the list of those pending for iPtr. Do it in a way that keeps the revision list sorted in
*	increasing order of firstIndex.  Also, eliminate revisions that are subsets of other revisions.
*
* Results:
*	None.
*
* Side effects:
*	RevPtr is added to iPtr's revision list.
*
*----------------------------------------------------------------------
*/
static __device__ void InsertRev(Interp *iPtr, register HistoryRev *revPtr)
{
	register HistoryRev *curPtr, *prevPtr;
	for (curPtr = iPtr->revPtr, prevPtr = NULL; curPtr != NULL; prevPtr = curPtr, curPtr = curPtr->nextPtr) {
		// If this revision includes the new one (or vice versa) then just eliminate the one that is a subset of the other.
		if (revPtr->firstIndex <= curPtr->firstIndex && revPtr->lastIndex >= curPtr->firstIndex) {
			curPtr->firstIndex = revPtr->firstIndex;
			curPtr->lastIndex = revPtr->lastIndex;
			curPtr->newSize = revPtr->newSize;
			_freeFast(curPtr->newBytes);
			curPtr->newBytes = revPtr->newBytes;
			_freeFast((char *)revPtr);
			return;
		}
		if (revPtr->firstIndex >= curPtr->firstIndex && revPtr->lastIndex <= curPtr->lastIndex) {
			_freeFast(revPtr->newBytes);
			_freeFast((char *)revPtr);
			return;
		}
		if (revPtr->firstIndex < curPtr->firstIndex) {
			break;
		}
	}
	// Insert revPtr just after prevPtr.
	if (prevPtr == NULL) {
		revPtr->nextPtr = iPtr->revPtr;
		iPtr->revPtr = revPtr;
	} else {
		revPtr->nextPtr = prevPtr->nextPtr;
		prevPtr->nextPtr = revPtr;
	}
}

/*
*----------------------------------------------------------------------
*
* RevCommand --
*	This procedure is invoked by the "history" command to record a command revision.  See the comments at the beginning of the
*	file for more information about revisions.
*
* Results:
*	None.
*
* Side effects:
*	Revision information is recorded.
*
*----------------------------------------------------------------------
*/
static __device__ void RevCommand(register Interp *iPtr, char *string)
{
	if (iPtr->evalFirst == NULL || iPtr->revDisables > 0) {
		return;
	}
	register HistoryRev *revPtr = (HistoryRev *)_allocFast(sizeof(HistoryRev));
	revPtr->firstIndex = (int)(iPtr->evalFirst - iPtr->historyFirst);
	revPtr->lastIndex = (int)(iPtr->evalLast - iPtr->historyFirst);
	revPtr->newSize = strlen(string);
	revPtr->newBytes = (char *)_allocFast((unsigned)(revPtr->newSize+1));
	strcpy(revPtr->newBytes, string);
	InsertRev(iPtr, revPtr);
}

/*
*----------------------------------------------------------------------
*
* RevResult --
*	This procedure is invoked by the "history" command to record a result revision.  See the comments at the beginning of the
*	file for more information about revisions.
*
* Results:
*	None.
*
* Side effects:
*	Revision information is recorded.
*
*----------------------------------------------------------------------
*/
static __device__ void RevResult(register Interp *iPtr, char *string)
{
	if (iPtr->evalFirst == NULL || iPtr->revDisables > 0) {
		return;
	}
	// Expand the replacement range to include the brackets that surround the command.  If there aren't any brackets (i.e. this command was
	// invoked at top-level) then don't do any revision.  Also, if there are several commands in brackets, of which this is just one, then don't do any revision.
	char *evalFirst = iPtr->evalFirst;
	char *evalLast = iPtr->evalLast + 1;
	while (true) {
		if (evalFirst == iPtr->historyFirst) {
			return;
		}
		evalFirst--;
		if (*evalFirst == '[') {
			break;
		}
		if (!isspace(*evalFirst)) {
			return;
		}
	}
	if (*evalLast != ']') {
		return;
	}

	register HistoryRev *revPtr = (HistoryRev *)_allocFast(sizeof(HistoryRev));
	revPtr->firstIndex = (int)(evalFirst - iPtr->historyFirst);
	revPtr->lastIndex = (int)(evalLast - iPtr->historyFirst);
	const char *args[2];
	args[0] = string;
	revPtr->newBytes = Tcl_Merge(1, args);
	revPtr->newSize = strlen(revPtr->newBytes);
	InsertRev(iPtr, revPtr);
}

/*
*----------------------------------------------------------------------
*
* DoRevs --
*	This procedure is called to apply the history revisions that have been recorded in iPtr.
*
* Results:
*	None.
*
* Side effects:
*	The most recent entry in the history for iPtr may be modified.
*
*----------------------------------------------------------------------
*/
static __device__ void DoRevs(register Interp *iPtr)
{
	if (iPtr->revPtr == NULL) {
		return;
	}
	// The revision is done in two passes.  The first pass computes the amount of space needed for the revised event, and the second pass
	// pieces together the new event and frees up the revisions.
	register HistoryEvent *eventPtr = &iPtr->events[iPtr->curEvent];
	unsigned int size = strlen(eventPtr->command) + 1;
	register HistoryRev *revPtr;
	for (revPtr = iPtr->revPtr; revPtr != NULL; revPtr = revPtr->nextPtr) {
		size -= revPtr->lastIndex + 1 - revPtr->firstIndex;
		size += revPtr->newSize;
	}

	char *newCommand = (char *)_allocFast(size);
	char *p = newCommand;
	int bytesSeen = 0;
	for (revPtr = iPtr->revPtr; revPtr != NULL;) {
		HistoryRev *nextPtr = revPtr->nextPtr;
		int count = revPtr->firstIndex - bytesSeen;
		if (count > 0) {
			strncpy(p, eventPtr->command + bytesSeen, (size_t)count);
			p += count;
		}
		strncpy(p, revPtr->newBytes, (size_t)revPtr->newSize);
		p += revPtr->newSize;
		bytesSeen = revPtr->lastIndex+1;
		_freeFast(revPtr->newBytes);
		_freeFast((char *) revPtr);
		revPtr = nextPtr;
	}
	strcpy(p, eventPtr->command + bytesSeen);

	// Replace the command in the event.
	_freeFast(eventPtr->command);
	eventPtr->command = newCommand;
	eventPtr->bytesAvl = size;
	iPtr->revPtr = NULL;
}

/*
*----------------------------------------------------------------------
*
* GetEvent --
*	Given a textual description of an event (see the manual page for legal values) find the corresponding event and return its command string.
*
* Results:
*	The return value is a pointer to the event named by "string". If no such event exists, then NULL is returned and an error message is left in iPtr.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ HistoryEvent *GetEvent(register Interp *iPtr, char *string)
{
	// First check for a numeric specification of an event.
	int index;
	if (isdigit(*string) || *string == '-') {
		int eventNum;
		if (Tcl_GetInt((Tcl_Interp *)iPtr, string, &eventNum) != TCL_OK) {
			return NULL;
		}
		if (eventNum < 0) {
			eventNum += iPtr->curEventNum;
		}
		if (eventNum > iPtr->curEventNum) {
			Tcl_AppendResult((Tcl_Interp *)iPtr, "event \"", string, "\" hasn't occurred yet", (char *)NULL);
			return NULL;
		}
		if (eventNum <= iPtr->curEventNum-iPtr->numEvents || eventNum <= 0) {
			Tcl_AppendResult((Tcl_Interp *)iPtr, "event \"", string, "\" is too far in the past", (char *)NULL);
			return NULL;
		}
		index = iPtr->curEvent + (eventNum - iPtr->curEventNum);
		if (index < 0) {
			index += iPtr->numEvents;
		}
		return &iPtr->events[index];
	}

	// Next, check for an event that contains the string as a prefix or that matches the string in the sense of Tcl_StringMatch.
	int length = strlen(string);
	for (index = iPtr->curEvent - 1; ; index--) {
		if (index < 0) {
			index += iPtr->numEvents;
		}
		if (index == iPtr->curEvent) {
			break;
		}
		register HistoryEvent *eventPtr = &iPtr->events[index];
		if (!strncmp(eventPtr->command, string, (size_t)length) || Tcl_StringMatch(eventPtr->command, string)) {
			return eventPtr;
		}
	}
	Tcl_AppendResult((Tcl_Interp *)iPtr, "no event matches \"", string, "\"", (char *)NULL);
	return NULL;
}

/*
*----------------------------------------------------------------------
*
* SubsAndEval --
*	Generate a new command by making a textual substitution in the "cmd" argument.  Then execute the new command.
*
* Results:
*	The return value is a standard Tcl error.
*
* Side effects:
*	History gets revised if the substitution is occurring on a recorded command line.  Also, the re-executed command
*	may produce side-effects.
*
*----------------------------------------------------------------------
*/
static __device__ int SubsAndEval(register Interp *iPtr, char *cmd, char *old, char *new_)
{
	// Figure out how much space it will take to hold the substituted command (and complain if the old string doesn't appear in the original command).
	int oldLength = strlen(old);
	int newLength = strlen(new_);
	char *src = cmd;
	int count = 0;
	while (true) {
		src = (char *)strstr(src, old);
		if (src == NULL) {
			break;
		}
		src += oldLength;
		count++;
	}
	if (count == 0) {
		Tcl_AppendResult((Tcl_Interp *)iPtr, "\"", old, "\" doesn't appear in event", (char *)NULL);
		return TCL_ERROR;
	}
	int length = strlen(cmd) + count*(newLength - oldLength);

	// Generate a substituted command.
	char *newCmd = (char *)_allocFast((unsigned)(length + 1));
	char *dst = newCmd;
	while (true) {
		src = (char *)strstr(cmd, old);
		if (src == NULL) {
			strcpy(dst, cmd);
			break;
		}
		strncpy(dst, cmd, (size_t)(src-cmd));
		dst += src-cmd;
		strcpy(dst, new_);
		dst += newLength;
		cmd = src + oldLength;
	}

	RevCommand(iPtr, newCmd);
	int result = Tcl_Eval((Tcl_Interp *)iPtr, newCmd, 0, 0);
	_freeFast(newCmd);
	return result;
}

/*
*----------------------------------------------------------------------
*
* GetWords --
*	Given a command string, return one or more words from the command string.
*
* Results:
*	The return value is a pointer to a dynamically-allocated string containing the words of command specified by "words".
*	If the word specifier has improper syntax then an error message is placed in iPtr->result and NULL is returned.
*
* Side effects:
*	Memory is allocated.  It is the caller's responsibilty to free the returned string..
*
*----------------------------------------------------------------------
*/
static __device__ char *GetWords(register Interp *iPtr, char *command, char *words)
{
	// Figure out whether we're looking for a numerical range or for a pattern.
	char *pattern = NULL;
	int first = 0; // First word desired. -1 means last word only.
	int last = -1; // Last word desired.  -1 means use everything up to the end.
	char *start;
	if (*words == '$') {
		if (words[1] != '\0') {
			goto error;
		}
		first = -1;
	} else if (isdigit(*words)) {
		first = strtoul(words, &start, 0);
		if (*start == 0) {
			last = first;
		} else if (*start == '-') {
			start++;
			if (*start == '$') {
				start++;
			} else if (isdigit(*start)) {
				last = strtoul(start, &start, 0);
			} else {
				goto error;
			}
			if (*start != 0) {
				goto error;
			}
		}
		if (first > last && last != -1) {
			goto error;
		}
	} else {
		pattern = words;
	}

	// Scan through the words one at a time, copying those that are relevant into the result string.  Allocate a result area large enough to hold all the words if necessary.
	int index; // Index of current word.
	register char *next;
	char *result = (char *)_allocFast((unsigned)(strlen(command) + 1));
	char *dst = result;
	for (next = command; isspace(*next); next++) { } // Empty loop body:  just find start of first word.
	for (index = 0; *next != 0; index++) {
		start = next;
		char *end = TclWordEnd(next, 0);
		if (*end != 0) {
			end++;
			for (next = end; isspace(*next); next++) { } // Empty loop body:  just find start of next word.
		}
		if (first > index || (first == -1 && *next != 0)) {
			continue;
		}
		if (last != -1 && last < index) {
			continue;
		}
		if (pattern != NULL) {
			char savedChar = *end;
			*end = 0;
			int match = Tcl_StringMatch(start, pattern);
			*end = savedChar;
			if (!match) {
				continue;
			}
		}
		if (dst != result) {
			*dst = ' ';
			dst++;
		}
		strncpy(dst, start, (size_t)(end-start));
		dst += end-start;
	}
	*dst = 0;

	// Check for an out-of-range argument index.
	if (last >= index || first >= index) {
		_freeFast(result);
		Tcl_AppendResult((Tcl_Interp *)iPtr, "word selector \"", words, "\" specified non-existent words", (char *)NULL);
		return NULL;
	}
	return result;

error:
	Tcl_AppendResult((Tcl_Interp *)iPtr, "bad word selector \"", words, "\": should be num-num or pattern", (char *)NULL);
	return NULL;
}
