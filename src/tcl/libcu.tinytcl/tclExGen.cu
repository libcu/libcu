// tclXgeneral.c --
//
//      Contains general extensions to the basic TCL command set.
//-----------------------------------------------------------------------------
// Copyright 1992 Karl Lehenbauer and Mark Diekhans.
//
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided
// that the above copyright notice appear in all copies.  Karl Lehenbauer and Mark Diekhans make no representations about the suitability of this
// software for any purpose.  It is provided "as is" without express or implied warranty.


//#include <signal.h>
#include <unistdcu.h>

#include "tclExInt.h"

// These globals must be set by main for the information to be defined.
__device__ char *tclxVersion = (char *)"?";   // Extended Tcl version number.
__device__ int tclxPatchlevel = 0;     // Extended Tcl patch level.

__device__ char *tclAppName = NULL;		// Application name
__device__ char *tclAppLongname = NULL;		// Long, natural language application name
__device__ char *tclAppVersion = NULL;		// Version number of the application

/*
*-----------------------------------------------------------------------------
*
* Tcl_InfoxCmd --
*    Implements the TCL infox command:
*        infox option
*
*-----------------------------------------------------------------------------
*/
__device__ int Tcl_InfoxCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2) {
		Tcl_AppendResult(interp, "bad # args: ", args[0], " option", (char *)NULL);
		return TCL_ERROR;
	}
	if (STREQU("version", args[1])) {
		Tcl_SetResult(interp, tclxVersion, TCL_STATIC);
	}
	else if (STREQU("patchlevel", args[1])) {
		char numBuf[32];
		sprintf(numBuf, "%d", tclxPatchlevel);
		Tcl_SetResult(interp, numBuf, TCL_VOLATILE);
	}
	else if (STREQU("appname", args[1])) {
		if (tclAppName != NULL)
			Tcl_SetResult(interp, tclAppName, TCL_STATIC);
	}
	else if (STREQU("applongname", args[1])) {
		if (tclAppLongname != NULL)
			Tcl_SetResult(interp, tclAppLongname, TCL_STATIC);
	}
	else if (STREQU("appversion", args[1])) {
		if (tclAppVersion != NULL)
			Tcl_SetResult(interp, tclAppVersion, TCL_STATIC);
	}
	else {
		Tcl_AppendResult(interp, "illegal option \"", args[1], "\" expect one of: version, patchlevel, appname, ", "applongname, or appversion", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*-----------------------------------------------------------------------------
*
* Tcl_SleepCmd --
*    Implements the TCL sleep command:
*        sleep seconds
*
*-----------------------------------------------------------------------------
*/
__device__ int Tcl_SleepCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc != 2) {
		Tcl_AppendResult(interp, "bad # args: ", args[0], " seconds", (char *)NULL);
		return TCL_ERROR;
	}
	sleep(atoi(args[1]));
	return TCL_OK;
}

/*
*-----------------------------------------------------------------------------
*
* Tcl_LoopCmd --
*     Implements the TCL loop command:
*         loop var start end[increment] command
*
* Results:
*      Standard TCL results.
*
*-----------------------------------------------------------------------------
*/
__device__ int Tcl_LoopCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
	if (argc < 5 || argc > 6) {
		Tcl_AppendResult(interp, "bad # args: ", args[0], " var first limit [incr] command", (char *)NULL);
		return TCL_ERROR;
	}
	int first;
	if (Tcl_GetInt(interp, args[2], &first) != TCL_OK)
		return TCL_ERROR;
	int limit;
	if (Tcl_GetInt(interp, args[3], &limit) != TCL_OK)
		return TCL_ERROR;
	const char *command;
	int incr = 1;
	if (argc == 5)
		command = args[4];
	else {
		if (Tcl_GetInt(interp, args[4], &incr) != TCL_OK)
			return TCL_ERROR;
		command = args[5];
	}
	char itxt[12];
	int result = TCL_OK;
	int i;
	for (i = first; (i < limit && incr > 0) || (i > limit && incr < 0); i += incr) {
		sprintf(itxt, "%d", i);
		if (Tcl_SetVar(interp, (char *)args[1], itxt, TCL_LEAVE_ERR_MSG) == NULL)
			return TCL_ERROR;
		result = Tcl_Eval(interp, (char *)command, 0, (char **)NULL);
		if (result != TCL_OK) {
			if (result == TCL_CONTINUE) {
				result = TCL_OK;
			}
			else if (result == TCL_BREAK) {
				result = TCL_OK;
				break;
			}
			else if (result == TCL_ERROR) {
				char buf[64];
				sprintf(buf, "\n    (\"loop\" body line %d)", interp->errorLine);
				Tcl_AddErrorInfo(interp, buf);
				break;
			}
			else {
				break;
			}
		}
	}
	// Set variable to its final value.
	sprintf(itxt, "%d", i);
	if (Tcl_SetVar(interp, (char *)args[1], itxt, TCL_LEAVE_ERR_MSG) == NULL)
		return TCL_ERROR;
	return result;
}

#if NOTSUP

#define MAX_SIGNALS 32
static __device__ int *sigloc;
static __device__ unsigned long sigsblocked;

static __device__ void signal_handler(int sig) {
	// We just remember which signal occurred. Tcl_Eval() will notice this as soon as it can and throw an error
	*sigloc = sig;
}

static __device__ void signal_ignorer(int sig) {
	// We just remember which signals occurred
	sigsblocked |= (1 << sig);
}

/**
* Given the name of a signal, returns the signal value if found, or returns -1 if not found. We accept -SIGINT, SIGINT, INT or any lowercase version
*/
static __device__ int find_signal_by_name(const char *name) {
	int i;
	// Remove optional - and SIG from the front of the name
	if (*name == '-') {
		name++;
	}
	if (!strncmp(name, "sig", 3)) {
		name += 3;
	}
	for (i = 1; i < MAX_SIGNALS; i++) {
		// Tcl_SignalId() returns names such as SIGINT, and returns "unknown signal id" if unknown, so this will work
		if (!strcmp(Tcl_SignalId(i) + 3, name)) {
			return i;
		}
	}
	return -1;
}

#endif

/*
*-----------------------------------------------------------------------------
*
* Tcl_SignalCmd --
*     Implements the TCL signal command:
*         signal ?handle|ignore|default|throw SIG...?
*
*     Specifies which signals are handled by Tcl code. If the one of the given signals is caught, it causes a TCL_SIGNAL
*     exception to be thrown which can be caught by catch.
*
*     Use 'signal ignore' to ignore the signal(s)
*     Use 'signal default' to go back to the default behaviour
*     Use 'signal throw' to rethrow a signal caught in a catch (or simulate a signal)
*
*     If no arguments are given, returns the list of signals which are being handled
*
* Results:
*      Standard TCL results.
*
*-----------------------------------------------------------------------------
*/
__device__ int Tcl_SignalCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
#if NOTSUP
#define ACTION_HANDLE 1
#define ACTION_IGNORE -1
#define ACTION_DEFAULT 0
	static int handling[MAX_SIGNALS];
	int action = ACTION_HANDLE;
	int i;
	if (argc == 1) {
		Tcl_AppendResult(interp, "bad # args: ", args[0], " handle|ignore|default|throw ?SIG...?", (char *)NULL);
		return TCL_ERROR;
	}
	if (!_strcmp(args[1], "throw")) {
		if (argc > 2) {
			int sig = SIGINT;
			if (argc > 2) {
				if ((sig = find_signal_by_name(args[2])) < 0) {
					Tcl_AppendResult(interp, args[0], " unknown signal ", args[2], (char *)NULL);
					return TCL_ERROR;
				}
			}
			// Set the canonical name of the signal as the result
			Tcl_SetResult(interp, Tcl_SignalId(sig), TCL_STATIC);
		}
		// And simply say we caught the signal
		return TCL_SIGNAL;
	}
	if (!_strcmp(args[1], "ignore")) {
		action = ACTION_IGNORE;
	}
	else if (!_strcmp(args[1], "default")) {
		action = ACTION_DEFAULT;
	}

	if (argc == 2) {
		for (i = 1; i < MAX_SIGNALS; i++) {
			if (handling[i] == action) {
				Tcl_AppendElement(interp, Tcl_SignalId(i), 0);
			}
		}
		return TCL_OK;
	}

	// Make sure we know where to store the signals which occur
	if (!sigloc) {
		sigloc = &((Interp *)interp)->signal;
	}

	// Catch all the signals we care about
	struct sigaction sa;
	if (action != ACTION_DEFAULT) {
		sa.sa_flags = 0;
		sigemptyset(&sa.sa_mask);
		if (action == ACTION_HANDLE) {
			sa.sa_handler = signal_handler;
		}
		else {
			sa.sa_handler = signal_ignorer;
		}
	}

	// Iterate through the provided signals
	for (i = 2; i < argc; i++) {
		int sig = find_signal_by_name(args[i]);
		if (sig < 0) {
			Tcl_AppendResult(interp, args[0], " unknown signal ", args[i], (char *)NULL);
			return TCL_ERROR;
		}
		static struct sigaction sa_old[MAX_SIGNALS];
		if (action != handling[sig]) {
			// Need to change the action for this signal
			switch (action) {
			case ACTION_HANDLE:
			case ACTION_IGNORE:
				if (handling[sig] == ACTION_DEFAULT) {
					sigaction(sig, &sa, &sa_old[sig]);
				}
				else {
					sigaction(sig, &sa, 0);
				}
				break;
			case ACTION_DEFAULT:
				// Restore old handler
				sigaction(sig, &sa_old[sig], 0);
			}
			handling[sig] = action;
		}
	}
#endif
	return TCL_OK;
}

/*
*-----------------------------------------------------------------------------
*
* Tcl_KillCmd --
*     Implements the TCL kill command:
*         kill SIG pid
*
* Results:
*      Standard TCL results.
*
*-----------------------------------------------------------------------------
*/
__device__ int Tcl_KillCmd(ClientData dummy, Tcl_Interp *interp, int argc, const char *args[]) {
#if NOTSUP
	if (argc != 3) {
		Tcl_AppendResult(interp, "bad # args: ", args[0], " SIG pid", (char *)NULL);
		return TCL_ERROR;
	}
	int sig = find_signal_by_name(args[1]);
	if (sig < 0) {
		Tcl_AppendResult(interp, args[0], " unknown signal ", args[1], (char *)NULL);
		return TCL_ERROR;
	}
	if (!kill(_atoi(args[2]), sig)) {
		return TCL_OK;
	}
	Tcl_AppendResult(interp, "Failed to deliver signal", (char *)NULL);
#endif
	return TCL_ERROR;
}

__device__ void TclEx_InitGeneral(Tcl_Interp *interp) {
	Tcl_CreateCommand(interp, "infox", Tcl_InfoxCmd, (ClientData)NULL, NULL);
	Tcl_CreateCommand(interp, "loop", Tcl_LoopCmd, (ClientData)NULL, NULL);
	Tcl_CreateCommand(interp, "signal", Tcl_SignalCmd, (ClientData)NULL, NULL);
	Tcl_CreateCommand(interp, "sleep", Tcl_SleepCmd, (ClientData)NULL, NULL);
	Tcl_CreateCommand(interp, "kill", Tcl_KillCmd, (ClientData)NULL, NULL);
}


