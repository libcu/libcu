// tclmod.c
//
// Copyright (c) 2005 Snapgear
//
// See the file "license.terms" for information on usage and redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.

#include "tclMod.h"
#include <stringcu.h>

__device__ int tcl_split_one_arg(Tcl_Interp *interp, int *argc, const char **args[])
{
	if (*argc == 1 && strchr(*args[0], ' ')) {
		if (Tcl_SplitList(interp, (char *)*args[0], argc, args) == TCL_OK) {
			return 1;
		}
	}
	return 0;
}

/*
* Implements the common 'commands' subcommand
*/
static __device__ int tclmod_cmd_commands(Tcl_Interp *interp, int argc, const char *args[])
{
	return TCL_OK; // Nothing to do, since the result has already been created
}

/*
* Builtin command.
*/
__constant__ static const tclmod_command_type tclmod_command_entry = {
	"commands", // cmd
	nullptr, // args
	tclmod_cmd_commands, // function
	0, // minargs
	0, // maxargs
	TCL_MODFLAG_HIDDEN | TCL_MODFLAG_BUILTIN, // flags
	"Returns a list of supported commands", // description
};

/*
* Returns 0 if no match.
* Returns 1 if match and args OK.
* Returns -1 if match but args not OK (leaves error in interp->result)
*/
static __device__ int check_match_command(Tcl_Interp *interp, const tclmod_command_type *ct, int argc, const char *args[])
{
	if (!strcmp(ct->cmd, args[1])) {
		if (argc == 3 && !strcmp(args[2], "?")) {
			Tcl_AppendResult (interp, "Usage: ", args[0], " ", ct->cmd, " ", ct->args, "\n\n", ct->description, (char *)NULL);
			return -1;
		}
		if (argc < ct->minargs + 2 || (ct->maxargs >= 0 && argc > ct->maxargs + 2)) {
			Tcl_AppendResult (interp, "wrong # args: should be \"", args[0], " ", ct->cmd, " ", ct->args, "\"", (char *)NULL);
			return -1;
		}
		return 1;
	}
	return 0;
}

__device__ const tclmod_command_type *tclmod_parse_cmd(Tcl_Interp *interp, const tclmod_command_type *command_table, int argc, const char *args[])
{
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " command ...\"\n", (char *)NULL);
		Tcl_AppendResult(interp, "Use \"", args[0], " ?\" or \"", args[0], " command ?\" for help", (char *)NULL);
		return 0;
	}

	const tclmod_command_type *ct;
	for (ct = command_table; ct->cmd; ct++) {
		int ret = check_match_command(interp, ct, argc, args);
		if (ret == 1) {
			return ct; // Matched and args OK
		}
		if (ret == -1) {
			return 0; // Matched, but bad args
		}
	}

	// No match, so see if it is a builtin command
	if (!strcmp(args[1], "commands")) {
		const tclmod_command_type *ct;
		for (ct = command_table; ct->cmd; ct++) {
			if (!(ct->flags & TCL_MODFLAG_HIDDEN)) {
				Tcl_AppendElement(interp, (char *)ct->cmd, 0);
			}
		}
		return &tclmod_command_entry;
	}

	// No, so show usage
	if (!strcmp(args[1], "?")) {
		Tcl_AppendResult(interp, "Usage: \"", args[0], " command ...\", where command is one of: ", (char *)NULL);
	}
	else {
		Tcl_AppendResult(interp, "Error: ", args[0], ", unknown command \"", args[1], "\": should be ", (char *)NULL);
	}

	const char *sep = "";
	for (ct = command_table; ct->cmd; ct++) {
		if (!(ct->flags & TCL_MODFLAG_HIDDEN)) {
			Tcl_AppendResult(interp, sep, ct->cmd, (char *)NULL);
			sep = ", ";
		}
	}
	return 0;
}
