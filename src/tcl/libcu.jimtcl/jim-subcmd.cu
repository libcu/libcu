#pragma region License
/*
* Makes it easy to support "ensembles". i.e. commands with subcommands
* like [string] and [array]
*
* (c) 2008 Steve Bennett <steveb@workware.net.au>
*
*/
#pragma endregion

//#include <stdio.h>
#include <stringcu.h>
#include "jim-subcmd.h"

// Implements the common 'commands' subcommand
static __device__ int subcmd_null(Jim_Interp *interp, int argc, Jim_Obj *const *argv) {
	return JIM_OK; // Nothing to do, since the result has already been created
}

// Do-nothing command to support -commands and -usage
__constant__ static const jim_subcmd_type _dummy_subcmd = {
	"dummy", NULL, subcmd_null, 0, 0, JIM_MODFLAG_HIDDEN
};
static __device__ void add_commands(Jim_Interp *interp, const jim_subcmd_type *ct, const char *sep) {
	const char *s = "";
	for (; ct->cmd; ct++)
		if (!(ct->flags & JIM_MODFLAG_HIDDEN)) {
			Jim_AppendStrings(interp, Jim_GetResult(interp), s, ct->cmd, NULL);
			s = sep;
		}
}

static __device__ void bad_subcmd(Jim_Interp *interp, const jim_subcmd_type *command_table, const char *type, Jim_Obj *cmd, Jim_Obj *subcmd) {
	Jim_SetResult(interp, Jim_NewEmptyStringObj(interp));
	Jim_AppendStrings(interp, Jim_GetResult(interp), Jim_String(cmd), ", ", type, " command \"", Jim_String(subcmd), "\": should be ", NULL);
	add_commands(interp, command_table, ", ");
}

static __device__ void show_cmd_usage(Jim_Interp *interp, const jim_subcmd_type *command_table, int argc, Jim_Obj *const *argv) {
	Jim_SetResult(interp, Jim_NewEmptyStringObj(interp));
	Jim_AppendStrings(interp, Jim_GetResult(interp), "Usage: \"", Jim_String(argv[0]), " command ... \", where command is one of: ", NULL);
	add_commands(interp, command_table, ", ");
}

static __device__ void add_cmd_usage(Jim_Interp *interp, const jim_subcmd_type *ct, Jim_Obj *cmd) {
	if (cmd)
		Jim_AppendStrings(interp, Jim_GetResult(interp), Jim_String(cmd), " ", NULL);
	Jim_AppendStrings(interp, Jim_GetResult(interp), ct->cmd, NULL);
	if (ct->args && *ct->args)
		Jim_AppendStrings(interp, Jim_GetResult(interp), " ", ct->args, NULL);
}

static __device__ void set_wrong_args(Jim_Interp *interp, const jim_subcmd_type *command_table, Jim_Obj *subcmd) {
	Jim_SetResultString(interp, "wrong # args: should be \"", -1);
	add_cmd_usage(interp, command_table, subcmd);
	Jim_AppendStrings(interp, Jim_GetResult(interp), "\"", NULL);
}

__device__ const jim_subcmd_type *Jim_ParseSubCmd(Jim_Interp *interp, const jim_subcmd_type * command_table, int argc, Jim_Obj *const *argv) {
	const char *cmdname = Jim_String(argv[0]);
	if (argc < 2) {
		Jim_SetResult(interp, Jim_NewEmptyStringObj(interp));
		Jim_AppendStrings(interp, Jim_GetResult(interp), "wrong # args: should be \"", cmdname, " command ...\"\n", NULL);
		Jim_AppendStrings(interp, Jim_GetResult(interp), "Use \"", cmdname, " -help ?command?\" for help", NULL);
		return 0;
	}

	Jim_Obj *cmd = argv[1];
	// Check for the help command
	int help = 0;
	if (Jim_CompareStringImmediate(interp, cmd, "-help")) {
		if (argc == 2) {
			// Usage for the command, not the subcommand
			show_cmd_usage(interp, command_table, argc, argv);
			return &_dummy_subcmd;
		}
		help = 1;
		// Skip the 'help' command
		cmd = argv[2];
	}

	// Check for special builtin '-commands' command first
	if (Jim_CompareStringImmediate(interp, cmd, "-commands")) {
		// Build the result here
		Jim_SetResult(interp, Jim_NewEmptyStringObj(interp));
		add_commands(interp, command_table, " ");
		return &_dummy_subcmd;
	}

	int cmdlen;
	const char *cmdstr = Jim_GetString(cmd, &cmdlen);

	const jim_subcmd_type *ct;
	const jim_subcmd_type *partial = 0;
	for (ct = command_table; ct->cmd; ct++) {
		// Found an exact match
		if (Jim_CompareStringImmediate(interp, cmd, ct->cmd))
			break;
		if (!strncmp(cmdstr, ct->cmd, cmdlen)) {
			if (partial) {
				// Ambiguous
				if (help) {
					// Just show the top level help here
					show_cmd_usage(interp, command_table, argc, argv);
					return &_dummy_subcmd;
				}
				bad_subcmd(interp, command_table, "ambiguous", argv[0], argv[1 + help]);
				return 0;
			}
			partial = ct;
		}
		continue;
	}

	// If we had an unambiguous partial match
	if (partial && !ct->cmd)
		ct = partial;

	if (!ct->cmd) {
		// No matching command
		if (help) {
			// Just show the top level help here
			show_cmd_usage(interp, command_table, argc, argv);
			return &_dummy_subcmd;
		}
		bad_subcmd(interp, command_table, "unknown", argv[0], argv[1 + help]);
		return 0;
	}

	if (help) {
		Jim_SetResultString(interp, "Usage: ", -1);
		// subcmd
		add_cmd_usage(interp, ct, argv[0]);
		return &_dummy_subcmd;
	}

	// Check the number of args
	if (argc - 2 < ct->minargs || (ct->maxargs >= 0 && argc - 2 > ct->maxargs)) {
		Jim_SetResultString(interp, "wrong # args: should be \"", -1);
		// subcmd
		add_cmd_usage(interp, ct, argv[0]);
		Jim_AppendStrings(interp, Jim_GetResult(interp), "\"", NULL);
		return 0;
	}
	// Good command
	return ct;
}

__device__ int Jim_CallSubCmd(Jim_Interp *interp, const jim_subcmd_type * ct, int argc, Jim_Obj *const *argv) {
	int ret = JIM_ERROR;
	if (ct) {
		ret = (ct->flags & JIM_MODFLAG_FULLARGV ? ct->function(interp, argc, argv) : ct->function(interp, argc - 2, argv + 2));
		if (ret < 0) {
			set_wrong_args(interp, ct, argv[0]);
			ret = JIM_ERROR;
		}
	}
	return ret;
}

__device__ int Jim_SubCmdProc(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv) {
	const jim_subcmd_type *ct = Jim_ParseSubCmd(interp, (const jim_subcmd_type *)Jim_CmdPrivData(interp), argc, argv);
	return Jim_CallSubCmd(interp, ct, argc, argv);
}
