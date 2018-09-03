//#include <stdlib.h>
//#include <string.h>
//#include <stdio.h>
#include "jim.h"
#include "jimautoconf.h"
#include "jim-subcmd.h"

static __device__ int history_cmd_getline(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#if __CUDACC__
	char *line = nullptr; //Jim_HistoryGetline(Jim_String(argv[0]));
#else
	char *line = Jim_HistoryGetline(Jim_String(argv[0]));
#endif
	// On EOF returns -1 if varName was specified; otherwise the empty string.
	if (line == NULL) {
		if (argc == 2)
			Jim_SetResultInt(interp, -1);
		return JIM_OK;
	}
	Jim_Obj *objPtr = Jim_NewStringObjNoAlloc(interp, line, -1);
	// Returns the length of the string if varName was specified
	if (argc == 2) {
		if (Jim_SetVariable(interp, argv[1], objPtr) != JIM_OK) {
			Jim_FreeNewObj(interp, objPtr);
			return JIM_ERROR;
		}
		Jim_SetResultInt(interp, Jim_Length(objPtr));
	}
	else
		Jim_SetResult(interp, objPtr);
	return JIM_OK;
}

static __device__ int history_cmd_load(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#if __CUDACC__
	//Jim_HistoryLoad(Jim_String(argv[0]));
#else
	Jim_HistoryLoad(Jim_String(argv[0]));
#endif
	return JIM_OK;
}

static __device__ int history_cmd_save(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#if __CUDACC__
	//Jim_HistorySave(Jim_String(argv[0]));
#else
	Jim_HistorySave(Jim_String(argv[0]));
#endif
	return JIM_OK;
}

static __device__ int history_cmd_add(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#if __CUDACC__
	//Jim_HistoryAdd(Jim_String(argv[0]));
#else
	Jim_HistoryAdd(Jim_String(argv[0]));
#endif
	return JIM_OK;
}

static __device__ int history_cmd_show(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#if __CUDACC__
	//Jim_HistoryShow();
#else
	Jim_HistoryShow();
#endif
	return JIM_OK;
}

__constant__ static const jim_subcmd_type _history_command_table[] = {
	{ "getline", "prompt ?varname?", history_cmd_getline, 1, 2 }, // Description: Reads one line from the user. Similar to gets.
	{ "load", "filename", history_cmd_load, 1, 1, }, // Description: Loads history from the given file, if possible
	{ "save", "filename", history_cmd_save, 1, 1 }, // Description: Saves history to the given file
	{ "add", "line", history_cmd_add, 1, 1 }, // Description: Adds the line to the history ands saves
	{ "show", NULL, history_cmd_show, 0, 0 }, // Description: Displays the history
	{ NULL }
};

static __device__ int JimHistorySubCmdProc(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	return Jim_CallSubCmd(interp, Jim_ParseSubCmd(interp, _history_command_table, argc, argv), argc, argv);
}

static __device__ void JimHistoryDelProc(ClientData privData, Jim_Interp *interp)
{
	Jim_Free(privData);
}

__device__ int Jim_historyInit(Jim_Interp *interp)
{
	if (Jim_PackageProvide(interp, "history", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	void **history = (void **)Jim_Alloc(sizeof(*history));
	*history = NULL;
	Jim_CreateCommand(interp, "history", JimHistorySubCmdProc, history, JimHistoryDelProc);
	return JIM_OK;
}
