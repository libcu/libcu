#pragma region License
/*
* Implements the tcl::prefix command for Jim Tcl
*
* (c) 2011 Steve Bennett <steveb@workware.net.au>
*
* See LICENSE for license details.
*/
#pragma endregion

#include "jim.h"
#include "utf8.h"

// Returns the common initial length of the two strings.
static __device__ int JimStringCommonLength(const char *str1, int charlen1, const char *str2, int charlen2)
{
	int maxlen = 0;
	while (charlen1-- && charlen2--) {
		int c1;
		int c2;
		str1 += utf8_tounicode(str1, &c1);
		str2 += utf8_tounicode(str2, &c2);
		if (c1 != c2)
			break;
		maxlen++;
	}
	return maxlen;
}

// [tcl::prefix]
__constant__ static const char *const _prefix_options[] = { "match", "all", "longest", NULL };
__constant__ static const char *const _prefix_matchoptions[] = { "-error", "-exact", "-message", NULL };
static __device__ int Jim_TclPrefixCoreCommand(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	enum { OPT_MATCH, OPT_ALL, OPT_LONGEST };
	if (argc < 2) {
		Jim_WrongNumArgs(interp, 1, argv, "subcommand ?arg ...?");
		return JIM_ERROR;
	}
	int option;
	if (Jim_GetEnum(interp, argv[1], _prefix_options, &option, NULL, JIM_ERRMSG | JIM_ENUM_ABBREV) != JIM_OK)
		return JIM_ERROR;
	Jim_Obj *objPtr;
	Jim_Obj *stringObj;
	switch (option) {
	case OPT_MATCH:{
		enum { OPT_MATCH_ERROR, OPT_MATCH_EXACT, OPT_MATCH_MESSAGE };
		if (argc < 4) {
			Jim_WrongNumArgs(interp, 2, argv, "?options? table string");
			return JIM_ERROR;
		}
		int i;
		Jim_Obj *errorObj = NULL;
		Jim_Obj *messageObj = NULL;
		int flags = JIM_ERRMSG | JIM_ENUM_ABBREV;
		Jim_Obj *tableObj = argv[argc - 2];
		stringObj = argv[argc - 1];
		argc -= 2;
		for (i = 2; i < argc; i++) {
			int matchoption;
			if (Jim_GetEnum(interp, argv[i], _prefix_matchoptions, &matchoption, "option", JIM_ERRMSG | JIM_ENUM_ABBREV) != JIM_OK)
				return JIM_ERROR;
			switch (matchoption) {
			case OPT_MATCH_EXACT:
				flags &= ~JIM_ENUM_ABBREV;
				break;
			case OPT_MATCH_ERROR:
				if (++i == argc) {
					Jim_SetResultString(interp, "missing error options", -1);
					return JIM_ERROR;
				}
				errorObj = argv[i];
				if (Jim_Length(errorObj) % 2) {
					Jim_SetResultString(interp, "error options must have an even number of elements", -1);
					return JIM_ERROR;
				}
				break;
			case OPT_MATCH_MESSAGE:
				if (++i == argc) {
					Jim_SetResultString(interp, "missing message", -1);
					return JIM_ERROR;
				}
				messageObj = argv[i];
				break;
			}
		}
		// Do the match
		int tablesize = Jim_ListLength(interp, tableObj);
		const char **table = (const char **)Jim_Alloc((tablesize + 1) * sizeof(*table));
		for (i = 0; i < tablesize; i++) {
			Jim_ListIndex(interp, tableObj, i, &objPtr, JIM_NONE);
			table[i] = Jim_String(objPtr);
		}
		table[i] = NULL;
		int ret = Jim_GetEnum(interp, stringObj, table, &i, messageObj ? Jim_String(messageObj) : NULL, flags);
		Jim_Free(table);
		if (ret == JIM_OK) {
			Jim_ListIndex(interp, tableObj, i, &objPtr, JIM_NONE);
			Jim_SetResult(interp, objPtr);
			return JIM_OK;
		}
		if (tablesize == 0) {
			Jim_SetResultFormatted(interp, "bad option \"%#s\": no valid options", stringObj);
			return JIM_ERROR;
		}
		if (errorObj) {
			if (Jim_Length(errorObj) == 0) {
				Jim_ResetResult(interp);
				return JIM_OK;
			}
			// Do this the easy way. Build a list to evaluate
			objPtr = Jim_NewStringObj(interp, "return -level 0 -code error", -1);
			Jim_ListAppendList(interp, objPtr, errorObj);
			Jim_ListAppendElement(interp, objPtr, Jim_GetResult(interp));
			return Jim_EvalObjList(interp, objPtr);
		}
		return JIM_ERROR; }
	case OPT_ALL:
		if (argc != 4) {
			Jim_WrongNumArgs(interp, 2, argv, "table string");
			return JIM_ERROR;
		}
		else {
			int listlen = Jim_ListLength(interp, argv[2]);
			objPtr = Jim_NewListObj(interp, NULL, 0);
			for (int i = 0; i < listlen; i++) {
				Jim_Obj *valObj = Jim_ListGetIndex(interp, argv[2], i);
				if (Jim_StringCompareLenObj(interp, argv[3], valObj, 0) == 0)
					Jim_ListAppendElement(interp, objPtr, valObj);
			}
			Jim_SetResult(interp, objPtr);
			return JIM_OK;
		}
	case OPT_LONGEST:
		if (argc != 4) {
			Jim_WrongNumArgs(interp, 2, argv, "table string");
			return JIM_ERROR;
		}
		else if (Jim_ListLength(interp, argv[2])) {
			const char *longeststr = NULL;
			int longestlen = 0;
			stringObj = argv[3];
			int listlen = Jim_ListLength(interp, argv[2]);
			for (int i = 0; i < listlen; i++) {
				Jim_Obj *valObj = Jim_ListGetIndex(interp, argv[2], i);
				if (Jim_StringCompareLenObj(interp, stringObj, valObj, 0))
					continue; // Does not begin with 'string'
				if (longeststr == NULL) {
					longestlen = Jim_Utf8Length(interp, valObj);
					longeststr = Jim_String(valObj);
				}
				else
					longestlen = JimStringCommonLength(longeststr, longestlen, Jim_String(valObj), Jim_Utf8Length(interp, valObj));
			}
			if (longeststr)
				Jim_SetResultString(interp, longeststr, longestlen);
			return JIM_OK;
		}
	}
	return JIM_ERROR;
}

__device__ int Jim_tclprefixInit(Jim_Interp *interp)
{
	if (Jim_PackageProvide(interp, "tclprefix", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	Jim_CreateCommand(interp, "tcl::prefix", Jim_TclPrefixCoreCommand, NULL, NULL);
	return JIM_OK;
}
