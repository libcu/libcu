#ifdef jim_ext_package

//#include <string.h>
#include "jimautoconf.h"
#include "jim-subcmd.h"
#ifdef HAVE_UNISTD_H
#include <unistdcu.h>
#else
#define R_OK 4
#endif

// All packages have a fixed, dummy version
__constant__ static const char *_package_version_1 = "1.0";

// -----------------------------------------------------------------------------
// Packages handling
// -----------------------------------------------------------------------------
#pragma region Packages handling

__device__ int Jim_PackageProvide(Jim_Interp *interp, const char *name, const char *ver, int flags)
{
	// If the package was already provided returns an error
	Jim_HashEntry *he = Jim_FindHashEntry(&interp->packages, name);
	// An empty result means the automatic entry. This can be replaced
	if (he && *(const char *)he->u.val) {
		if (flags & JIM_ERRMSG)
			Jim_SetResultFormatted(interp, "package \"%s\" was already provided", name);
		return JIM_ERROR;
	}
	Jim_ReplaceHashEntry(&interp->packages, name, (char *)ver);
	return JIM_OK;
}

// Searches along a of paths for the given package.
// Returns the allocated path to the package file if found, or NULL if not found.
static __device__ char *JimFindPackage(Jim_Interp *interp, Jim_Obj *prefixListObj, const char *pkgName)
{
	char *buf = (char *)Jim_Alloc(JIM_PATH_LEN);
	int prefixc = Jim_ListLength(interp, prefixListObj);
	for (int i = 0; i < prefixc; i++) {
		Jim_Obj *prefixObjPtr = Jim_ListGetIndex(interp, prefixListObj, i);
		const char *prefix = Jim_String(prefixObjPtr);
		// Loadable modules are tried first
#ifdef jim_ext_load
		snprintf(buf, JIM_PATH_LEN, "%s/%s.so", prefix, pkgName);
		if (access(buf, R_OK) == 0)
			return buf;
#endif
		if (strcmp(prefix, ".") == 0)
			snprintf(buf, JIM_PATH_LEN, "%s.tcl", pkgName);
		else
			snprintf(buf, JIM_PATH_LEN, "%s/%s.tcl", prefix, pkgName);
		if (access(buf, R_OK) == 0)
			return buf;
	}
	Jim_Free(buf);
	return NULL;
}

// Search for a suitable package under every dir specified by JIM_LIBPATH, and load it if possible. If a suitable package was loaded with success JIM_OK is returned, otherwise JIM_ERROR is returned.
static __device__ int JimLoadPackage(Jim_Interp *interp, const char *name, int flags)
{
	int retCode = JIM_ERROR;
	Jim_Obj *libPathObjPtr = Jim_GetVariableStr(interp, JIM_LIBPATH, JIMGLOBAL_);
	if (libPathObjPtr) {
		// Scan every directory for the the first match
		char *path = JimFindPackage(interp, libPathObjPtr, name);
		if (path) {
			// Note: Even if the file fails to load, we consider the package loaded. This prevents issues with recursion. Use a dummy version of "" to signify this case.
			Jim_PackageProvide(interp, name, "", 0);
			// Try to load/source it
			const char *p = strrchr(path, '.');
			if (p && !strcmp(p, ".tcl")) {
				Jim_IncrRefCount(libPathObjPtr);
				retCode = Jim_EvalFileGlobal(interp, path);
				Jim_DecrRefCount(interp, libPathObjPtr);
			}
#ifdef jim_ext_load
			else
				retCode = Jim_LoadLibrary(interp, path);
#endif
			// Upon failure, remove the dummy entry
			if (retCode != JIM_OK)
				Jim_DeleteHashEntry(&interp->packages, name);
			Jim_Free(path);
		}
		return retCode;
	}
	return JIM_ERROR;
}

__device__ int Jim_PackageRequire(Jim_Interp *interp, const char *name, int flags)
{
	// Start with an empty error string
	Jim_ResetResult(interp);
	Jim_HashEntry *he = Jim_FindHashEntry(&interp->packages, name);
	if (he == NULL) {
		// Try to load the package
		int retcode = JimLoadPackage(interp, name, flags);
		if (retcode != JIM_OK) {
			if (flags & JIM_ERRMSG) {
				int len = Jim_Length(Jim_GetResult(interp));
				Jim_SetResultFormatted(interp, "%#s%sCan't load package %s", Jim_GetResult(interp), len ? "\n" : "", name);
			}
			return retcode;
		}
		// In case the package did not 'package provide'
		Jim_PackageProvide(interp, name, _package_version_1, 0);
		// Now it must exist
		he = Jim_FindHashEntry(&interp->packages, name);
	}
	Jim_SetResultString(interp, (const char *)he->u.val, -1);
	return JIM_OK;
}

// package provide name ?version?
//      This procedure is invoked to declare that a particular package is now present in an interpreter. The package must not already be provided in the interpreter.
// Results:
//      Returns JIM_OK and sets results as "1.0" (the given version is ignored)
static __device__ int package_cmd_provide(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	return Jim_PackageProvide(interp, Jim_String(argv[0]), _package_version_1, JIM_ERRMSG);
}

// package require name ?version?
//      This procedure is load a given package. Note that the version is ignored.
//
// Results:
//      Returns JIM_OK and sets the package version.
static __device__ int package_cmd_require(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	// package require failing is important enough to add to the stack
	interp->addStackTrace++;
	return Jim_PackageRequire(interp, Jim_String(argv[0]), JIM_ERRMSG);
}

// package list
//      Returns a list of known packages
//
// Results:
//      Returns JIM_OK and sets a list of known packages.
static __device__ int package_cmd_list(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	Jim_Obj *listObjPtr = Jim_NewListObj(interp, NULL, 0);
	Jim_HashEntry *he;
	Jim_HashTableIterator *htiter = Jim_GetHashTableIterator(&interp->packages);
	while ((he = Jim_NextHashEntry(htiter)) != NULL)
		Jim_ListAppendElement(interp, listObjPtr, Jim_NewStringObj(interp, (const char *)he->key, -1));
	Jim_FreeHashTableIterator(htiter);
	Jim_SetResult(interp, listObjPtr);
	return JIM_OK;
}

__constant__ static const jim_subcmd_type _package_command_table[] = {
	{ "provide", "name ?version?", package_cmd_provide, 1, 2 }, // Description: Indicates that the current script provides the given package
	{ "require", "name ?version?", package_cmd_require, 1, 2 }, // Description: Loads the given package by looking in standard places
	{ "list", NULL, package_cmd_list, 0, 0 }, // Description: Lists all known packages
	{ NULL }
};

__device__ int Jim_packageInit(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "package", Jim_SubCmdProc, (void *)_package_command_table, NULL);
	return JIM_OK;
}

#pragma endregion
#endif