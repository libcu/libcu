// A TCL Interface to SQLite
#if HASJIMTCL
#include "TclContext.cu.h"
//#include <string.h>
//#include <stdlib.h>

#define NUM_PREPARED_STMTS 10
#define MAX_PREPARED_STMTS 100

#pragma region BLOB
#ifndef OMIT_INCRBLOB

// Close all incrblob channels opened using database connection pDb. This is called when shutting down the database connection.
static __device__ void CloseIncrblobChannels(TclContext *tctx)
{
	IncrblobChannel *next;
	for (IncrblobChannel *p = tctx->Incrblobs; p; p = next)
	{
		next = p->Next;
		// Note: Calling unregister here call Jim_Close on the incrblob channel, which deletes the IncrblobChannel structure at *p. So do not call Jim_Free() here.
		Jim_UnregisterChannel(tctx->Interp, p->Channel);
	}
}

// Close an incremental blob channel.
static __device__ int IncrblobClose(ClientData instanceData, Jim_Interp *interp)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	int rc = Vdbe::Blob_Close(p->Blob);
	Context *ctx = p->Ctx->Ctx;

	// Remove the channel from the SqliteDb.pIncrblob list.
	if (p->Next)
		p->Next->Prev = p->Prev;
	if (p->Prev)
		p->Prev->Next = p->Next;
	if (p->Ctx->Incrblobs == p)
		p->Ctx->Incrblobs = p->Next;

	// Free the IncrblobChannel structure
	Jim_Free((char *)p);

	if (rc != RC_OK)
	{
		Jim_SetResult(interp, (char *)DataEx::ErrMsg(ctx), JIM_VOLATILE);
		return JIM_ERROR;
	}
	return JIM_OK;
}

// Read data from an incremental blob channel.
static __device__ int IncrblobInput(ClientData instanceData, char *buf, int bufSize, int *errorCodePtr)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	int read = bufSize; // Number of bytes to read
	int blob = Vdbe::Blob_Bytes(p->Blob); // Total size of the blob
	if ((p->Seek + read) > blob)
		read = blob - p->Seek;
	if (read <= 0)
		return 0;
	RC rc = Vdbe::Blob_Read(p->Blob, (void *)buf, read, p->Seek);
	if (rc != RC_OK)
	{
		*errorCodePtr = rc;
		return -1;
	}
	p->Seek += read;
	return read;
}

//  Write data to an incremental blob channel.
static __device__ int IncrblobOutput(ClientData instanceData, const char *buf, int toWrite, int *errorCodePtr)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	int write = toWrite; // Number of bytes to write
	int blob = Vdbe::Blob_Bytes(p->Blob); // Total size of the blob
	if ((p->Seek + write) > blob)
	{
		*errorCodePtr = EINVAL;
		return -1;
	}
	if (write <= 0)
		return 0;
	RC rc = Vdbe::Blob_Write(p->Blob, (void *)buf, write, p->Seek);
	if (rc != RC_OK)
	{
		*errorCodePtr = EIO;
		return -1;
	}
	p->Seek += write;
	return write;
}

// Seek an incremental blob channel.
static __device__ int IncrblobSeek(ClientData instanceData, long offset, int seekMode, int *errorCodePtr)
{
	IncrblobChannel *p = (IncrblobChannel *)instanceData;
	switch (seekMode)
	{
	case SEEK_SET: p->Seek = offset; break;
	case SEEK_CUR: p->Seek += offset; break;
	case SEEK_END: p->Seek = Vdbe::Blob_Bytes(p->Blob) + offset; break;
	default: assert(!"Bad seekMode");
	}
	return p->Seek;
}

static __device__ void IncrblobWatch(ClientData instanceData, int mode) { } // NO-OP
static __device__ int IncrblobHandle(ClientData instanceData, int dir, ClientData *ptr)
{
	return JIM_ERROR;
}

__constant__ static Jim_ChannelType IncrblobChannelType = {
	"incrblob",					// typeName                            
	JIM_CHANNEL_VERSION_2,		// version                             
	IncrblobClose,				// closeProc                           
	IncrblobInput,				// inputProc                           
	IncrblobOutput,				// outputProc                          
	IncrblobSeek,				// seekProc                            
	nullptr,                    // setOptionProc                       
	nullptr,                    // getOptionProc                       
	IncrblobWatch,				// watchProc (this is a no-op)         
	IncrblobHandle,				// getHandleProc (always returns error)
	nullptr,					// close2Proc                          
	nullptr,					// blockModeProc                       
	nullptr,					// flushProc                           
	nullptr,					// handlerProc                         
	nullptr,					// wideSeekProc                        
};

// Create a new incrblob channel.
static __device__ int CreateIncrblobChannel(Jim_Interp *interp, TclContext *pDb, const char *dbName, const char *tableName, const char *columnName, int64 row, bool isReadonly)
{
	IncrblobChannel *p;
	Context *ctx = tctx->Ctx;
	int rc;
	int flags = JIM_READABLE | (isReadonly ? 0 : JIM_WRITABLE);

	Blob *blob;
	RC rc = Vdbe::Blob_Open(ctx, dbName, tableName, columnName, row, !isReadonly, &blob);
	if (rc != RC_OK)
	{
		Jim_SetResult(interp, (char *)DataEx::ErrMsg(ctx), JIM_VOLATILE);
		return JIM_ERROR;
	}

	p = (IncrblobChannel *)Jim_Alloc(sizeof(IncrblobChannel));
	p->Seek = 0;
	p->Blob = blob;

	static int count = 0; // This variable is used to name the channels: "incrblob_[incr count]"
	char channelName[64];
	snprintf(channelName, sizeof(channelName), "incrblob_%d", ++count);
	p->Channel = Jim_CreateChannel(&IncrblobChannelType, channelName, p, flags);
	Jim_RegisterChannel(interp, p->Channel);

	// Link the new channel into the SqliteDb.pIncrblob list.
	p->Next = tctx->Incrblob;
	p->Prev = nullptr;
	if (p->Next)
		p->Next->Prev = p;
	tctx->Incrblob = p;
	p->Ctx = tctx;

	Jim_SetResult(interp, (char *)Jim_GetChannelName(p->Channel), JIM_VOLATILE);
	return JIM_OK;
}
#else
#define CloseIncrblobChannels(tctx)
#endif
#pragma endregion

#pragma region Stmt

// Look at the script prefix in pCmd.  We will be executing this script after first appending one or more arguments.  This routine analyzes
// the script to see if it is safe to use Jim_EvalObjv() on the script rather than the more general Jim_EvalEx().  Jim_EvalObjv() is much faster.
//
// Scripts that are safe to use with Jim_EvalObjv() consists of a command name followed by zero or more arguments with no [...] or $
// or {...} or ; to be seen anywhere.  Most callback scripts consist of just a single procedure name and they meet this requirement.
static __device__ bool SafeToUseEvalObjv(Jim_Interp *interp, Jim_Obj *cmd)
{
	// We could try to do something with Jim_Parse().  But we will instead just do a search for forbidden characters.  If any of the forbidden
	// characters appear in pCmd, we will report the string as unsafe.
	int n;
	const char *z = Jim_GetString(cmd, &n);
	while (n-- > 0)
	{
		int c = *(z++);
		if (c == '$' || c == '[' || c == ';') return false;
	}
	return true;
}

// Find an SqlFunc structure with the given name.  Or create a new one if an existing one cannot be found.  Return a pointer to the structure.
static __device__ SqlFunc *FindSqlFunc(TclContext *tctx, const char *name)
{
	SqlFunc *newFunc = (SqlFunc *)Jim_Alloc(sizeof(*newFunc) + strlen(name) + 1);
	newFunc->Name = (char *)&newFunc[1];
	int i;
	for (i = 0; name[i]; i++) { newFunc->Name[i] = _tolower(name[i]); }
	newFunc->Name[i] = 0;
	for (SqlFunc *p = tctx->Funcs; p; p = p->Next)
	{ 
		if (!strcmp(p->Name, newFunc->Name))
		{
			Jim_Free((char *)newFunc);
			return p;
		}
	}
	newFunc->Interp = tctx->Interp;
	newFunc->Ctx = tctx;
	newFunc->Script = nullptr;
	newFunc->Next = tctx->Funcs;
	tctx->Funcs = newFunc;
	return newFunc;
}

// Free a single SqlPreparedStmt object.
static __device__ void DbFreeStmt(SqlPreparedStmt *stmt)
{
#ifdef _TEST
	if (!Vdbe::Sql(stmt->Stmt))
		Jim_Free((char *)stmt->Sql);
#endif
	Vdbe::Finalize(stmt->Stmt);
	Jim_Free(stmt);
}

// Finalize and free a list of prepared statements
static __device__ void FlushStmtCache(TclContext *tctx)
{
	SqlPreparedStmt *next;
	for (SqlPreparedStmt *p = tctx->Stmts.data; p; p = next)
	{
		next = p->Next;
		DbFreeStmt(p);
	}
	tctx->Stmts.length = 0;
	tctx->StmtLast = nullptr;
	tctx->Stmts.data = nullptr;
}

// JIM calls this procedure when an sqlite3 database command is deleted.
static __device__ void DbDeleteCmd(ClientData data, Jim_Interp *interp)
{
	TclContext *tctx = (TclContext *)data;
	FlushStmtCache(tctx);
	CloseIncrblobChannels(tctx);
	DataEx::Close(tctx->Ctx);
	while (tctx->Funcs)
	{
		SqlFunc *func = tctx->Funcs;
		tctx->Funcs = func->Next;
		_assert(func->Ctx == tctx);
		Jim_DecrRefCount(interp, func->Script);
		Jim_Free(func);
	}
	while (tctx->Collates)
	{
		SqlCollate *collate = tctx->Collates;
		tctx->Collates = collate->Next;
		Jim_Free(collate);
	}
	if (tctx->Busy)
		Jim_Free(tctx->Busy);
	if (tctx->Trace)
		Jim_Free(tctx->Trace);
	if (tctx->Profile)
		Jim_Free(tctx->Profile);
	if (tctx->Auth)
		Jim_Free(tctx->Auth);
	if (tctx->NullText)
		Jim_Free(tctx->NullText);
	if (tctx->UpdateHook)
		Jim_DecrRefCount(interp, tctx->UpdateHook);
	if (tctx->RollbackHook)
		Jim_DecrRefCount(interp, tctx->RollbackHook);
	if (tctx->WalHook)
		Jim_DecrRefCount(interp, tctx->WalHook);
	if (tctx->CollateNeeded)
		Jim_DecrRefCount(interp, tctx->CollateNeeded);
	Jim_Free(tctx);
}

#pragma endregion

#pragma region Hooks

// This routine is called when a database file is locked while trying to execute SQL.
static __device__ int DbBusyHandler(void *cd, int tries)
{
	TclContext *tctx = (TclContext *)cd;
	Jim_Interp *interp = tctx->Interp;
	char b[30];
	snprintf(b, sizeof(b), "%d", tries);
	Jim_Obj *objPtr = Jim_NewStringObj(interp, tctx->Busy, -1);
	Jim_AppendStrings(interp, objPtr, " ", b, NULL);
	int rc = Jim_EvalObj(interp, objPtr);
	if (rc != JIM_OK || _atoi(Jim_String(Jim_GetResult(interp))))
		return 0;
	return 1;
}

#ifndef OMIT_PROGRESS_CALLBACK
// This routine is invoked as the 'progress callback' for the database.
static __device__ int DbProgressHandler(void *cd)
{
	TclContext *tctx = (TclContext *)cd;
	Jim_Interp *interp = tctx->Interp;
	_assert(tctx->Progress);
	int rc = Jim_Eval(interp, tctx->Progress);
	if (rc != JIM_OK || _atoi(Jim_String(Jim_GetResult(interp))))
		return RC_ERROR;
	return RC_OK;
}
#endif

#ifndef OMIT_TRACE
// This routine is called by the SQLite trace handler whenever a new block of SQL is executed.  The JIM script in pDb->zTrace is executed.
static __device__ void DbTraceHandler(void *cd, const char *sql)
{
	TclContext *tctx = (TclContext *)cd;
	Jim_Interp *interp = tctx->Interp;
	Jim_Obj *str = Jim_NewStringObj(interp, tctx->Trace, -1);
	Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, sql, -1));
	Jim_Eval(interp, sql);
	Jim_ResetResult(interp);
}

// This routine is called by the SQLite profile handler after a statement SQL has executed.  The JIM script in pDb->zProfile is evaluated.
static __device__ void DbProfileHandler(void *cd, const char *sql, uint64 tm)
{
	TclContext *tctx = (TclContext *)cd;
	Jim_Interp *interp = tctx->Interp;
	char tmAsString[100];
	snprintf(tmAsString, sizeof(tmAsString)-1, "%lld", tm);
	Jim_Obj *str = Jim_NewStringObj(interp, tctx->Profile, -1);
	Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, sql, -1));
	Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, tmAsString, -1));
	Jim_EvalObj(interp, str);
	Jim_ResetResult(interp);
}
#endif

// This routine is called when a transaction is committed.  The JIM script in pDb->zCommit is executed.  If it returns non-zero or
// if it throws an exception, the transaction is rolled back instead of being committed.
static __device__ ::RC DbCommitHandler(void *cd)
{
	TclContext *tctx = (TclContext *)cd;
	int rc = Jim_Eval(tctx->Interp, tctx->Commit);
	if (rc != JIM_OK || _atoi(Jim_String(Jim_GetResult(tctx->Interp))))
		return RC_ERROR;
	return RC_OK;
}

static __device__ void DbRollbackHandler(void *clientData)
{
	TclContext *tctx = (TclContext *)clientData;
	_assert(tctx->RollbackHook);
	Jim_EvalObjBackground(tctx->Interp, tctx->RollbackHook);
}

// This procedure handles wal_hook callbacks.
static __device__ int DbWalHandler(void *clientData, Context *ctx, const char *dbName,  int entrys)
{
	TclContext *tctx = (TclContext *)clientData;
	Jim_Interp *interp = tctx->Interp;
	_assert(tctx->WalHook);
	int rc = RC_OK;
	Jim_Obj *p = Jim_DuplicateObj(interp, tctx->WalHook);
	Jim_IncrRefCount(p);
	Jim_ListAppendElement(interp, p, Jim_NewStringObj(interp, dbName, -1));
	Jim_ListAppendElement(interp, p, Jim_NewIntObj(interp, entrys));
	if (Jim_EvalObj(interp, p) != JIM_OK || Jim_GetInt(interp, Jim_GetResult(interp), &rc) != JIM_OK)
		Jim_BackgroundError(interp);
	Jim_DecrRefCount(interp, p);
	return rc;
}

#if defined(_TEST) && defined(ENABLE_UNLOCK_NOTIFY)
static __device__ void SetTestUnlockNotifyVars(Jim_Interp *interp, int argId, int argsLength)
{
	char b[64];
	snprintf(b, sizeof(b), "%d", argId);
	Jim_SetVar(interp, "sqlite_unlock_notify_arg", b, JIMGLOBAL__ONLY);
	snprintf(b, sizeof(b), "%d", argsLength);
	Jim_SetVar(interp, "sqlite_unlock_notify_argcount", b, JIMGLOBAL__ONLY);
}
#else
#define SetTestUnlockNotifyVars(x,y,z)
#endif

#ifdef ENABLE_UNLOCK_NOTIFY
static __device__ void DbUnlockNotify(void **args, int argsLength)
{
	for (int i = 0; i < argsLength; i++)
	{
		const int flags = (JIM_EVALGLOBAL_ | JIM_EVAL_DIRECT);
		TclContext *tctx = (TclContext *)args[i];
		Jim_Interp *interp = tctx->Interp;
		SetTestUnlockNotifyVars(interp, i, argsLength);
		_assert(tctx->UnlockNotify);
		Jim_EvalObjEx(interp, *tctx->UnlockNotify, flags);
		Jim_DecrRefCount(tctx->UnlockNotify);
		tctx->UnlockNotify = nullptr;
	}
}
#endif

static __device__ void DbUpdateHandler(void *p, TK op, const char *dbName, const char *tableName, int64 rowid)
{
	TclContext *tctx = (TclContext *)p;
	Jim_Interp *interp = tctx->Interp;
	_assert(tctx->UpdateHook);
	_assert(op == TK_INSERT || op == TK_UPDATE || op == TK_DELETE);
	Jim_Obj *cmd = Jim_DuplicateObj(interp, tctx->UpdateHook);
	Jim_IncrRefCount(cmd);
	Jim_ListAppendElement(0, cmd, Jim_NewStringObj(interp, (op == TK_INSERT?"INSERT":(op == TK_UPDATE?"UPDATE":"DELETE")), -1));
	Jim_ListAppendElement(interp, cmd, Jim_NewStringObj(interp, dbName, -1));
	Jim_ListAppendElement(interp, cmd, Jim_NewStringObj(interp, tableName, -1));
	Jim_ListAppendElement(interp, cmd, Jim_NewIntObj(interp, rowid));
	Jim_EvalObj(interp, cmd);
}

static __device__ void TclCollateNeeded(void *p, Context *ctx, TEXTENCODE encode, const char *name)
{
	TclContext *tctx = (TclContext *)p;
	Jim_Interp *interp = tctx->Interp;
	Jim_Obj *script = Jim_DuplicateObj(interp, tctx->CollateNeeded);
	//Jim_IncrRefCount(script);
	Jim_ListAppendElement(interp, script, Jim_NewStringObj(interp, name, -1));
	Jim_EvalObj(interp, script);
	//Jim_DecrRefCount(interp, script);
}

// This routine is called to evaluate an SQL collation function implemented using JIM script.
static __device__ int TclSqlCollate(void *p1, int aLength, const void *a, int bLength, const void *b)
{
	SqlCollate *p = (SqlCollate *)p1;
	Jim_Interp *interp = p->Interp;
	Jim_Obj *cmd = Jim_NewStringObj(interp, p->Script, -1);
	//Jim_IncrRefCount(cmd);
	Jim_ListAppendElement(interp, cmd, Jim_NewStringObj(interp, (const char *)a, aLength));
	Jim_ListAppendElement(interp, cmd, Jim_NewStringObj(interp, (const char *)b, bLength));
	Jim_EvalObj(interp, cmd);
	//Jim_DecrRefCount(interp, cmd);
	return _atoi(Jim_String(Jim_GetResult(interp)));
}

// This routine is called to evaluate an SQL function implemented using JIM script.
static __device__ void TclSqlFunc(FuncContext *fctx, int argc, Mem **argv)
{
	SqlFunc *p = (SqlFunc *)Vdbe::User_Data(fctx);
	Jim_Interp *interp = p->Interp;
	Jim_Obj *cmd;
	int rc;
	if (argc == 0)
	{
		// If there are no arguments to the function, call Jim_EvalObjEx on the script object directly.  This allows the JIM compiler to generate
		// bytecode for the command on the first invocation and thus make subsequent invocations much faster.
		cmd = p->Script;
		//Jim_IncrRefCount(cmd);
		rc = Jim_EvalObj(interp, cmd);
		//Jim_DecrRefCount(interp, cmd);
	}
	else
	{
		// If there are arguments to the function, make a shallow copy of the script object, lappend the arguments, then evaluate the copy.
		//
		// By "shallow" copy, we mean a only the outer list Jim_Obj is duplicated. The new Jim_Obj contains pointers to the original list elements. 
		// That way, when Jim_EvalObjv() is run and shimmers the first element of the list to tclCmdNameType, that alternate representation will
		// be preserved and reused on the next invocation.
		cmd = Jim_DuplicateObj(interp, p->Script);
		Jim_IncrRefCount(cmd);
		for (int i = 0; i < argc; i++)
		{
			Mem *in = argv[i];
			Jim_Obj *val;
			// Set pVal to contain the i'th column of this row.
			switch (Vdbe::Value_Type(in))
			{
			case TYPE_BLOB: {
				int bytes = Vdbe::Value_Bytes(in);
				val = Jim_NewStringObj(interp, (char *)Vdbe::Value_Blob(in), bytes);
				break; }
			case TYPE_INTEGER: {
				int64 v = Vdbe::Value_Int64(in);
				//val = (v >= -2147483647 && v <= 2147483647 ? Jim_NewIntObj(interp, v) : Jim_NewIntObj(interp, v));
				val = Jim_NewIntObj(interp, v);
				break; }
			case TYPE_FLOAT: {
				double r = Vdbe::Value_Double(in);
				val = Jim_NewDoubleObj(interp, r);
				break; }
			case TYPE_NULL: {
				val = Jim_NewStringObj(interp, "", 0);
				break; }
			default: {
				int bytes = Vdbe::Value_Bytes(in);
				val = Jim_NewStringObj(interp, (char *)Vdbe::Value_Text(in), bytes);
				break; }
			}
			Jim_ListAppendElement(interp, cmd, val);
		}
		// Jim_EvalOb() will automatically call Jim_EvalObjVector() if pCmd is a list without a string representation.  To prevent this from happening, make sure pCmd has a valid string representation
		if (!p->UseEvalObjv)
			Jim_String(cmd);
		rc = Jim_EvalObj(interp, cmd);
		Jim_DecrRefCount(interp, cmd);
	}

	if (rc && rc != JIM_RETURN)
		Vdbe::Result_Error(fctx, Jim_String(Jim_GetResult(interp)), -1); 
	else
	{
		Jim_Obj *var = Jim_GetResult(interp);
		int n;
		char *data;
		// XXX: Jim Tcl doesn't have bytearray or boolean
		const char *typeName = (var->typePtr ? var->typePtr->name : "");
		char c = typeName[0];
#if 0
		if (c == 'b' && !strcmp(typeName, "bytearray") && var->Bytes == 0)
		{
			// Only return a BLOB type if the Tcl variable is a bytearray and has no string representation.
			data = Jim_GetByteArray(var, &n);
			Vdbe::Result_Blob(fctx, data, n, DESTRUCTOR_TRANSIENT);
		}
		else if (c == 'b' && !strcmp(typeName, "boolean"))
		{
			Jim_GetWide(nullptr, var, &n);
			Vdbe::Result_Int(fctx, n);
		} else
#endif
			if (c == 'd' && !strcmp(typeName, "double"))
			{
				double r;
				Jim_GetDouble(nullptr, var, &r);
				Vdbe::Result_Double(fctx, r);
			}
			// XXX: Is a cooerced double better as a double or an int?
			else if ((c == 'c' && !strcmp(typeName, "coerced-double")) || (c == 'i' && !strcmp(typeName, "int")))
			{
				int64 v;
				Jim_GetWide(interp, var, &v);
				Vdbe::Result_Int64(fctx, v);
			}
			else
			{
				data = (char *)Jim_GetString(var, &n);
				Vdbe::Result_Text(fctx, data, n, DESTRUCTOR_TRANSIENT);
			}
	}
}

#ifndef OMIT_AUTHORIZATION
// This is the authentication function.  It appends the authentication type code and the two arguments to zCmd[] then invokes the result
// on the interpreter.  The reply is examined to determine if the authentication fails or succeeds.
static __device__ ARC AuthCallback(void *arg, int code, const char *arg1, const char *arg2, const char *arg3, const char *arg4)
{
	TclContext *tctx = (TclContext *)arg;
	Jim_Interp *interp = tctx->Interp;
	if (tctx->DisableAuth) return ARC_OK;

	char *codeName;
	switch (code)
	{
	case AUTH_COPY              : codeName="AUTH_COPY"; break;
	case AUTH_CREATE_INDEX      : codeName="AUTH_CREATE_INDEX"; break;
	case AUTH_CREATE_TABLE      : codeName="AUTH_CREATE_TABLE"; break;
	case AUTH_CREATE_TEMP_INDEX : codeName="AUTH_CREATE_TEMP_INDEX"; break;
	case AUTH_CREATE_TEMP_TABLE : codeName="AUTH_CREATE_TEMP_TABLE"; break;
	case AUTH_CREATE_TEMP_TRIGGER: codeName="AUTH_CREATE_TEMP_TRIGGER"; break;
	case AUTH_CREATE_TEMP_VIEW  : codeName="AUTH_CREATE_TEMP_VIEW"; break;
	case AUTH_CREATE_TRIGGER    : codeName="AUTH_CREATE_TRIGGER"; break;
	case AUTH_CREATE_VIEW       : codeName="AUTH_CREATE_VIEW"; break;
	case AUTH_DELETE            : codeName="AUTH_DELETE"; break;
	case AUTH_DROP_INDEX        : codeName="AUTH_DROP_INDEX"; break;
	case AUTH_DROP_TABLE        : codeName="AUTH_DROP_TABLE"; break;
	case AUTH_DROP_TEMP_INDEX   : codeName="AUTH_DROP_TEMP_INDEX"; break;
	case AUTH_DROP_TEMP_TABLE   : codeName="AUTH_DROP_TEMP_TABLE"; break;
	case AUTH_DROP_TEMP_TRIGGER : codeName="AUTH_DROP_TEMP_TRIGGER"; break;
	case AUTH_DROP_TEMP_VIEW    : codeName="AUTH_DROP_TEMP_VIEW"; break;
	case AUTH_DROP_TRIGGER      : codeName="AUTH_DROP_TRIGGER"; break;
	case AUTH_DROP_VIEW         : codeName="AUTH_DROP_VIEW"; break;
	case AUTH_INSERT            : codeName="AUTH_INSERT"; break;
	case AUTH_PRAGMA            : codeName="AUTH_PRAGMA"; break;
	case AUTH_READ              : codeName="AUTH_READ"; break;
	case AUTH_SELECT            : codeName="AUTH_SELECT"; break;
	case AUTH_TRANSACTION       : codeName="AUTH_TRANSACTION"; break;
	case AUTH_UPDATE            : codeName="AUTH_UPDATE"; break;
	case AUTH_ATTACH            : codeName="AUTH_ATTACH"; break;
	case AUTH_DETACH            : codeName="AUTH_DETACH"; break;
	case AUTH_ALTER_TABLE       : codeName="AUTH_ALTER_TABLE"; break;
	case AUTH_REINDEX           : codeName="AUTH_REINDEX"; break;
	case AUTH_ANALYZE           : codeName="AUTH_ANALYZE"; break;
	case AUTH_CREATE_VTABLE     : codeName="AUTH_CREATE_VTABLE"; break;
	case AUTH_DROP_VTABLE       : codeName="AUTH_DROP_VTABLE"; break;
	case AUTH_FUNCTION          : codeName="AUTH_FUNCTION"; break;
	case AUTH_SAVEPOINT         : codeName="AUTH_SAVEPOINT"; break;
	default                     : codeName="????"; break;
	}
	Jim_Obj *str = Jim_NewStringObj(interp, tctx->Auth, -1);
	// XXX: list or string here?
	Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, codeName, -1));
	if (arg1) Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, arg1, -1));
	if (arg2) Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, arg2, -1));
	if (arg3) Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, arg3, -1));
	if (arg4) Jim_ListAppendElement(interp, str, Jim_NewStringObj(interp, arg4, -1));
	Jim_IncrRefCount(str);
	int rc = Jim_EvalGlobal(interp, Jim_String(str));
	Jim_DecrRefCount(interp, str);
	rc = ARC_OK;
	const char *reply = (rc == RC_OK ? Jim_String(Jim_GetResult(interp)) : "ARC_DENY");
	if (!strcmp(reply, "ARC_OK")) rc = ARC_OK;
	else if (!strcmp(reply, "ARC_DENY")) rc = ARC_DENY;
	else if (!strcmp(reply, "ARC_IGNORE")) rc = ARC_IGNORE;
	else rc = 999;
	return (ARC)rc;
}
#endif

#pragma endregion

// JIM: Note that Jim Tcl can't do encoding conversion, so this simply returns the string as an object.
static __device__ Jim_Obj *dbTextToObj(Jim_Interp *interp, char const *text) { return Jim_NewStringObj(interp, (text ? text : ""), -1); }

#pragma region GetLine

static __device__ char *LocalGetLine(char *prompt, FILE *in)
{
	int lineLength = 100;
	char *line = (char *)_alloc(lineLength);
	if (!line) return nullptr;

	int n = 0;
	while (true)
	{
		if (n+100 > lineLength)
		{
			lineLength = lineLength*2 + 100;
			line = (char *)_realloc(line, lineLength);
			if (!line) return nullptr;
		}
		if (!_fgets(&line[n], lineLength - n, in))
		{
			if (n == 0)
			{
				_free(line);
				return 0;
			}
			line[n] = 0;
			break;
		}
		while (line[n]) { n++; }
		if (n > 0 && line[n-1] == '\n')
		{
			n--;
			line[n] = 0;
			break;
		}
	}
	line = (char *)_realloc(line, n + 1);
	return line;
}

#pragma endregion

#pragma region DB

__constant__ static const char *_trans_ends[] = {
	"RELEASE _tcl_transaction",        // rc==JIM_ERROR, nTransaction!=0
	"COMMIT",                          // rc!=JIM_ERROR, nTransaction==0
	"ROLLBACK TO _tcl_transaction ; RELEASE _tcl_transaction",
	"ROLLBACK"                         // rc==JIM_ERROR, nTransaction==0
};
static __device__ int DbTransPostCmd(ClientData data[], Jim_Interp *interp, int result)
{
	TclContext *tctx = (TclContext *)data[0];
	Context *ctx = tctx->Ctx;
	int rc = result;

	tctx->Transactions--;
	const char *end = _trans_ends[(rc == JIM_ERROR) * 2 + (tctx->Transactions == 0)];

	tctx->DisableAuth++;
	if (DataEx::Exec(ctx, end, nullptr, nullptr, nullptr))
	{
		// This is a tricky scenario to handle. The most likely cause of an error is that the exec() above was an attempt to commit the 
		// top-level transaction that returned SQLITE_BUSY. Or, less likely, that an IO-error has occurred. In either case, throw a Jim exception
		// and try to rollback the transaction.
		//
		// But it could also be that the user executed one or more BEGIN, COMMIT, SAVEPOINT, RELEASE or ROLLBACK commands that are confusing
		// this method's logic. Not clear how this would be best handled.
		if (rc != JIM_ERROR)
		{
			Jim_AppendString(interp, Jim_GetResult(interp), DataEx::ErrMsg(ctx), -1);
			rc = JIM_ERROR;
		}
		DataEx::Exec(ctx, "ROLLBACK", nullptr, nullptr, nullptr);
	}
	tctx->DisableAuth--;
	return rc;
}

static __device__ RC DbPrepare(TclContext *tctx, const char *sql, Vdbe **stmt, const char **out)
{
#ifdef _TEST
	if (tctx->LegacyPrepare)
		return Prepare::Prepare_(tctx->Ctx, sql, -1, stmt, out);
#endif
	return Prepare::Prepare_v2(tctx->Ctx, sql, -1, stmt, out);
}

static __device__ RC DbPrepareAndBind(TclContext *tctx, char const *sql, char const **out, SqlPreparedStmt **preStmt)
{
	Context *ctx = tctx->Ctx;
	Jim_Interp *interp = tctx->Interp;
	*preStmt = nullptr;

	// Trim spaces from the start of zSql and calculate the remaining length.
	while (isspace(sql[0])) sql++;
	int sqlLength = strlen(sql);

	SqlPreparedStmt *p;
	Vdbe *stmt;
	int vars;
	for (p = tctx->Stmts; p; p = p->Next)
	{
		int n = p->SqlLength;
		if (sqlLength >= n && !memcmp(p->Sql, sql, n) && (sql[n] == 0 || sql[n-1] == ';'))
		{
			stmt = p->Stmt;
			*out = &sql[p->SqlLength];

			// When a prepared statement is found, unlink it from the cache list.  It will later be added back to the beginning
			// of the cache list in order to implement LRU replacement.
			if (p->Prev)
				p->Prev->Next = p->Next;
			else
				tctx->Stmts = p->Next;
			if (p->Next)
				p->Next->Prev = p->Prev;
			else
				tctx->StmtLast = p->Prev;
			tctx->Stmts.length--;
			vars = Vdbe::Bind_ParameterCount(stmt);
			break;
		}
	}

	// If no prepared statement was found. Compile the SQL text. Also allocate a new SqlPreparedStmt structure.
	if (!p)
	{
		if (DbPrepare(tctx, sql, &stmt, out) != RC_OK)
		{
			Jim_SetResult(interp, dbTextToObj(interp, DataEx::ErrMsg(ctx)));
			return RC_ERROR;
		}
		if (!stmt)
		{
			if (DataEx::ErrCode(ctx) != RC_OK)
			{
				// A compile-time error in the statement.
				Jim_SetResult(interp, dbTextToObj(interp, DataEx::ErrMsg(ctx)));
				return RC_ERROR;
			}
			// The statement was a no-op.  Continue to the next statement in the SQL string.
			else
				return RC_OK; 
		}

		_assert(!p);
		vars = Vdbe::Bind_ParameterCount(stmt);
		int bytes = sizeof(SqlPreparedStmt) + vars * sizeof(Jim_Obj *);
		p = (SqlPreparedStmt *)_alloc(bytes);
		memset(p, 0, bytes);

		p->Stmt = stmt;
		p->SqlLength = (int)(*out - sql);
		p->Sql = Vdbe::Sql(stmt);
		p->Parms.data = (Jim_Obj **)&p[1];
#ifdef _TEST
		if (!p->Sql)
		{
			char *copy = (char *)_alloc(p->SqlLength + 1);
			memcpy(copy, sql, p->SqlLength);
			copy[p->SqlLength] = '\0';
			p->Sql = copy;
		}
#endif
	}
	_assert(p);
	_assert(strlen(p->Sql) == p->SqlLength);
	_assert(!memcmp(p->Sql, sql, p->SqlLength));

	// Bind values to parameters that begin with $ or :
	int parmsLength = 0;
	for (int i = 1; i <= vars; i++)
	{
		const char *varName = Vdbe::Bind_ParameterName(stmt, i);
		if (varName && (varName[0] == '$' || varName[0] == ':' || varName[0] == '@'))
		{
			Jim_Obj *var = Jim_GetVariableStr(interp, (char *)&varName[1], 0);
			if (var)
			{
				int n;
				char *data;
				const char *typeName = (var->typePtr ? var->typePtr->name : "");
				char c = typeName[0];
				// XXX: Jim Tcl doesn't have bytearray or boolean
				if (varName[0] == '@') //|| (c == 'b' && !strcmp(typeName, "bytearray") && var->Bytes == 0))
				{
					// Load a BLOB type if the Tcl variable is a bytearray and it has no string representation or the host parameter name begins with "@".
					data = (char *)Jim_GetByteArray(var, &n);
					Vdbe::Bind_Blob(stmt, i, data, n, DESTRUCTOR_STATIC);
					Jim_IncrRefCount(var);
					p->Parms[parmsLength++] = var;
#if 0
				}
				else if (c == 'b' && !strcmp(typeName, "boolean"))
				{
					Jim_GetWide(interp, var, &n);
					Vdbe::Bind_Int(stmt, i, n);
#endif
				}
				else if (c == 'd' && !strcmp(typeName, "double"))
				{
					double r;
					Jim_GetDouble(interp, var, &r);
					Vdbe::Bind_Double(stmt, i, r);
				}
				else if ((c == 'c' && !strcmp(typeName, "coerced-double")) || (c == 'i' && !strcmp(typeName, "int")))
				{
					int64 v;
					Jim_GetWide(interp, var, &v);
					Vdbe::Bind_Int64(stmt, i, v);
				}
				else
				{
					data = (char *)Jim_GetString(var, &n);
					Vdbe::Bind_Text(stmt, i, data, n, DESTRUCTOR_STATIC);
					Jim_IncrRefCount(var);
					p->Parms[parmsLength++] = var;
				}
			}
			else
				Vdbe::Bind_Null(stmt, i);
		}
	}
	p->Parms.length = parmsLength;
	*preStmt = p;

	return RC_OK;
}

static __device__ void DbReleaseStmt(TclContext *tctx, SqlPreparedStmt *preStmt, bool discard)
{
	// Free the bound string and blob parameters
	Jim_Interp *interp = tctx->Interp;
	for (int i = 0; i < preStmt->Parms.length; i++)
		Jim_DecrRefCount(interp, preStmt->Parms[i]);
	preStmt->Parms.length = 0;

	// If the cache is turned off, deallocated the statement
	if (tctx->MaxStmt <= 0 || discard)
		DbFreeStmt(preStmt);
	else
	{
		// Add the prepared statement to the beginning of the cache list.
		preStmt->Next = tctx->Stmts.data;
		preStmt->Prev = nullptr;
		if (tctx->Stmts.data)
			tctx->Stmts.data->Prev = preStmt;
		tctx->Stmts.data = preStmt;
		if (!tctx->StmtLast)
		{
			_assert(tctx->Stmts.length == 0);
			tctx->StmtLast = preStmt;
		}
		else
			_assert(tctx->Stmts.length > 0);
		tctx->Stmts.length++;

		// If we have too many statement in cache, remove the surplus from the end of the cache list.
		while (tctx->Stmts.length > tctx->MaxStmt)
		{
			SqlPreparedStmt *last = tctx->StmtLast;
			tctx->StmtLast = last->Prev;
			tctx->StmtLast->Next = nullptr;
			tctx->Stmts.length--;
			DbFreeStmt(last);
		}
	}
}

#pragma endregion

#pragma region EVAL

// dbEvalInit()
// dbEvalStep()
// dbEvalFinalize()
// dbEvalRowInfo()
// dbEvalColumnValue()

typedef struct DbEvalContext DbEvalContext;
struct DbEvalContext
{
	TclContext *Ctx;            // Database handle
	Jim_Obj *Sql;               // Object holding string zSql
	const char *SqlAsString;    // Remaining SQL to execute
	SqlPreparedStmt *PreStmt;   // Current statement
	int Cols;                   // Number of columns returned by pStmt
	Jim_Obj *Array;             // Name of array variable
	Jim_Obj **ColNames;         // Array of column names
};

// Release any cache of column names currently held as part of the DbEvalContext structure passed as the first argument.
static __device__ void DbReleaseColumnNames(DbEvalContext *p)
{
	Jim_Interp *interp = p->Ctx->Interp;
	if (p->ColNames)
	{
		for (int i = 0; i < p->Cols; i++)
			Jim_DecrRefCount(interp, p->ColNames[i]);
		_free(p->ColNames);
		p->ColNames = nullptr;
	}
	p->Cols = 0;
}

// Initialize a DbEvalContext structure.
//
// If pArray is not NULL, then it contains the name of a Jim array variable. The "*" member of this array is set to a list containing
// the names of the columns returned by the statement as part of each call to dbEvalStep(), in order from left to right. e.g. if the names 
// of the returned columns are a, b and c, it does the equivalent of the tcl command:
//
//     set ${pArray}(*) {a b c}
static __device__ void DbEvalInit(DbEvalContext *p, TclContext *tctx, Jim_Obj *sql, Jim_Obj *array_)
{
	memset(p, 0, sizeof(DbEvalContext));
	p->Ctx = tctx;
	p->SqlAsString = Jim_String(sql);
	p->Sql = sql;
	Jim_IncrRefCount(sql);
	if (array_)
	{
		p->Array = array_;
		Jim_IncrRefCount(array_);
	}
}

// Obtain information about the row that the DbEvalContext passed as the first argument currently points to.
static __device__ void DbEvalRowInfo(DbEvalContext *p, int *colsOut, Jim_Obj ***colNamesOut)
{
	Jim_Interp *interp = p->Ctx->Interp;
	// Compute column names
	if (!p->ColNames)
	{
		Vdbe *stmt = p->PreStmt->Stmt;
		int i;
		Jim_Obj **colNames = nullptr; // Array of column names
		int cols = p->Cols = Vdbe::Column_Count(stmt); // Number of columns returned by pStmt
		if (cols > 0 && (colNamesOut || p->Array))
		{
			colNames = (Jim_Obj **)_alloc(sizeof(Jim_Obj *) * cols);
			for (i = 0; i < cols; i++)
			{
				colNames[i] = dbTextToObj(interp, Vdbe::Column_Name(stmt, i));
				Jim_IncrRefCount(colNames[i]);
			}
			p->ColNames = colNames;
		}

		// If results are being stored in an array variable, then create the array(*) entry for that array
		if (p->Array)
		{
			Jim_Obj *colList = Jim_NewListObj(interp, colNames, cols);
			Jim_Obj *star = Jim_NewStringObj(interp, "*", -1);
			Jim_IncrRefCount(star);
			Jim_ObjSetVar2(interp, p->Array, star, colList); //: Jim_SetDictKeysVector(interp, p->Array, &star, 1, colList, 0);
			Jim_DecrRefCount(interp, star);
		}
	}

	if (colNamesOut)
		*colNamesOut = p->ColNames;
	if (colsOut)
		*colsOut = p->Cols;
}

// Return one of JIM_OK, JIM_BREAK or JIM_ERROR. If JIM_ERROR is returned, then an error message is stored in the interpreter before
// returning.
//
// A return value of JIM_OK means there is a row of data available. The data may be accessed using dbEvalRowInfo() and dbEvalColumnValue(). This
// is analogous to a return of SQLITE_ROW from sqlite3_step(). If JIM_BREAK is returned, then the SQL script has finished executing and there are
// no further rows available. This is similar to SQLITE_DONE.
static __device__ RC DbEvalStep(DbEvalContext *p)
{
	Jim_Interp *interp = p->Ctx->Interp;
	const char *prevSql = nullptr; // Previous value of p->zSql
	while (p->SqlAsString[0] || p->PreStmt)
	{
		RC rc;
		if (!p->PreStmt)
		{
			prevSql = (p->SqlAsString == prevSql ? nullptr : p->SqlAsString);
			rc = DbPrepareAndBind(p->Ctx, p->SqlAsString, &p->SqlAsString, &p->PreStmt);
			if (rc != RC_OK) return rc;
		}
		else
		{
			TclContext *tctx = p->Ctx;
			SqlPreparedStmt *preStmt = p->PreStmt;
			Vdbe *stmt = preStmt->Stmt;

			rc = stmt->Step();
			if (rc == RC_ROW)
				return RC_OK;
			if (p->Array)
				DbEvalRowInfo(p, 0, 0);
			rc = Vdbe::Reset(stmt);

			tctx->Steps = Vdbe::Stmt_Status(stmt, Vdbe::STMTSTATUS_FULLSCAN_STEP, true);
			tctx->Sorts = Vdbe::Stmt_Status(stmt, Vdbe::STMTSTATUS_SORT, true);
			tctx->Indexs = Vdbe::Stmt_Status(stmt, Vdbe::STMTSTATUS_AUTOINDEX, true);
			DbReleaseColumnNames(p);
			p->PreStmt = nullptr;

			if (rc != RC_OK)
			{
				// If a run-time error occurs, report the error and stop reading the SQL.
				DbReleaseStmt(tctx, preStmt, true);
#if _TEST
				if (p->Ctx->LegacyPrepare && rc == RC_SCHEMA && prevSql)
				{
					// If the runtime error was an SQLITE_SCHEMA, and the database handle is configured to use the legacy sqlite3_prepare() 
					// interface, retry prepare()/step() on the same SQL statement. This only happens once. If there is a second SQLITE_SCHEMA
					// error, the error will be returned to the caller.
					p->SqlAsString = prevSql;
					continue;
				}
#endif
				Jim_SetResult(interp, dbTextToObj(interp, (char *)DataEx::ErrMsg(tctx->Ctx)));
				return RC_ERROR;
			}
			else
				DbReleaseStmt(tctx, preStmt, false);
		}
	}

	// Finished
	return RC_DONE;
}

// Free all resources currently held by the DbEvalContext structure passed as the first argument. There should be exactly one call to this function
// for each call to dbEvalInit().
static __device__ void DbEvalFinalize(DbEvalContext *p)
{
	Jim_Interp *interp = p->Ctx->Interp;
	if (p->PreStmt)
	{
		Vdbe::Reset(p->PreStmt->Stmt);
		DbReleaseStmt(p->Ctx, p->PreStmt, false);
		p->PreStmt = nullptr;
	}
	if (p->Array)
	{
		Jim_DecrRefCount(interp, p->Array);
		p->Array = nullptr;
	}
	Jim_DecrRefCount(interp, p->Sql);
	DbReleaseColumnNames(p);
}

// Return a pointer to a Jim_Obj structure with ref-count 0 that contains the value for the iCol'th column of the row currently pointed to by
// the DbEvalContext structure passed as the first argument.
static __device__ Jim_Obj *DbEvalColumnValue(DbEvalContext *p, int colId)
{
	Jim_Interp *interp = p->Ctx->Interp;
	Vdbe *stmt = p->PreStmt->Stmt;
	switch (Vdbe::Column_Type(stmt, colId))
	{
	case TYPE_BLOB: {
		int bytes = Vdbe::Column_Bytes(stmt, colId);
		const void *blob = Vdbe::Column_Blob(stmt, colId);
		if (!blob) bytes = 0;
		return Jim_NewByteArrayObj(interp, (char *)blob, bytes); }
	case TYPE_INTEGER: {
		int64 v = Vdbe::Column_Int64(stmt, colId);
		//return (v >= -2147483647 && v <= 2147483647 ? Jim_NewIntObj(interp, v) : Jim_NewIntObj(interp, v)); }
		return Jim_NewIntObj(interp, v); }
	case TYPE_FLOAT: {
		return Jim_NewDoubleObj(interp, Vdbe::Column_Double(stmt, colId)); }
	case TYPE_NULL: {
		return dbTextToObj(interp, p->Ctx->NullText); }
	}
	return dbTextToObj(interp, (char *)Vdbe::Column_Text(stmt, colId));
}

//__inline static __device__ int Jim_ObjSetVar2(Jim_Interp *interp, Jim_Obj *nameObjPtr, Jim_Obj *keyObjPtr, Jim_Obj *valObjPtr) { return Jim_SetDictKeysVector(interp, nameObjPtr, &keyObjPtr, 1, valObjPtr, 0); }

// This function is part of the implementation of the command:
//
//   $db eval SQL ?ARRAYNAME? SCRIPT
static __device__ int DbEvalNextCmd(ClientData data[], Jim_Interp *interp, int result)
{
	int rc = result;

	// The first element of the data[] array is a pointer to a DbEvalContext structure allocated using Jim_Alloc(). The second element of data[]
	// is a pointer to a Jim_Obj containing the script to run for each row returned by the queries encapsulated in data[0].
	DbEvalContext *p = (DbEvalContext *)data[0];
	Jim_Obj *script = (Jim_Obj *)data[1];
	Jim_Obj *array = p->Array;

	while ((rc == RC_OK || rc == RC_ROW) && (rc = DbEvalStep(p)) == RC_OK)
	{
		int cols;
		Jim_Obj **colNames;
		DbEvalRowInfo(p, &cols, &colNames);
		for (int i = 0 ; i < cols; i++)
		{
			Jim_Obj *val = DbEvalColumnValue(p, i);
			if (!array)
				Jim_SetVariable(interp, colNames[i], val);
			else
				Jim_ObjSetVar2(interp, array, colNames[i], val);
		}

		// The required interpreter variables are now populated with the data  from the current row.
		// No NRE in Jim Tcl, so evaluate pScript directly and continue with the next iteration of this while(...) loop.
		rc = Jim_EvalObj(interp, script);
	}

	Jim_DecrRefCount(interp, script);
	DbEvalFinalize(p);
	Jim_Free(p);

	if (rc == RC_OK || rc == RC_DONE)
	{
		Jim_ResetResult(interp);
		rc = RC_OK;
	}
	return rc;
}

#pragma endregion

#pragma region DbObjCmd

__constant__ static const char *DB_strs[] = {
	"authorizer",         "backup",            "busy",
	"cache",              "changes",           "close",
	"collate",            "collation_needed",  "commit_hook",
	"complete",           "copy",              "enable_load_extension",
	"errorcode",          "eval",              "exists",
	"function",           "incrblob",          "interrupt",
	"last_insert_rowid",  "nullvalue",         "onecolumn",
	"profile",            "progress",          "rekey",
	"restore",            "rollback_hook",     "status",
	"timeout",            "total_changes",     "trace",
	"transaction",        "unlock_notify",     "update_hook",
	"version",            "wal_hook",          0
};
// don't leave trailing commas on DB_enum, it confuses the AIX xlc compiler
enum DB_enum {
	DB_AUTHORIZER,        DB_BACKUP,           DB_BUSY,
	DB_CACHE,             DB_CHANGES,          DB_CLOSE,
	DB_COLLATE,           DB_COLLATION_NEEDED, DB_COMMIT_HOOK,
	DB_COMPLETE,          DB_COPY,             DB_ENABLE_LOAD_EXTENSION,
	DB_ERRORCODE,         DB_EVAL,             DB_EXISTS,
	DB_FUNCTION,          DB_INCRBLOB,         DB_INTERRUPT,
	DB_LAST_INSERT_ROWID, DB_NULLVALUE,        DB_ONECOLUMN,
	DB_PROFILE,           DB_PROGRESS,         DB_REKEY,
	DB_RESTORE,           DB_ROLLBACK_HOOK,    DB_STATUS,
	DB_TIMEOUT,           DB_TOTAL_CHANGES,    DB_TRACE,
	DB_TRANSACTION,       DB_UNLOCK_NOTIFY,    DB_UPDATE_HOOK,
	DB_VERSION,           DB_WAL_HOOK
};

__constant__ static const char *TTYPE_strs[] = { "deferred",   "exclusive",  "immediate", nullptr };
enum TTYPE_enum { TTYPE_DEFERRED, TTYPE_EXCLUSIVE, TTYPE_IMMEDIATE };


// The "sqlite" command below creates a new Jim command for each connection it opens to an SQLite database.  This routine is invoked
// whenever one of those connection-specific commands is executed in Jim.  For example, if you run Jim code like this:
//
//       sqlite3 db1  "my_database"
//       db1 close
//
// The first command opens a connection to the "my_database" database and calls that connection "db1".  The second command causes this subroutine to be invoked.
static __device__ int DbObjCmd(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	if (argc < 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "SUBCOMMAND ...");
		return JIM_ERROR;
	}
	int choice;
	if (Jim_GetEnum(interp, args[1], DB_strs, &choice, "option", JIM_ERRMSG | JIM_ENUM_ABBREV))
		return JIM_ERROR;

	TclContext *p = (TclContext *)clientData;
	int rc = JIM_OK;
	switch ((enum DB_enum)choice)
	{
	case DB_AUTHORIZER: {
		//    $db authorizer ?CALLBACK?
		//
		// Invoke the given callback to authorize each SQL operation as it is compiled.  5 arguments are appended to the callback before it is invoked:
		//
		//   (1) The authorization type (ex: SQLITE_CREATE_TABLE, SQLITE_INSERT, ...)
		//   (2) First descriptive name (depends on authorization type)
		//   (3) Second descriptive name
		//   (4) Name of the database (ex: "main", "temp")
		//   (5) Name of trigger that is doing the access
		//
		// The callback should return on of the following strings: SQLITE_OK, SQLITE_IGNORE, or SQLITE_DENY.  Any other return value is an error.
		//
		// If this method is invoked with no arguments, the current authorization callback string is returned.
#ifdef OMIT_AUTHORIZATION
		Jim_AppendResult("authorization not available in this build", 0);
		return JIM_ERROR;
#else
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return JIM_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Auth)
				Jim_SetResultString(interp, p->Auth, -1);
		}
		else
		{
			if (p->Auth)
				Jim_Free(p->Auth);
			int len;
			const char *auth = Jim_GetString(args[2], &len);
			if (auth && len > 0)
			{
				p->Auth = (char *)Jim_Alloc(len + 1);
				memcpy(p->Auth, auth, len + 1);
			}
			else
				p->Auth = nullptr;
			if (p->Auth)
			{
				p->Interp = interp;
				Auth::SetAuthorizer(p->Ctx, AuthCallback, p);
			}
			else
				Auth::SetAuthorizer(p->Ctx, nullptr, nullptr);
		}
#endif
		return RC_OK; }

	case DB_BACKUP: {
		//    $db backup ?DATABASE? FILENAME
		//
		// Open or create a database file named FILENAME.  Transfer the content of local database DATABASE (default: "main") into the FILENAME database.
		const char *srcDb;
		const char *destFile;
		if (argc == 3)
		{
			srcDb = "main";
			destFile = Jim_String(args[2]);
		}
		else if (argc == 4)
		{
			srcDb = Jim_String(args[2]);
			destFile = Jim_String(args[3]);
		}
		else
		{
			Jim_WrongNumArgs(interp, 2, args, "?DATABASE? FILENAME");
			return JIM_ERROR;
		}

		Context *destCtx;
		rc = DataEx::Open(destFile, &destCtx);
		if (rc != RC_OK)
		{
			Jim_SetResultFormatted(interp, "cannot open target database: %s", DataEx::ErrMsg(destCtx));
			DataEx::Close(destCtx);
			return JIM_ERROR;
		}
		Backup *backup = Backup::Init(destCtx, "main", p->Ctx, srcDb);
		if (!backup)
		{
			Jim_SetResultFormatted(interp, "backup failed: %s", DataEx::ErrMsg(destCtx));
			DataEx::Close(destCtx);
			return JIM_ERROR;
		}
		while ((rc = backup->Step(100)) == RC_OK) { }
		Backup::Finish(backup);
		if (rc == RC_DONE)
			rc = JIM_OK;
		else
		{
			Jim_SetResultFormatted(interp, "backup failed: %s", DataEx::ErrMsg(destCtx));
			rc = JIM_ERROR;
		}
		DataEx::Close(destCtx);
		return rc; }

	case DB_BUSY: {
		//    $db busy ?CALLBACK?
		//
		// Invoke the given callback if an SQL statement attempts to open a locked database file.
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "CALLBACK");
			return JIM_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Busy)
				Jim_SetResultString(interp, p->Busy, -1);
		}
		else
		{
			if (p->Busy)
				Jim_Free(p->Busy);
			int len;
			const char *busy = Jim_GetString(args[2], &len);
			if (busy && len > 0)
			{
				p->Busy = (char *)Jim_Alloc(len + 1);
				memcpy(p->Busy, busy, len + 1);
			}
			else
				p->Busy = nullptr;
			if (p->Busy)
			{
				p->Interp = interp;
				DataEx::BusyHandler(p->Ctx, DbBusyHandler, p);
			}
			else
				DataEx::BusyHandler(p->Ctx, nullptr, nullptr);
		}
		return rc; }

	case DB_CACHE: {
		//     $db cache flush
		//     $db cache size n
		//
		// Flush the prepared statement cache, or set the maximum number of cached statements.
		if (argc <= 2)
		{
			Jim_WrongNumArgs(interp, 1, args, "cache option ?arg?");
			return JIM_ERROR;
		}
		const char *subCmd = Jim_String(args[2]);
		if (subCmd[0] == 'f' && !strcmp(subCmd, "flush"))
		{
			if (argc != 3)
			{
				Jim_WrongNumArgs(interp, 2, args, "flush");
				return JIM_ERROR;
			}
			else
				FlushStmtCache(p);
		}
		else if (subCmd[0] == 's' && !strcmp(subCmd, "size"))
		{
			if (argc != 4)
			{
				Jim_WrongNumArgs(interp, 2, args, "size n");
				return JIM_ERROR;
			}
			else
			{
				int64 n;
				if (Jim_GetWide(interp, args[3], &n) == JIM_ERROR)
				{
					Jim_SetResultFormatted(interp, "cannot convert \"%#s\" to integer", args[3]);
					return JIM_ERROR;
				}
				else
				{
					if (n < 0)
					{
						FlushStmtCache(p);
						n = 0;
					}
					else if (n > MAX_PREPARED_STMTS)
						n = MAX_PREPARED_STMTS;
					p->MaxStmt = (int)n;
				}
			}
		}
		else
		{
			Jim_SetResultFormatted(interp, "bad option \"%#s\": must be flush or size", args[2]);
			return JIM_ERROR;
		}
		return rc; }

	case DB_CHANGES: {
		//     $db changes
		//
		// Return the number of rows that were modified, inserted, or deleted by the most recent INSERT, UPDATE or DELETE statement, not including 
		// any changes made by trigger programs.
		if (argc != 2)
		{
			Jim_WrongNumArgs(interp, 2, args, "");
			return JIM_ERROR;
		}
		Jim_SetResultInt(interp, DataEx::CtxChanges(p->Ctx));
		return rc; }

	case DB_CLOSE: {
		//    $db close
		//
		// Shutdown the database
		Jim_DeleteCommand(interp, Jim_String(args[0]));
		return rc; }

	case DB_COLLATE: {
		//     $db collate NAME SCRIPT
		//
		// Create a new SQL collation function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
		if (argc != 4)
		{
			Jim_WrongNumArgs(interp, 2, args, "NAME SCRIPT");
			return JIM_ERROR;
		}
		const char *name = Jim_String(args[2]);
		int scriptLength;
		const char *script = Jim_GetString(args[3], &scriptLength);
		SqlCollate *collate = (SqlCollate *)Jim_Alloc(sizeof(*collate) + scriptLength + 1);
		if (!collate) return JIM_ERROR;
		collate->Interp = interp;
		collate->Next = p->Collates;
		collate->Script = (char *)&collate[1];
		p->Collates = collate;
		memcpy(collate->Script, script, scriptLength+1);
		if (DataEx::CreateCollation(p->Ctx, name, TEXTENCODE_UTF8, collate, TclSqlCollate))
		{
			Jim_SetResultString(interp, (char *)DataEx::ErrMsg(p->Ctx), -1);
			return JIM_ERROR;
		}
		return rc; }

	case DB_COLLATION_NEEDED: {
		//     $db collation_needed SCRIPT
		//
		// Create a new SQL collation function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "SCRIPT");
			return JIM_ERROR;
		}
		if (p->CollateNeeded)
			Jim_DecrRefCount(interp, p->CollateNeeded);
		p->CollateNeeded = Jim_DuplicateObj(interp, args[2]);
		Jim_IncrRefCount(p->CollateNeeded);
		DataEx::CollationNeeded(p->Ctx, p, TclCollateNeeded);
		return rc; }

	case DB_COMMIT_HOOK: {
		//    $db commit_hook ?CALLBACK?
		//
		// Invoke the given callback just before committing every SQL transaction. If the callback throws an exception or returns non-zero, then the
		// transaction is aborted.  If CALLBACK is an empty string, the callback is disabled.
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return JIM_ERROR;
		} else if (argc == 2)
		{
			if (p->Commit)
				Jim_SetResultString(interp, p->Commit, -1);
		}
		else
		{
			if (p->Commit)
				Jim_Free(p->Commit);
			int len;
			const char *commit = Jim_GetString(args[2], &len);
			if (commit && len > 0)
			{
				p->Commit = (char *)Jim_Alloc(len + 1);
				memcpy(p->Commit, commit, len+1);
			}
			else
				p->Commit = nullptr;
			if (p->Commit)
			{
				p->Interp = interp;
				DataEx::CommitHook(p->Ctx, DbCommitHandler, p);
			}
			else
				DataEx::CommitHook(p->Ctx, nullptr, nullptr);
		}
		return rc; }

	case DB_COMPLETE: {
		//    $db complete SQL
		//
		// Return TRUE if SQL is a complete SQL statement.  Return FALSE if additional lines of input are needed.  This is similar to the
		// built-in "info complete" command of Jim.
#ifndef OMIT_COMPLETE
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "SQL");
			return JIM_ERROR;
		}
		bool isComplete = Parse::Complete(Jim_String(args[2]));
		Jim_SetResultBool(interp, isComplete);
#endif
		return rc; }

	case DB_COPY: {
		//    $db copy conflict-algorithm table filename ?SEPARATOR? ?NULLINDICATOR?
		//
		// Copy data into table from filename, optionally using SEPARATOR as column separators.  If a column contains a null string, or the
		// value of NULLINDICATOR, a NULL is inserted for the column. conflict-algorithm is one of the sqlite conflict algorithms:
		//    rollback, abort, fail, ignore, replace
		// On success, return the number of lines processed, not necessarily same as 'db changes' due to conflict-algorithm selected.
		//
		// This code is basically an implementation/enhancement of the sqlite3 shell.c ".import" command.
		//
		// This command usage is equivalent to the sqlite2.x COPY statement, which imports file data into a table using the PostgreSQL COPY file format:
		//   $db copy $conflit_algo $table_name $filename \t \\N
		if (argc < 5 || argc > 7)
		{
			Jim_WrongNumArgs(interp, 2, args, "CONFLICT-ALGORITHM TABLE FILENAME ?SEPARATOR? ?NULLINDICATOR?");
			return JIM_ERROR;
		}
		int i;
		const char *sep = (argc >= 6 ? Jim_String(args[5]) : "\t");
		const char *null = (argc >= 7 ? Jim_String(args[6]) : "");
		const char *conflict = Jim_String(args[2]); // The conflict algorithm to use
		const char *table = Jim_String(args[3]); // Insert data into this table
		const char *file = Jim_String(args[4]); // The file from which to extract data
		int sepLength = strlen(sep); // Number of bytes in zSep[]
		int nullLength = strlen(null); // Number of bytes in zNull[]
		if (sepLength == 0)
		{
			Jim_SetResultString(interp, "Error: non-null separator required for copy", -1);
			return JIM_ERROR;
		}
		if (strcmp(conflict, "rollback") && strcmp(conflict, "abort") && strcmp(conflict, "fail") && strcmp(conflict, "ignore") && strcmp(conflict, "replace"))
		{
			Jim_SetResultFormatted(interp, "Error: \"%s\", conflict-algorithm must be one of: rollback, abort, fail, ignore, or replace", conflict);
			return JIM_ERROR;
		}
		char *sql = _mprintf("SELECT * FROM '%q'", table); // An SQL statement
		if (!sql)
		{
			Jim_SetResultFormatted(interp, "Error: no such table: %s", table);
			return JIM_ERROR;
		}
		int bytes = strlen(sql); // Number of bytes in an SQL string
		Vdbe *stmt; // A statement
		rc = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
		_free(sql);
		int cols;// Number of columns in the table
		if (rc)
		{
			Jim_SetResultFormatted(interp, "Error: %s", DataEx::ErrMsg(p->Ctx));
			cols = 0;
		}
		else
			cols = Vdbe::Column_Count(stmt);
		Vdbe::Finalize(stmt);
		if (cols == 0)
			return JIM_ERROR;
		sql = (char *)_alloc(bytes + 50 + cols*2);
		if (!sql)
		{
			Jim_SetResultString(interp, "Error: can't malloc()", -1);
			return JIM_ERROR;
		}
		snprintf(sql, bytes+50, "INSERT OR %q INTO '%q' VALUES(?", conflict, table);
		int j = strlen(sql);
		for (i = 1; i < cols; i++)
		{
			sql[j++] = ',';
			sql[j++] = '?';
		}
		sql[j++] = ')';
		sql[j] = 0;
		rc = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
		_free(sql);
		if (rc)
		{
			Jim_SetResultFormatted(interp, "Error: %s", DataEx::ErrMsg(p->Ctx));
			Vdbe::Finalize(stmt);
			return JIM_ERROR;
		}
		FILE *in = _fopen(file, "rb"); // The input file
		if (!in)
		{
			Jim_SetResultFormatted(interp, "Error: cannot open file: %s", file);
			Vdbe::Finalize(stmt);
			return JIM_ERROR;
		}
		char **colNames = (char **)malloc(sizeof(colNames[0]) * (cols+1)); // zLine[] broken up into columns
		if (!colNames)
		{
			Jim_SetResultString(interp, "Error: can't malloc()", -1);
			_fclose(in);
			return JIM_ERROR;
		}
		DataEx::Exec(p->Ctx, "BEGIN", 0, 0, 0);
		char *commit = "COMMIT"; // How to commit changes
		char *line; // A single line of input from the file
		int lineno = 0; // Line number of input file
		char lineNum[80]; // Line number print buffer
		//Jim_Obj *result; // interp result
		while ((line = LocalGetLine(0, in)) != 0)
		{
			char *z;
			lineno++;
			colNames[0] = line;
			for (i = 0, z = line; *z; z++)
			{
				if (*z == sep[0] && !strncmp(z, sep, sepLength))
				{
					*z = 0;
					i++;
					if (i < cols)
					{
						colNames[i] = &z[sepLength];
						z += sepLength-1;
					}
				}
			}
			if (i+1 != cols)
			{
				int errLength = strlen(file) + 200;
				char *err = (char *)malloc(errLength);
				if (err)
				{
					snprintf(err, errLength, "Error: %s line %d: expected %d columns of data but found %d", file, lineno, cols, i+1);
					Jim_SetResultString(interp, err, -1);
					free(err);
				}
				commit = "ROLLBACK";
				break;
			}
			for (i = 0; i < cols; i++)
			{
				// check for null data, if so, bind as null
				if ((nullLength > 0 && !strcmp(colNames[i], null)) || !strlen(colNames[i]))
					Vdbe::Bind_Null(stmt, i+1);
				else
					Vdbe::Bind_Text(stmt, i+1, colNames[i], -1, DESTRUCTOR_STATIC);
			}
			stmt->Step();
			rc = Vdbe::Reset(stmt);
			free(line);
			if (rc != RC_OK)
			{
				Jim_SetResultFormatted(interp, "Error: %s", DataEx::ErrMsg(p->Ctx));
				commit = "ROLLBACK";
				break;
			}
		}
		free(colNames);
		_fclose(in);
		Vdbe::Finalize(stmt);
		DataEx::Exec(p->Ctx, commit, 0, 0, 0);

		if (commit[0] == 'C')
		{
			// success, set result as number of lines processed
			Jim_SetResultInt(interp, lineno);
			rc = JIM_OK;
		}
		else
		{
			// failure, append lineno where failed
			snprintf(lineNum, sizeof(lineNum), "%d", lineno);
			Jim_AppendStrings(interp, Jim_GetResult(interp), ", failed while processing line: ", lineNum, nullptr);
			rc = JIM_ERROR;
		}
		return rc; }

	case DB_ENABLE_LOAD_EXTENSION: {
		//    $db enable_load_extension BOOLEAN
		//
		// Turn the extension loading feature on or off.  It if off by default.
#ifndef OMIT_LOAD_EXTENSION
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "BOOLEAN");
			return JIM_ERROR;
		}
		bool onoff;
		if (Jim_GetBoolean(interp, args[2], &onoff))
			return JIM_ERROR;
		DataEx::EnableLoadExtension(p->Ctx, onoff);
#else
		Jim_SetResultString(interp, "extension loading is turned off at compile-time", -1);
		return JIM_ERROR;
#endif
		return rc; }

	case DB_ERRORCODE: {
		//    $db errorcode
		//
		// Return the numeric error code that was returned by the most recent call to sqlite3_exec().
		Jim_SetResultInt(interp, DataEx::ErrCode(p->Ctx));
		return rc; }

	case DB_EXISTS: 
	case DB_ONECOLUMN: {
		//    $db exists $sql
		//    $db onecolumn $sql
		//
		// The onecolumn method is the equivalent of:
		//     lindex [$db eval $sql] 0
		DbEvalContext sEval;
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "SQL");
			return JIM_ERROR;
		}

		DbEvalInit(&sEval, p, args[2], nullptr);
		rc = DbEvalStep(&sEval);
		if (choice == DB_ONECOLUMN)
		{
			if (rc == JIM_OK)
				Jim_SetResult(interp, DbEvalColumnValue(&sEval, 0));
			else if (rc == JIM_BREAK)
				Jim_ResetResult(interp);
		}
		else if (rc == JIM_BREAK || rc == JIM_OK)
			Jim_SetResultBool(interp, (rc == JIM_OK));
		DbEvalFinalize(&sEval);

		if (rc == JIM_BREAK)
			rc = JIM_OK;
		return rc; }

	case DB_EVAL: {
		//    $db eval $sql ?array? ?{  ...code... }?
		//
		// The SQL statement in $sql is evaluated.  For each row, the values are placed in elements of the array named "array" and ...code... is executed.
		// If "array" and "code" are omitted, then no callback is every invoked. If "array" is an empty string, then the values are placed in variables
		// that have the same name as the fields extracted by the query.
		if (argc < 3 || argc > 5)
		{
			Jim_WrongNumArgs(interp, 2, args, "SQL ?ARRAY-NAME? ?SCRIPT?");
			return JIM_ERROR;
		}

		if (argc == 3)
		{
			DbEvalContext sEval;
			Jim_Obj *ret = Jim_NewListObj(interp, nullptr, 0);
			Jim_IncrRefCount(ret);
			DbEvalInit(&sEval, p, args[2], nullptr);

			while ((rc = DbEvalStep(&sEval)) == RC_OK)
			{
				int cols;
				DbEvalRowInfo(&sEval, &cols, nullptr);
				for (int i = 0; i < cols; i++)
					Jim_ListAppendElement(interp, ret, DbEvalColumnValue(&sEval, i));
			}
			DbEvalFinalize(&sEval);
			if (rc == RC_DONE)
			{
				Jim_SetResult(interp, ret);
				rc = RC_OK;
			}
			Jim_DecrRefCount(interp, ret);
		}
		else
		{
			ClientData cd[2];
			Jim_Obj *array = (argc == 5 && Jim_Length(args[3]) ? args[3] : nullptr);
			Jim_Obj *script = args[argc-1];
			Jim_IncrRefCount(script);

			DbEvalContext *p2 = (DbEvalContext *)Jim_Alloc(sizeof(DbEvalContext));
			DbEvalInit(p2, p, args[2], array);

			cd[0] = (ClientData)p2;
			cd[1] = (ClientData)script;
			rc = DbEvalNextCmd(cd, interp, RC_OK);
		}
		return rc; }

	case DB_FUNCTION: {
		//     $db function NAME [-argcount N] SCRIPT
		//
		// Create a new SQL function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
		Jim_Obj *script;
		int args4 = -1;
		if (argc == 6)
		{
			const char *z = Jim_String(args[3]);
			int n = strlen(z);
			if (n > 2 && !strncmp(z, "-argcount",n))
			{
				if (Jim_GetInt(interp, args[4], &args4)) return JIM_ERROR;
				if (args4 < 0)
				{
					Jim_SetResultString(interp, "number of arguments must be non-negative", -1);
					return JIM_ERROR;
				}
			}
			script = args[5];
		}
		else if (argc != 4)
		{
			Jim_WrongNumArgs(interp, 2, args, "NAME [-argcount N] SCRIPT");
			return JIM_ERROR;
		}
		else
			script = args[3];
		const char *name = Jim_String(args[2]);
		SqlFunc *func = FindSqlFunc(p, name);
		if (!func) return JIM_ERROR;
		if (func->Script)
			Jim_DecrRefCount(interp, func->Script);
		func->Script = script;
		Jim_IncrRefCount(script);
		func->UseEvalObjv = SafeToUseEvalObjv(interp, script);
		rc = DataEx::CreateFunction(p->Ctx, name, args4, TEXTENCODE_UTF8, func, TclSqlFunc, 0, 0);
		if (rc != RC_OK)
		{
			rc = JIM_ERROR;
			Jim_SetResultString(interp, (char *)DataEx::ErrMsg(p->Ctx), -1);
		}
		return rc; }

	case DB_INCRBLOB: {
		//     $db incrblob ?-readonly? ?DB? TABLE COLUMN ROWID
#ifdef OMIT_INCRBLOB
		Jim_SetResultString(interp, "incrblob not available in this build", -1);
		return JIM_ERROR;
#else
		// Check for the -readonly option
		int isReadonly = (argc > 3 && !strcmp(Jim_String(args[2]), "-readonly") ? 1 : 0);
		if (argc != (5+isReadonly) && argc != (6+isReadonly))
		{
			Jim_WrongNumArgs(interp, 2, args, "?-readonly? ?DB? TABLE COLUMN ROWID");
			return JIM_ERROR;
		}

		const char *db = (argc == (6+isReadonly) ? Jim_String(args[2]) : "main");
		const char *table = Jim_String(args[argc-3]);
		const char *column = Jim_String(args[argc-2]);
		int64 rows;
		rc = Jim_GetWide(interp, args[argc-1], &rows);

		if (rc == JIM_OK)
			rc = CreateIncrblobChannel(interp, p, db, table, column, rows, isReadonly);
#endif
		return rc; }

	case DB_INTERRUPT: {
		//     $db interrupt
		//
		// Interrupt the execution of the inner-most SQL interpreter.  This causes the SQL statement to return an error of SQLITE_INTERRUPT.
		DataEx::Interrupt(p->Ctx);
		return rc; }

	case DB_NULLVALUE: {
		//     $db nullvalue ?STRING?
		//
		// Change text used when a NULL comes back from the database. If ?STRING? is not present, then the current string used for NULL is returned.
		// If STRING is present, then STRING is returned.
		if (argc != 2 && argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "NULLVALUE");
			return JIM_ERROR;
		}
		if (argc == 3)
		{
			int len;
			const char *null = Jim_GetString(args[2], &len);
			if (p->NullText)
				Jim_Free(p->NullText);
			if (null && len > 0)
			{
				p->NullText = (char *)Jim_Alloc(len + 1);
				memcpy(p->NullText, null, len);
				p->NullText[len] = '\0';
			}
			else
				p->NullText = nullptr;
		}
		Jim_SetResult(interp, dbTextToObj(interp, p->NullText));
		return rc; }

	case DB_LAST_INSERT_ROWID: {
		//     $db last_insert_rowid 
		//
		// Return an integer which is the ROWID for the most recent insert.
		if (argc != 2)
		{
			Jim_WrongNumArgs(interp, 2, args, "");
			return JIM_ERROR;
		}
		int64 rowid = DataEx::CtxLastInsertRowid(p->Ctx);
		Jim_SetResultInt(interp, rowid);
		return rc; }

							   // The DB_ONECOLUMN method is implemented together with DB_EXISTS.

	case DB_PROGRESS: {
		//    $db progress ?N CALLBACK?
		// 
		// Invoke the given callback every N virtual machine opcodes while executing queries.
		if (argc == 2)
		{
			if (p->Progress)
				Jim_AppendString(interp, Jim_GetResult(interp), p->Progress, -1);
		}
		else if (argc == 4)
		{
			int N;
			if (Jim_GetInt(interp, args[2], &N) != JIM_OK)
				return JIM_ERROR;
			if (p->Progress)
				Jim_Free(p->Progress);
			int len;
			const char *progress = Jim_GetString(args[3], &len);
			if (progress && len > 0)
			{
				p->Progress = (char *)Jim_Alloc(len + 1);
				memcpy(p->Progress, progress, len+1);
			}
			else
				p->Progress = nullptr;
#ifndef OMIT_PROGRESS_CALLBACK
			if (p->Progress)
			{
				p->Interp = interp;
				DataEx::ProgressHandler(p->Ctx, N, DbProgressHandler, p);
			}
			else
				DataEx::ProgressHandler(p->Ctx, 0, nullptr, nullptr);
#endif
		}
		else
		{
			Jim_WrongNumArgs(interp, 2, args, "N CALLBACK");
			return JIM_ERROR;
		}
		return rc; }

	case DB_PROFILE: {
		//    $db profile ?CALLBACK?
		//
		// Make arrangements to invoke the CALLBACK routine after each SQL statement that has run.  The text of the SQL and the amount of elapse time are
		// appended to CALLBACK before the script is run.
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return JIM_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Profile)
				Jim_SetResultString(interp, p->Profile, -1);
		}
		else
		{
			if (p->Profile)
				Jim_Free(p->Profile);
			int len;
			const char *profile = Jim_GetString(args[2], &len);
			if (profile && len > 0)
			{
				p->Profile = (char *)Jim_Alloc(len + 1);
				memcpy(p->Profile, profile, len+1);
			}
			else
				p->Profile = nullptr;
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
			if (p->Profile)
			{
				p->Interp = interp;
				DataEx::Profile(p->Ctx, DbProfileHandler, p);
			}
			else
				DataEx::Profile(p->Ctx, nullptr, nullptr);
#endif
		}
		return rc; }

	case DB_REKEY: {
		//     $db rekey KEY
		//
		// Change the encryption key on the currently open database.
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "KEY");
			return JIM_ERROR;
		}
#ifdef HAS_CODEC
		int nKey;
		void *pKey = Jim_GetByteArray(args[2], &nKey);
		rc2 = sqlite3_rekey(p->Ctx, pKey, nKey);
		if (rc2)
		{
			Jim_SetResultString(interp, DataEx::ErrStr(rc2), -1);
			rc = JIM_ERROR;
		}
#endif
		return rc; }
	case DB_RESTORE: {
		//    $db restore ?DATABASE? FILENAME
		//
		// Open a database file named FILENAME.  Transfer the content  of FILENAME into the local database DATABASE (default: "main").
		const char *srcFile;
		const char *destDb;
		if (argc == 3)
		{
			destDb = "main";
			srcFile = Jim_String(args[2]);
		}
		else if (argc == 4)
		{
			destDb = Jim_String(args[2]);
			srcFile = Jim_String(args[3]);
		}
		else
		{
			Jim_WrongNumArgs(interp, 2, args, "?DATABASE? FILENAME");
			return JIM_ERROR;
		}
		Context *src;
		rc = DataEx::Open_v2(srcFile, &src, VSystem::OPEN::OPEN_READONLY, 0);
		if (rc != RC_OK)
		{
			Jim_SetResultFormatted(interp, "cannot open source database: %s", DataEx::ErrMsg(src));
			DataEx::Close(src);
			return JIM_ERROR;
		}
		Backup *backup = Backup::Init(p->Ctx, destDb, src, "main");
		if (!backup)
		{
			Jim_SetResultFormatted(interp, "restore failed: %s", DataEx::ErrMsg(p->Ctx));
			DataEx::Close(src);
			return JIM_ERROR;
		}
		int timeout = 0;
		while ((rc = backup->Step(100)) == RC_OK || rc == RC_BUSY)
		{
			if (rc == RC_BUSY)
			{
				if (timeout++ >= 3) break;
				DataEx::Sleep(100);
			}
		}
		Backup::Finish(backup);
		if (rc == RC_DONE)
			rc = JIM_OK;
		else if (rc == RC_BUSY || rc == RC_LOCKED)
		{
			Jim_SetResultString(interp, "restore failed: source database busy", -1);
			rc = JIM_ERROR;
		}
		else
		{
			Jim_SetResultFormatted(interp, "restore failed: %s", DataEx::ErrMsg(p->Ctx));
			rc = JIM_ERROR;
		}
		DataEx::Close(src);
		return rc; }

	case DB_STATUS: {
		//     $db status (step|sort|autoindex)
		//
		// Display SQLITE_STMTSTATUS_FULLSCAN_STEP or SQLITE_STMTSTATUS_SORT for the most recent eval.
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "(step|sort|autoindex)");
			return JIM_ERROR;
		}
		const char *op = Jim_String(args[2]);
		int v;
		if (!strcmp(op, "step")) v = p->Steps;
		else if (!strcmp(op, "sort")) v = p->Sorts;
		else if (!strcmp(op, "autoindex")) v = p->Indexs;
		else
		{
			Jim_SetResultString(interp, "bad argument: should be autoindex, step, or sort", -1);
			return JIM_ERROR;
		}
		Jim_SetResultInt(interp, v);
		return rc; }

	case DB_TIMEOUT: {
		//     $db timeout MILLESECONDS
		//
		// Delay for the number of milliseconds specified when a file is locked.
		if (argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "MILLISECONDS");
			return JIM_ERROR;
		}
		int ms;
		if (Jim_GetInt(interp, args[2], &ms)) return JIM_ERROR;
		DataEx::BusyTimeout(p->Ctx, ms);
		return rc; }

	case DB_TOTAL_CHANGES: {
		//     $db total_changes
		//
		// Return the number of rows that were modified, inserted, or deleted since the database handle was created.
		if (argc != 2)
		{
			Jim_WrongNumArgs(interp, 2, args, "");
			return JIM_ERROR;
		}
		Jim_SetResultInt(interp, DataEx::CtxTotalChanges(p->Ctx));
		return rc; }

	case DB_TRACE: {
		//    $db trace ?CALLBACK?
		//
		// Make arrangements to invoke the CALLBACK routine for each SQL statement that is executed.  The text of the SQL is appended to CALLBACK before
		// it is executed.
		if (argc > 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return JIM_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Trace)
				Jim_AppendString(interp, Jim_GetResult(interp), p->Trace, 1);
		}
		else
		{
			if (p->Trace)
				Jim_Free(p->Trace);
			int len;
			const char *trace = Jim_GetString(args[2], &len);
			if (trace && len > 0)
			{
				p->Trace = (char *)Jim_Alloc(len + 1);
				memcpy(p->Trace, trace, len+1);
			}
			else
				p->Trace = nullptr;
#if !defined(OMIT_TRACE) && !defined(OMIT_FLOATING_POINT)
			if (p->Trace)
			{
				p->Interp = interp;
				DataEx::Trace(p->Ctx, DbTraceHandler, p);
			}
			else
				DataEx::Trace(p->Ctx, nullptr, nullptr);
#endif
		}
		return rc; }

	case DB_TRANSACTION: {
		//    $db transaction [-deferred|-immediate|-exclusive] SCRIPT
		//
		// Start a new transaction (if we are not already in the midst of a transaction) and execute the JIM script SCRIPT.  After SCRIPT
		// completes, either commit the transaction or roll it back if SCRIPT throws an exception.  Or if no new transation was started, do nothing.
		// pass the exception on up the stack.
		//
		// This command was inspired by Dave Thomas's talk on Ruby at the 2005 O'Reilly Open Source Convention (OSCON).
		if (argc != 3 && argc != 4)
		{
			Jim_WrongNumArgs(interp, 2, args, "[TYPE] SCRIPT");
			return JIM_ERROR;
		}

		const char *begin = "SAVEPOINT _tcl_transaction";
		if (p->Transactions == 0 && argc == 4)
		{
			int ttype;
			if (Jim_GetEnum(interp, args[2], TTYPE_strs, &ttype, "transaction type", JIM_ERRMSG | JIM_ENUM_ABBREV))
				return JIM_ERROR;
			switch ((TTYPE_enum)ttype)
			{
			case TTYPE_DEFERRED: break; // no-op
			case TTYPE_EXCLUSIVE: begin = "BEGIN EXCLUSIVE"; break;
			case TTYPE_IMMEDIATE: begin = "BEGIN IMMEDIATE"; break;
			}
		}
		Jim_Obj *script = args[argc-1];

		// Run the SQLite BEGIN command to open a transaction or savepoint.
		p->DisableAuth++;
		rc = DataEx::Exec(p->Ctx, begin, 0, 0, 0);
		p->DisableAuth--;
		if (rc != RC_OK)
		{
			Jim_SetResultString(interp, DataEx::ErrMsg(p->Ctx), -1);
			return JIM_ERROR;
		}
		p->Transactions++;

		// No NRE in Jim Tcl, so evaluate the script directly, then call function DbTransPostCmd() to commit (or rollback) the transaction or savepoint.
		rc = DbTransPostCmd(&clientData, interp, Jim_EvalObj(interp, script));
		return rc; }

	case DB_UNLOCK_NOTIFY: {
		//    $db unlock_notify ?script?
#ifndef ENABLE_UNLOCK_NOTIFY
		Jim_SetResultString(interp, "unlock_notify not available in this build", -1);
		rc = JIM_ERROR;
#else
		if (argc != 2 && argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?SCRIPT?");
			rc = JIM_ERROR;
		}
		else
		{
			if (p->UnlockNotify)
			{
				Jim_DecrRefCount(interp, p->UnlockNotify);
				p->UnlockNotify = nullptr;
			}

			void (*notify)(void **, int) = nullptr;
			void *notifyArg = nullptr;
			if (argc == 3)
			{
				notify = DbUnlockNotify;
				notifyArg = (void *)p;
				p->UnlockNotify = args[2];
				Jim_IncrRefCount(p->UnlockNotify);
			}

			if (DataEx::UnlockNotify(p->Ctx, notify, notifyArg))
			{
				Jim_SetResultString(interp, DataEx::ErrMsg(p->Ctx), -1);
				rc = JIM_ERROR;
			}
		}
#endif
		return rc; }

	case DB_WAL_HOOK: 
	case DB_UPDATE_HOOK: 
	case DB_ROLLBACK_HOOK: {
		//    $db wal_hook ?script?
		//    $db update_hook ?script?
		//    $db rollback_hook ?script?
		// set ppHook to point at pUpdateHook or pRollbackHook, depending on whether [$db update_hook] or [$db rollback_hook] was invoked.
		Jim_Obj **hook; 
		if (choice == DB_UPDATE_HOOK) hook = &p->UpdateHook;
		else if (choice == DB_WAL_HOOK) hook = &p->WalHook;
		else hook = &p->RollbackHook;

		if (argc != 2 && argc != 3)
		{
			Jim_WrongNumArgs(interp, 2, args, "?SCRIPT?");
			return JIM_ERROR;
		}
		if (*hook)
		{
			Jim_SetResult(interp, *hook);
			if (argc == 3)
			{
				Jim_DecrRefCount(interp, *hook);
				*hook = nullptr;
			}
		}
		if (argc == 3)
		{
			_assert(!(*hook));
			if (Jim_Length(args[2]) > 0)
			{
				*hook = (Jim_Obj *)args[2];
				Jim_IncrRefCount(*hook);
			}
		}
		DataEx::UpdateHook(p->Ctx, (p->UpdateHook?DbUpdateHandler:nullptr), p);
		DataEx::RollbackHook(p->Ctx, (p->RollbackHook?DbRollbackHandler:nullptr), p);
		DataEx::WalHook(p->Ctx, (p->WalHook?DbWalHandler:nullptr), p);
		return rc; }
	case DB_VERSION: {
		//    $db version
		//
		// Return the version string for this database.
		Jim_SetResultString(interp, LIBCU_VERSION, -1);
		return rc; }

	} // End of the SWITCH statement
	return rc;
}

#pragma endregion

#pragma region DbMain

//   sqlite3 DBNAME FILENAME ?-vfs VFSNAME? ?-key KEY? ?-readonly BOOLEAN?
//                           ?-create BOOLEAN? ?-nomutex BOOLEAN?
//
// This is the main Jim command.  When the "sqlite" Jim command is invoked, this routine runs to process that command.
//
// The first argument, DBNAME, is an arbitrary name for a new database connection.  This command creates a new command named
// DBNAME that is used to control that connection.  The database connection is deleted when the DBNAME command is deleted.
//
// The second argument is the name of the database file.
static __device__ int DbMain(void *cd, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
	// In normal use, each JIM interpreter runs in a single thread.  So by default, we can turn of mutexing on SQLite database connections.
	// However, for testing purposes it is useful to have mutexes turned on.  So, by default, mutexes default off.  But if compiled with
	// SQLITE_JIM_DEFAULT_FULLMUTEX then mutexes default on.
#ifdef JIM_DEFAULT_FULLMUTEX
	VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_FULLMUTEX);
#else
	VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_NOMUTEX);
#endif

	const char *arg;
	if (argc == 2)
	{
		arg = Jim_String(args[1]);
		if (!strcmp(arg, "-version"))
		{
			Jim_SetResultString(interp, LIBCU_VERSION, -1);
			return JIM_OK;
		}
		if (!strcmp(arg, "-has-codec"))
		{
#ifdef HAS_CODEC
			Jim_SetResultInt(interp, 1);
#else
			Jim_SetResultInt(interp, 0);
#endif
			return JIM_OK;
		}
	}
#ifdef HAS_CODEC
	void *key = nullptr;
	int keyLength = 0;
#endif
	const char *vfsName = nullptr;
	for (int i = 3; i + 1 < argc; i += 2)
	{
		arg = Jim_String(args[i]);
		bool b;
		if (!strcmp(arg, "-key"))
		{
#ifdef HAS_CODEC
			key = Jim_GetByteArray(args[i+1], &keyLength);
#endif
		}
		else if (!strcmp(arg, "-vfs"))
			vfsName = Jim_String(args[i+1]);
		else if (!strcmp(arg, "-readonly"))
		{
			if (Jim_GetBoolean(interp, args[i+1], &b)) return JIM_ERROR;
			if (b)
			{
				flags &= ~(VSystem::OPEN_READWRITE | VSystem::OPEN_CREATE);
				flags |= VSystem::OPEN_READONLY;
			}
			else
			{
				flags &= ~VSystem::OPEN_READONLY;
				flags |= VSystem::OPEN_READWRITE;
			}
		}
		else if (!strcmp(arg, "-create"))
		{
			if (Jim_GetBoolean(interp, args[i+1], &b)) return JIM_ERROR;
			if (b && (flags & VSystem::OPEN_READONLY) == 0)
				flags |= VSystem::OPEN_CREATE;
			else
				flags &= ~VSystem::OPEN_CREATE;
		}
		else if (!strcmp(arg, "-nomutex"))
		{
			if (Jim_GetBoolean(interp, args[i+1], &b)) return JIM_ERROR;
			if (b)
			{
				flags |= VSystem::OPEN_NOMUTEX;
				flags &= ~VSystem::OPEN_FULLMUTEX;
			}
			else
				flags &= ~VSystem::OPEN_NOMUTEX;
		}
		else if (!strcmp(arg, "-fullmutex"))
		{
			if (Jim_GetBoolean(interp, args[i+1], &b)) return JIM_ERROR;
			if (b)
			{
				flags |= VSystem::OPEN_FULLMUTEX;
				flags &= ~VSystem::OPEN_NOMUTEX;
			}
			else
				flags &= ~VSystem::OPEN_FULLMUTEX;
		}
		else if (!strcmp(arg, "-uri"))
		{
			if (Jim_GetBoolean(interp, args[i+1], &b)) return JIM_ERROR;
			if (b)
				flags |= VSystem::OPEN_URI;
			else
				flags &= ~VSystem::OPEN_URI;
		}
		else
		{
			Jim_SetResultFormatted(interp, "unknown option: %s", arg);
			return JIM_ERROR;
		}
	}
	if (argc < 3 || (argc & 1) != 1)
	{
		Jim_WrongNumArgs(interp, 1, args, "HANDLE FILENAME ?-vfs VFSNAME? ?-readonly BOOLEAN? ?-create BOOLEAN? ?-nomutex BOOLEAN? ?-fullmutex BOOLEAN? ?-uri BOOLEAN?"
#ifdef HAS_CODEC
			" ?-key CODECKEY?"
#endif
			);
		return JIM_ERROR;
	}
	char *errMsg = nullptr;
	TclContext *p = (TclContext *)Jim_Alloc(sizeof(*p));
	if (!p)
	{
		Jim_SetResultString(interp, "malloc failed", -1);
		return JIM_ERROR;
	}
	memset(p, 0, sizeof(*p));
	const char *fileName = Jim_String(args[2]);
	//fileName = Jim_TranslateFileName(interp, fileName, &translatedFilename);
	RC rc = DataEx::Open_v2(fileName, &p->Ctx, flags, vfsName);
	if (p->Ctx)
	{
		if (DataEx::ErrCode(p->Ctx) != RC_OK)
		{
			errMsg = _mprintf("%s", DataEx::ErrMsg(p->Ctx));
			DataEx::Close(p->Ctx);
			p->Ctx = nullptr;
		}
	}
	else
		errMsg = _mprintf("%s", DataEx::ErrStr(rc));
#ifdef HAS_CODEC
	if (p->Ctx)
		sqlite3_key(p->Ctx, key, keyLength);
#endif
	if (!p->Ctx)
	{
		Jim_SetResultString(interp, errMsg, -1);
		Jim_Free(p);
		_free(errMsg);
		return JIM_ERROR;
	}
	p->MaxStmt = NUM_PREPARED_STMTS;
	p->Interp = interp;
	arg = Jim_String(args[1]);
	Jim_CreateCommand(interp, arg, (Jim_CmdProc *)DbObjCmd, (ClientData)p, DbDeleteCmd);
	return JIM_OK;
}
#pragma endregion

// Make sure we have a PACKAGE_VERSION macro defined.  This will be defined automatically by the TEA makefile.  But other makefiles do not define it.
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION LIBCU_VERSION
#endif

// Initialize this module.
//
// This Jim module contains only a single new Jim command named "sqlite". (Hence there is no namespace.  There is no point in using a namespace
// if the extension only supplies one new name!)  The "sqlite" command is used to open a new SQLite database.  See the DbMain() routine above
// for additional information.
__device__ int Jim_sqlite3Init(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "sqlite3", DbMain, nullptr, nullptr);
	Jim_PackageProvide(interp, "sqlite3", PACKAGE_VERSION, 0);
	return JIM_OK;
}

#endif