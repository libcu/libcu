// A TCL Interface to SQLite
#if HASTINYTCL

#include "TclContext.cu.h"
#include <string.h>
#include <stdlib.h>

#define NUM_PREPARED_STMTS 10
#define MAX_PREPARED_STMTS 100

#pragma region BLOB
#ifdef OMIT_INCRBLOB

// Close all incrblob channels opened using database connection pDb.
// This is called when shutting down the database connection.
static __device__ void CloseIncrblobChannels(TclContext *tctx)
{
	IncrblobChannel *next;
	for (IncrblobChannel *p = tctx->Incrblob; p; p = next)
	{
		next = p->Next;
		// Note: Calling unregister here call Tcl_Close on the incrblob channel, which deletes the IncrblobChannel structure at *p. So do not call Tcl_Free() here.
		Tcl_UnregisterChannel(tctx->Interp, p->Channel);
	}
}

// Close an incremental blob channel.
static __device__ int IncrblobClose(ClientData instanceData, Tcl_Interp *interp)
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
	Tcl_Free((char *)p);

	if (rc != RC_OK)
	{
		Tcl_SetResult(interp, (char *)DataEx::ErrMsg(ctx), TCL_VOLATILE);
		return TCL_ERROR;
	}
	return TCL_OK;
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
	case SEEK_SET:
		p->Seek = offset;
		break;
	case SEEK_CUR:
		p->Seek += offset;
		break;
	case SEEK_END:
		p->Seek = Vdbe::Blob_Bytes(p->Blob) + offset;
		break;
	default: assert(!"Bad seekMode");
	}
	return p->Seek;
}

static __device__ void IncrblobWatch(ClientData instanceData, int mode) { } // NO-OP
static __device__ int IncrblobHandle(ClientData instanceData, int dir, ClientData *ptr)
{
	return TCL_ERROR;
}

__constant__ static Tcl_ChannelType IncrblobChannelType = {
	"incrblob",					// typeName                            
	TCL_CHANNEL_VERSION_2,		// version                             
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
static __device__ int CreateIncrblobChannel(Tcl_Interp *interp, TclContext *pDb, const char *dbName, const char *tableName, const char *columnName, int64 row, bool isReadonly)
{
	IncrblobChannel *p;
	Context *ctx = tctx->Ctx;
	int rc;
	int flags = TCL_READABLE | (isReadonly ? 0 : TCL_WRITABLE);

	Blob *blob;
	RC rc = Vdbe::Blob_Open(ctx, dbName, tableName, columnName, row, !isReadonly, &blob);
	if (rc != RC_OK)
	{
		Tcl_SetResult(interp, (char *)DataEx::ErrMsg(ctx), TCL_VOLATILE);
		return TCL_ERROR;
	}

	p = (IncrblobChannel *)Tcl_Alloc(sizeof(IncrblobChannel));
	p->Seek = 0;
	p->Blob = blob;

	static int count = 0; // This variable is used to name the channels: "incrblob_[incr count]"
	char channelName[64];
	__snprintf(channelName, sizeof(channelName), "incrblob_%d", ++count);
	p->Channel = Tcl_CreateChannel(&IncrblobChannelType, channelName, p, flags);
	Tcl_RegisterChannel(interp, p->Channel);

	// Link the new channel into the SqliteDb.pIncrblob list.
	p->Next = tctx->Incrblob;
	p->Prev = nullptr;
	if (p->Next)
		p->Next->Prev = p;
	tctx->Incrblob = p;
	p->Ctx = tctx;

	Tcl_SetResult(interp, (char *)Tcl_GetChannelName(p->Channel), TCL_VOLATILE);
	return TCL_OK;
}
#else
#define CloseIncrblobChannels(tctx)
#endif
#pragma endregion

#pragma region Stmt

// Look at the script prefix in pCmd.  We will be executing this script after first appending one or more arguments.  This routine analyzes
// the script to see if it is safe to use Tcl_EvalObjv() on the script rather than the more general Tcl_EvalEx().  Tcl_EvalObjv() is much faster.
//
// Scripts that are safe to use with Tcl_EvalObjv() consists of a command name followed by zero or more arguments with no [...] or $
// or {...} or ; to be seen anywhere.  Most callback scripts consist of just a single procedure name and they meet this requirement.
static __device__ bool SafeToUseEvalObjv(Tcl_Interp *interp, Tcl_Obj *cmd)
{
	// We could try to do something with Tcl_Parse().  But we will instead just do a search for forbidden characters.  If any of the forbidden
	// characters appear in pCmd, we will report the string as unsafe.
	int n;
	const char *z = Tcl_GetString(interp, cmd, &n);
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
	SqlFunc *newFunc = (SqlFunc *)Tcl_Alloc(sizeof(*newFunc) + strlen(name) + 1);
	newFunc->Name = (char *)&newFunc[1];
	int i;
	for (i = 0; name[i]; i++) { newFunc->Name[i] = __tolower(name[i]); }
	newFunc->Name[i] = 0;
	for (SqlFunc *p = tctx->Funcs; p; p = p->Next)
	{ 
		if (!_strcmp(p->Name, newFunc->Name))
		{
			Tcl_Free((char *)newFunc);
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
		Tcl_Free((char *)stmt->Sql);
#endif
	Vdbe::Finalize(stmt->Stmt);
	Tcl_Free(stmt);
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

// TCL calls this procedure when an sqlite3 database command is deleted.
static __device__ void DbDeleteCmd(ClientData db)
{
	TclContext *tctx = (TclContext *)db;
	FlushStmtCache(tctx);
	CloseIncrblobChannels(tctx);
	DataEx::Close(tctx->Ctx);
	while (tctx->Funcs)
	{
		SqlFunc *func = tctx->Funcs;
		tctx->Funcs = func->Next;
		_assert(func->Ctx == tctx);
		Tcl_DecrRefCount(func->Script);
		Tcl_Free(func);
	}
	while (tctx->Collates)
	{
		SqlCollate *collate = tctx->Collates;
		tctx->Collates = collate->Next;
		Tcl_Free(collate);
	}
	if (tctx->Busy)
		Tcl_Free(tctx->Busy);
	if (tctx->Trace)
		Tcl_Free(tctx->Trace);
	if (tctx->Profile)
		Tcl_Free(tctx->Profile);
	if (tctx->Auth)
		Tcl_Free(tctx->Auth);
	if (tctx->NullText)
		Tcl_Free(tctx->NullText);
	if (tctx->UpdateHook)
		Tcl_DecrRefCount(tctx->UpdateHook);
	if (tctx->RollbackHook)
		Tcl_DecrRefCount(tctx->RollbackHook);
	if (tctx->WalHook)
		Tcl_DecrRefCount(tctx->WalHook);
	if (tctx->CollateNeeded)
		Tcl_DecrRefCount(tctx->CollateNeeded);
	Tcl_Free(tctx);
}

#pragma endregion

#pragma region Hooks

// This routine is called when a database file is locked while trying to execute SQL.
static __device__ int DbBusyHandler(void *cd, int tries)
{
	TclContext *tctx = (TclContext *)cd;
	char b[30];
	__snprintf(b, sizeof(b), "%d", tries);
	int rc = Tcl_VarEval(tctx->Interp, tctx->Busy, " ", b, (char *)0);
	if (rc != TCL_OK || _atoi(tctx->Interp->result))
		return 0;
	return 1;
}

#ifndef OMIT_PROGRESS_CALLBACK
// This routine is invoked as the 'progress callback' for the database.
static __device__ ::RC DbProgressHandler(void *cd)
{
	TclContext *tctx = (TclContext *)cd;
	_assert(tctx->Progress);
	int rc = Tcl_Eval(tctx->Interp, tctx->Progress, 0, nullptr);
	if (rc != TCL_OK || _atoi(tctx->Interp->result))
		return RC_ERROR;
	return RC_OK;
}
#endif

#ifndef OMIT_TRACE
// This routine is called by the SQLite trace handler whenever a new block of SQL is executed.  The TCL script in pDb->zTrace is executed.
static __device__ void DbTraceHandler(void *cd, const char *sql)
{
	TclContext *tctx = (TclContext *)cd;
	char *cmd = _mprintf("%s%s", tctx->Trace, sql);
	Tcl_Eval(tctx->Interp, cmd, 0, nullptr);
	free(cmd);
	Tcl_ResetResult(tctx->Interp);
}

// This routine is called by the SQLite profile handler after a statement SQL has executed.  The TCL script in pDb->zProfile is evaluated.
static __device__ void DbProfileHandler(void *cd, const char *sql, uint64 tm)
{
	TclContext *tctx = (TclContext *)cd;
	char tmAsString[100];
	__snprintf(tmAsString, sizeof(tmAsString)-1, "%lld", tm);
	char *cmd = _mprintf("%s%s%s", tctx->Profile, sql, tmAsString);
	Tcl_Eval(tctx->Interp, cmd, 0, nullptr);
	free(cmd);
	Tcl_ResetResult(tctx->Interp);
}
#endif

// This routine is called when a transaction is committed.  The TCL script in pDb->zCommit is executed.  If it returns non-zero or
// if it throws an exception, the transaction is rolled back instead of being committed.
static __device__ ::RC DbCommitHandler(void *cd)
{
	TclContext *tctx = (TclContext *)cd;
	int rc = Tcl_Eval(tctx->Interp, tctx->Commit, 0, nullptr);
	if (rc != TCL_OK || _atoi(tctx->Interp->result))
		return RC_ERROR;
	return RC_OK;
}

static __device__ void DbRollbackHandler(void *clientData)
{
	TclContext *tctx = (TclContext *)clientData;
	_assert(tctx->RollbackHook);
	if (Tcl_Eval(tctx->Interp, tctx->RollbackHook, 0, nullptr) != TCL_OK)
		Tcl_BackgroundError(tctx->Interp);
}

// This procedure handles wal_hook callbacks.
static __device__ int DbWalHandler(void *clientData, Context *ctx, const char *dbName,  int entrys)
{
	TclContext *tctx = (TclContext *)clientData;
	Tcl_Interp *interp = tctx->Interp;
	_assert(tctx->WalHook);
	char b[50];
	Tcl_SetResult(interp, tctx->WalHook, nullptr);
	Tcl_AppendElement(interp, dbName, false);
	Tcl_AppendElement(interp, _itoa(entrys, b), false);
	char *cmd = interp->result;
	int rc = RC_OK;
	if (Tcl_Eval(interp, cmd, 0, nullptr) != TCL_OK || Tcl_GetInt(interp, interp->result, &rc) != TCL_OK)
		Tcl_BackgroundError(tctx->Interp);
	return rc;
}

#if defined(_TEST) && defined(ENABLE_UNLOCK_NOTIFY)
static __device__ void SetTestUnlockNotifyVars(Tcl_Interp *interp, int argId, int argsLength)
{
	char b[64];
	__snprintf(b, sizeof(b), "%d", argId);
	Tcl_SetVar(interp, "sqlite_unlock_notify_arg", b, TCLGLOBAL__ONLY);
	__snprintf(b, sizeof(b), "%d", argsLength);
	Tcl_SetVar(interp, "sqlite_unlock_notify_argcount", b, TCLGLOBAL__ONLY);
}
#else
#define SetTestUnlockNotifyVars(x,y,z)
#endif

#ifdef ENABLE_UNLOCK_NOTIFY
static __device__ void DbUnlockNotify(void **args, int argsLength)
{
	for (int i = 0; i < argsLength; i++)
	{
		const int flags = (TCL_EVALGLOBAL_ | TCL_EVAL_DIRECT);
		TclContext *tctx = (TclContext *)args[i];
		Tcl_Interp *interp = tctx->Interp;
		SetTestUnlockNotifyVars(interp, i, argsLength);
		_assert(tctx->UnlockNotify);
		Tcl_Eval(interp, *tctx->UnlockNotify, flags, nullptr);
		Tcl_DecrRefCount(tctx->UnlockNotify);
		tctx->UnlockNotify = nullptr;
	}
}
#endif

static __device__ void DbUpdateHandler(void *p, TK op, const char *dbName, const char *tableName, int64 rowid)
{
	TclContext *tctx = (TclContext *)p;
	Tcl_Interp *interp = tctx->Interp;
	_assert(tctx->UpdateHook);
	_assert(op == TK_INSERT || op == TK_UPDATE || op == TK_DELETE);
	char b[50];
	Tcl_SetResult(interp, tctx->UpdateHook, nullptr);
	Tcl_AppendElement(interp, (op == TK_INSERT?"INSERT":(op == TK_UPDATE?"UPDATE":"DELETE")), false);
	Tcl_AppendElement(interp, dbName, false);
	Tcl_AppendElement(interp, tableName, false);
	Tcl_AppendElement(interp, _itoa(rowid, b), false);
	char *cmd = interp->result;
	Tcl_Eval(interp, cmd, 0, nullptr);
}

static __device__ void TclCollateNeeded(void *p, Context *ctx, TEXTENCODE encode, const char *name)
{
	TclContext *tctx = (TclContext *)p;
	Tcl_Interp *interp = tctx->Interp;
	Tcl_SetResult(interp, tctx->CollateNeeded, nullptr);
	Tcl_AppendElement(interp, name, false);
	char *cmd = interp->result;
	Tcl_Eval(interp, cmd, 0, nullptr);
}

// This routine is called to evaluate an SQL collation function implemented using TCL script.
static __device__ int TclSqlCollate(void *p1, int aLength, const void *a, int bLength, const void *b)
{
	SqlCollate *p = (SqlCollate *)p1;
	Tcl_Interp *interp = p->Interp;
	Tcl_SetResult(interp, p->Script, nullptr);
	Tcl_AppendElement(p->Interp, (const char *)a, false); //Tcl_AppendElement(p->Interp, Tcl_Obj::NewStringObj((const char *)a, aLength), false);
	Tcl_AppendElement(p->Interp, (const char *)b, false); //Tcl_AppendElement(p->Interp, Tcl_Obj::NewStringObj((const char *)b, bLength), false);
	char *cmd = interp->result;
	Tcl_Eval(interp, cmd, 0, nullptr);
	return _atoi(interp->result);
}

// This routine is called to evaluate an SQL function implemented using TCL script.
static __device__ void TclSqlFunc(FuncContext *fctx, int argc, Mem **argv)
{
	SqlFunc *p = (SqlFunc *)Vdbe::User_Data(fctx);
	Tcl_Interp *interp = p->Interp;
	char *cmd;
	int rc;
	if (argc == 0)
	{
		// If there are no arguments to the function, call Tcl_EvalObjEx on the script object directly.  This allows the TCL compiler to generate
		// bytecode for the command on the first invocation and thus make subsequent invocations much faster.
		cmd = *p->Script;
		rc = Tcl_Eval(interp, cmd, 0, nullptr);
	}
	else
	{
		// If there are arguments to the function, make a shallow copy of the script object, lappend the arguments, then evaluate the copy.
		//
		// By "shallow" copy, we mean a only the outer list Tcl_Obj is duplicated. The new Tcl_Obj contains pointers to the original list elements. 
		// That way, when Tcl_EvalObjv() is run and shimmers the first element of the list to tclCmdNameType, that alternate representation will
		// be preserved and reused on the next invocation.
		const char **args;
		int argsLength;
		if (Tcl_SplitList(interp, *p->Script, &argsLength, &args))
		{
			Vdbe::Result_Error(fctx, interp->result, -1); 
			return;
		}
		int i;
		Tcl_SetResult(interp, "", nullptr);
		for (i = 0; i < argsLength; i++)
			Tcl_AppendElement(interp, args[i], false);
		for (i = 0; i < argc; i++)
		{
			Mem *in = argv[i];
			Tcl_Obj *val;
			// Set pVal to contain the i'th column of this row.
			char b[50];
			switch (Vdbe::Value_Type(in))
			{
			case TYPE_BLOB: {
				int bytes = Vdbe::Value_Bytes(in);
				val = Tcl_NewObj((char *)Vdbe::Value_Blob(in), bytes, "bytearray");
				break; }
			case TYPE_INTEGER: {
				int64 v = Vdbe::Value_Int64(in);
				val = (v >= -2147483647 && v <= 2147483647 ? Tcl_NewObj(_itoa((int)v, b), -1, "int") : Tcl_NewObj(_itoa64(v, b), -1, "wideInt"));
				break; }
			case TYPE_FLOAT: {
				double r = Vdbe::Value_Double(in);
				__snprintf(b, sizeof(b), "%f", r);
				val = Tcl_NewObj(b, -1, "double");
				break; }
			case TYPE_NULL: {
				val = Tcl_NewObj(p->Ctx->NullText, -1);
				break; }
			default: {
				int bytes = Vdbe::Value_Bytes(in);
				val = Tcl_NewObj((char *)Vdbe::Value_Text(in), bytes);
				break; }
			}
			Tcl_AppendElement(interp, *val, false);
			if (rc)
			{
				Vdbe::Result_Error(fctx, interp->result, -1); 
				return;
			}
		}
		cmd = interp->result;
		rc = Tcl_Eval(interp, cmd, 0, nullptr);
	}

	if (rc && rc != RC_BUSY)
		Vdbe::Result_Error(fctx, interp->result, -1); 
	else
	{
		Tcl_Obj *var = (Tcl_Obj *)interp->result;
		int n;
		char *data;
		const char *typeName = (var->TypePtr ? var->TypePtr : "");
		char c = typeName[0];
		if (c == 'b' && !_strcmp(typeName, "bytearray") && var->Bytes == 0)
		{
			// Only return a BLOB type if the Tcl variable is a bytearray and has no string representation.
			data = Tcl_GetString(interp, var, &n);
			Vdbe::Result_Blob(fctx, data, n, DESTRUCTOR_TRANSIENT);
		}
		else if (c == 'b' && !_strcmp(typeName, "boolean"))
		{
			Tcl_GetInt(interp, *var, &n);
			Vdbe::Result_Int(fctx, n);
		}
		else if (c == 'd' && !_strcmp(typeName, "double"))
		{
			double r;
			Tcl_GetDouble(interp, *var, &r);
			Vdbe::Result_Double(fctx, r);
		}
		else if ((c == 'w' && !_strcmp(typeName, "wideInt")) || (c == 'i' && !_strcmp(typeName, "int")))
		{
			int64 v;
			Tcl_GetWideInt(interp, *var, &v);
			Vdbe::Result_Int64(fctx, v);
		}
		else
		{
			data = Tcl_GetString(interp, var, &n);
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
	Tcl_Interp *interp = tctx->Interp;
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
	char *str = _mprintf("%s%s%s%s%s", tctx->Auth, codeName, (arg1?arg1:""), (arg2?arg2:""), (arg3?arg3:""), (arg4?arg4:""));
	int rc2 = TclGLOBAL_Eval(interp, str);
	free(str);
	ARC rc = ARC_OK;
	const char *reply = (rc == RC_OK ? interp->result : "ARC_DENY");
	if (!_strcmp(reply, "ARC_OK")) rc = ARC_OK;
	else if (!_strcmp(reply, "ARC_DENY")) rc = ARC_DENY;
	else if (!_strcmp(reply, "SQLITE_IGNORE")) rc = ARC_IGNORE;
	else rc = (ARC)999;
	return rc;
}
#endif

#pragma endregion

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
				free(line);
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

__constant__ static const char *_ends[] = {
	"RELEASE _tcl_transaction",        // rc==TCL_ERROR, nTransaction!=0
	"COMMIT",                          // rc!=TCL_ERROR, nTransaction==0
	"ROLLBACK TO _tcl_transaction ; RELEASE _tcl_transaction",
	"ROLLBACK"                         // rc==TCL_ERROR, nTransaction==0
};
static __device__ int DbTransPostCmd(ClientData data[], Tcl_Interp *interp, int result)
{
	TclContext *tctx = (TclContext *)data[0];
	Context *ctx = tctx->Ctx;
	int rc = result;

	tctx->Transactions--;
	const char *end = _ends[(rc == RC_ERROR) * 2 + (tctx->Transactions == 0)];

	tctx->DisableAuth++;
	if (DataEx::Exec(ctx, end, nullptr, nullptr, nullptr))
	{
		// This is a tricky scenario to handle. The most likely cause of an error is that the exec() above was an attempt to commit the 
		// top-level transaction that returned SQLITE_BUSY. Or, less likely, that an IO-error has occurred. In either case, throw a Tcl exception
		// and try to rollback the transaction.
		//
		// But it could also be that the user executed one or more BEGIN, COMMIT, SAVEPOINT, RELEASE or ROLLBACK commands that are confusing
		// this method's logic. Not clear how this would be best handled.
		if (rc != RC_ERROR)
		{
			Tcl_AppendResult(interp, DataEx::ErrMsg(ctx), 0);
			rc = RC_ERROR;
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
	Tcl_Interp *interp = tctx->Interp;
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
		if (sqlLength >= n && !_memcmp(p->Sql, sql, n) && (sql[n] == 0 || sql[n-1] == ';'))
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
			Tcl_SetResult(interp, (char *)DataEx::ErrMsg(ctx), nullptr);
			return RC_ERROR;
		}
		if (!stmt)
		{
			if (DataEx::ErrCode(ctx) != RC_OK)
			{
				Tcl_SetResult(interp, (char *)DataEx::ErrMsg(ctx), nullptr);
				return RC_ERROR; // A compile-time error in the statement.
			}
			else
				return RC_OK; // The statement was a no-op.  Continue to the next statement in the SQL string.
		}

		_assert(!p);
		vars = Vdbe::Bind_ParameterCount(stmt);
		int bytes = sizeof(SqlPreparedStmt) + vars * sizeof(Tcl_Obj *);
		p = (SqlPreparedStmt *)_alloc(bytes);
		_memset(p, 0, bytes);

		p->Stmt = stmt;
		p->SqlLength = (int)(*out - sql);
		p->Sql = Vdbe::Sql(stmt);
		p->Parms.data = (Tcl_Obj **)&p[1];
#ifdef _TEST
		if (!p->Sql)
		{
			char *copy = (char *)_alloc(p->SqlLength + 1);
			_memcpy(copy, sql, p->SqlLength);
			copy[p->SqlLength] = '\0';
			p->Sql = copy;
		}
#endif
	}
	_assert(p);
	_assert(strlen(p->Sql) == p->SqlLength);
	_assert(!_memcmp(p->Sql, sql, p->SqlLength));

	// Bind values to parameters that begin with $ or :
	int parmsLength = 0;
	for (int i = 1; i <= vars; i++)
	{
		const char *varName = Vdbe::Bind_ParameterName(stmt, i);
		if (varName && (varName[0] == '$' || varName[0] == ':' || varName[0] == '@'))
		{
			Tcl_Obj *var = (Tcl_Obj *)Tcl_GetVar2(interp, (char *)&varName[1], 0, 0);
			if (var)
			{
				int n;
				char *data;
				const char *typeName = (var->TypePtr ? var->TypePtr : "");
				char c = typeName[0];
				if (varName[0] == '@' || (c == 'b' && !_strcmp(typeName, "bytearray") && var->Bytes == 0))
				{
					// Load a BLOB type if the Tcl variable is a bytearray and it has no string representation or the host parameter name begins with "@".
					data = Tcl_GetString(interp, var, &n);
					Vdbe::Bind_Blob(stmt, i, data, n, DESTRUCTOR_STATIC);
					Tcl_IncrRefCount(var);
					p->Parms[parmsLength++] = var;
				}
				else if (c == 'b' && !_strcmp(typeName, "boolean"))
				{
					Tcl_GetInt(interp, *var, &n);
					Vdbe::Bind_Int(stmt, i, n);
				}
				else if (c == 'd' && !_strcmp(typeName, "double"))
				{
					double r;
					Tcl_GetDouble(interp, *var, &r);
					Vdbe::Bind_Double(stmt, i, r);
				}
				else if ((c == 'w' && !_strcmp(typeName, "wideInt")) || (c == 'i' && !_strcmp(typeName, "int")))
				{
					int64 v;
					Tcl_GetWideInt(interp, *var, &v);
					Vdbe::Bind_Int64(stmt, i, v);
				}
				else
				{
					data = Tcl_GetString(interp, var, &n);
					Vdbe::Bind_Text(stmt, i, data, n, DESTRUCTOR_STATIC);
					Tcl_IncrRefCount(var);
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
	for (int i = 0; i < preStmt->Parms.length; i++)
		Tcl_DecrRefCount(preStmt->Parms[i]);
	preStmt->Parms.length = 0;

	if (tctx->MaxStmt <= 0 || discard)
		DbFreeStmt(preStmt); // If the cache is turned off, deallocated the statement
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
	Tcl_Obj *Sql;               // Object holding string zSql
	const char *SqlAsString;    // Remaining SQL to execute
	SqlPreparedStmt *PreStmt;   // Current statement
	int Cols;                   // Number of columns returned by pStmt
	Tcl_Obj *Array;             // Name of array variable
	Tcl_Obj **ColNames;         // Array of column names
};

// Release any cache of column names currently held as part of the DbEvalContext structure passed as the first argument.
static __device__ void DbReleaseColumnNames(DbEvalContext *p)
{
	if (p->ColNames)
	{
		for (int i = 0; i < p->Cols; i++)
			Tcl_DecrRefCount(p->ColNames[i]);
		free(p->ColNames);
		p->ColNames = nullptr;
	}
	p->Cols = 0;
}

// Initialize a DbEvalContext structure.
//
// If pArray is not NULL, then it contains the name of a Tcl array variable. The "*" member of this array is set to a list containing
// the names of the columns returned by the statement as part of each call to dbEvalStep(), in order from left to right. e.g. if the names 
// of the returned columns are a, b and c, it does the equivalent of the tcl command:
//
//     set ${pArray}(*) {a b c}
static __device__ void DbEvalInit(DbEvalContext *p, TclContext *tctx, Tcl_Obj *sql, Tcl_Obj *array)
{
	_memset(p, 0, sizeof(DbEvalContext));
	p->Ctx = tctx;
	p->SqlAsString = *sql;
	p->Sql = sql;
	Tcl_IncrRefCount(sql);
	if (array)
	{
		p->Array = array;
		Tcl_IncrRefCount(array);
	}
}

// Obtain information about the row that the DbEvalContext passed as the first argument currently points to.
static __device__ void DbEvalRowInfo(DbEvalContext *p, int *colsOut, Tcl_Obj ***colNamesOut)
{
	// Compute column names
	if (!p->ColNames)
	{
		Vdbe *stmt = p->PreStmt->Stmt;
		int i;
		Tcl_Obj **colNames = nullptr; // Array of column names
		int cols = p->Cols = Vdbe::Column_Count(stmt); // Number of columns returned by pStmt
		if (cols > 0 && (colNamesOut || p->Array))
		{
			colNames = (Tcl_Obj **)_alloc(sizeof(Tcl_Obj *) * cols);
			for (i = 0; i < cols; i++)
			{
				colNames[i] = Tcl_NewObj(Vdbe::Column_Name(stmt, i), -1);
				Tcl_IncrRefCount(colNames[i]);
			}
			p->ColNames = colNames;
		}

		// If results are being stored in an array variable, then create the array(*) entry for that array
		if (p->Array)
		{
			Tcl_Interp *interp = p->Ctx->Interp;
			Tcl_SetResult(interp, "", nullptr);
			for (i = 0; i < cols; i++)
				Tcl_AppendElement(interp, *colNames[i], false);
			char *colList = interp->result;
			Tcl_SetVar2(interp, *p->Array, "*", colList, 0);
		}
	}

	if (colNamesOut)
		*colNamesOut = p->ColNames;
	if (colsOut)
		*colsOut = p->Cols;
}

// Return one of TCL_OK, TCL_BREAK or TCL_ERROR. If TCL_ERROR is returned, then an error message is stored in the interpreter before
// returning.
//
// A return value of TCL_OK means there is a row of data available. The data may be accessed using dbEvalRowInfo() and dbEvalColumnValue(). This
// is analogous to a return of SQLITE_ROW from sqlite3_step(). If TCL_BREAK is returned, then the SQL script has finished executing and there are
// no further rows available. This is similar to SQLITE_DONE.
static __device__ RC DbEvalStep(DbEvalContext *p)
{
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
				Tcl_SetResult(tctx->Interp, (char *)DataEx::ErrMsg(tctx->Ctx), TCL_VOLATILE);
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
	if (p->PreStmt)
	{
		Vdbe::Reset(p->PreStmt->Stmt);
		DbReleaseStmt(p->Ctx, p->PreStmt, false);
		p->PreStmt = nullptr;
	}
	if (p->Array)
	{
		Tcl_DecrRefCount(p->Array);
		p->Array = nullptr;
	}
	Tcl_DecrRefCount(p->Sql);
	DbReleaseColumnNames(p);
}

// Return a pointer to a Tcl_Obj structure with ref-count 0 that contains the value for the iCol'th column of the row currently pointed to by
// the DbEvalContext structure passed as the first argument.
static __device__ Tcl_Obj *DbEvalColumnValue(DbEvalContext *p, int colId)
{
	Vdbe *stmt = p->PreStmt->Stmt;
	char b[50];
	switch (Vdbe::Column_Type(stmt, colId))
	{
	case TYPE_BLOB: {
		int bytes = Vdbe::Column_Bytes(stmt, colId);
		const void *blob = Vdbe::Column_Blob(stmt, colId);
		if (!blob) bytes = 0;
		return Tcl_NewObj((char *)blob, bytes, "bytearray"); }
	case TYPE_INTEGER: {
		int64 v = Vdbe::Column_Int64(stmt, colId);
		return (v >= -2147483647 && v <= 2147483647 ? Tcl_NewObj(_itoa((int)v, b), -1, "int") : Tcl_NewObj(_itoa64(v, b), -1, "wideInt")); }
	case TYPE_FLOAT: {
		__snprintf(b, sizeof(b), "%f", Vdbe::Column_Double(stmt, colId));
		return Tcl_NewObj(b, -1, "double"); }
	case TYPE_NULL: {
		return Tcl_NewObj(p->Ctx->NullText, -1, "text"); }
	}
	return Tcl_NewObj((char *)Vdbe::Column_Text(stmt, colId), -1, "text");
}

// This function is part of the implementation of the command:
//
//   $db eval SQL ?ARRAYNAME? SCRIPT
static __device__ int DbEvalNextCmd(ClientData data[], Tcl_Interp *interp, int result)
{
	int rc = result;

	// The first element of the data[] array is a pointer to a DbEvalContext structure allocated using Tcl_Alloc(). The second element of data[]
	// is a pointer to a Tcl_Obj containing the script to run for each row returned by the queries encapsulated in data[0].
	DbEvalContext *p = (DbEvalContext *)data[0];
	Tcl_Obj *script = (Tcl_Obj *)data[1];
	Tcl_Obj *array = p->Array;

	while ((rc == RC_OK || rc == RC_ROW) && (rc = DbEvalStep(p)) == RC_OK)
	{
		int cols;
		Tcl_Obj **colNames;
		DbEvalRowInfo(p, &cols, &colNames);
		for (int i = 0 ; i < cols; i++)
		{
			Tcl_Obj *val = DbEvalColumnValue(p, i);
			if (!array)
				Tcl_SetVar2(interp, *colNames[i], nullptr, *val, false);
			else
				Tcl_SetVar2(interp, *array, *colNames[i], *val, false);
		}

		rc = Tcl_Eval(interp, *script, 0, nullptr);
	}

	Tcl_DecrRefCount(script);
	DbEvalFinalize(p);
	Tcl_Free(p);

	if (rc == RC_OK || rc == RC_DONE)
	{
		Tcl_ResetResult(interp);
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


// The "sqlite" command below creates a new Tcl command for each connection it opens to an SQLite database.  This routine is invoked
// whenever one of those connection-specific commands is executed in Tcl.  For example, if you run Tcl code like this:
//
//       sqlite3 db1  "my_database"
//       db1 close
//
// The first command opens a connection to the "my_database" database and calls that connection "db1".  The second command causes this subroutine to be invoked.
static __device__ int DbObjCmd(void *cd, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc < 2)
	{
		Tcl_WrongNumArgs(interp, 1, args, "SUBCOMMAND ...");
		return TCL_ERROR;
	}
	int choice;
	if (Tcl_GetIndex(interp, args[1], DB_strs, "option", 0, &choice))
		return TCL_ERROR;

	char b[50];
	TclContext *p = (TclContext *)cd;
	::RC rc2;
	int rc = TCL_OK;
	switch ((DB_enum)choice)
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
		Tcl_AppendResult("authorization not available in this build", 0);
		return TCL_ERROR;
#else
		if (argc > 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return TCL_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Auth)
				Tcl_AppendResult(interp, p->Auth, nullptr);
		}
		else
		{
			if (p->Auth)
				Tcl_Free(p->Auth);
			int len;
			char *auth = Tcl_GetString(interp, args[2], &len);
			if (auth && len > 0)
			{
				p->Auth = (char *)Tcl_Alloc(len + 1);
				_memcpy(p->Auth, auth, len + 1);
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

/*
	case DB_BACKUP: {
		//    $db backup ?DATABASE? FILENAME
		//
		// Open or create a database file named FILENAME.  Transfer the content of local database DATABASE (default: "main") into the FILENAME database.
		const char *srcDb;
		const char *destFile;
		if (argc == 3)
		{
			srcDb = "main";
			destFile = args[2];
		}
		else if (argc == 4)
		{
			srcDb = args[2];
			destFile = args[3];
		}
		else
		{
			Tcl_WrongNumArgs(interp, 2, args, "?DATABASE? FILENAME");
			return TCL_ERROR;
		}

		Context *destCtx;
		rc2 = DataEx::Open(destFile, &destCtx);
		if (rc2 != RC_OK)
		{
			Tcl_AppendResult(interp, "cannot open target database: ", DataEx::ErrMsg(destCtx), nullptr);
			DataEx::Close(destCtx);
			return TCL_ERROR;
		}
		Backup *backup = Backup::Init(destCtx, "main", p->Ctx, srcDb);
		if (!backup)
		{
			Tcl_AppendResult(interp, "backup failed: ", DataEx::ErrMsg(destCtx), nullptr);
			DataEx::Close(destCtx);
			return TCL_ERROR;
		}
		while ((rc2 = backup->Step(100)) == RC_OK) { }
		Backup::Finish(backup);
		if (rc == RC_DONE)
			rc = TCL_OK;
		else
		{
			Tcl_AppendResult(interp, "backup failed: ", DataEx::ErrMsg(destCtx), nullptr);
			rc = TCL_ERROR;
		}
		DataEx::Close(destCtx);
		return rc; }

	case DB_BUSY: {
		//    $db busy ?CALLBACK?
		//
		// Invoke the given callback if an SQL statement attempts to open a locked database file.
		if (argc > 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "CALLBACK");
			return TCL_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Busy)
				Tcl_AppendResult(interp, p->Busy, nullptr);
		}
		else
		{
			if (p->Busy)
				Tcl_Free(p->Busy);
			int len;
			char *busy = Tcl_GetString(interp, args[2], &len);
			if (busy && len > 0)
			{
				p->Busy = (char *)Tcl_Alloc(len + 1);
				_memcpy(p->Busy, busy, len + 1);
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
			Tcl_WrongNumArgs(interp, 1, args, "cache option ?arg?");
			return TCL_ERROR;
		}
		char *subCmd = (char *)args[2];
		if (subCmd[0] == 'f' && !_strcmp(subCmd,"flush"))
		{
			if (argc != 3)
			{
				Tcl_WrongNumArgs(interp, 2, args, "flush");
				return TCL_ERROR;
			}
			else
				FlushStmtCache(p);
		}
		else if (subCmd[0] == 's' && !_strcmp(subCmd, "size"))
		{
			if (argc != 4)
			{
				Tcl_WrongNumArgs(interp, 2, args, "size n");
				return TCL_ERROR;
			}
			else
			{
				int n;
				if (Tcl_GetInt(interp, args[3], &n) == TCL_ERROR)
				{
					Tcl_AppendResult(interp, "cannot convert \"", args[3], "\" to integer", nullptr);
					return TCL_ERROR;
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
					p->MaxStmt = n;
				}
			}
		}
		else
		{
			Tcl_AppendResult(interp, "bad option \"", args[2], "\": must be flush or size", nullptr);
			return TCL_ERROR;
		}
		return rc; }

	case DB_CHANGES: {
		//     $db changes
		//
		// Return the number of rows that were modified, inserted, or deleted by the most recent INSERT, UPDATE or DELETE statement, not including 
		// any changes made by trigger programs.
		if (argc != 2)
		{
			Tcl_WrongNumArgs(interp, 2, args, "");
			return TCL_ERROR;
		}
		char b[100];
		Tcl_SetResult(interp, _itoa(DataEx::CtxChanges(p->Ctx), b), TCL_VOLATILE);
		return rc; }

	case DB_CLOSE: {
		//    $db close
		//
		// Shutdown the database
		Tcl_DeleteCommand(interp, (char *)args[0]);
		return rc; }

	case DB_COLLATE: {
		//     $db collate NAME SCRIPT
		//
		// Create a new SQL collation function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
		if (argc != 4)
		{
			Tcl_WrongNumArgs(interp, 2, args, "NAME SCRIPT");
			return TCL_ERROR;
		}
		char *name = (char *)args[2];
		int scriptLength;
		char *script = Tcl_GetString(interp, args[3], &scriptLength);
		SqlCollate *collate = (SqlCollate *)Tcl_Alloc(sizeof(*collate) + scriptLength + 1);
		if (!collate) return TCL_ERROR;
		collate->Interp = interp;
		collate->Next = p->Collates;
		collate->Script = (char *)&collate[1];
		p->Collates = collate;
		_memcpy(collate->Script, script, scriptLength+1);
		if (DataEx::CreateCollation(p->Ctx, name, TEXTENCODE_UTF8, collate, TclSqlCollate))
		{
			Tcl_SetResult(interp, (char *)DataEx::ErrMsg(p->Ctx), TCL_VOLATILE);
			return TCL_ERROR;
		}
		return rc; }

	case DB_COLLATION_NEEDED: {
		//     $db collation_needed SCRIPT
		//
		// Create a new SQL collation function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
		if (argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "SCRIPT");
			return TCL_ERROR;
		}
		if (p->CollateNeeded)
			Tcl_DecrRefCount(p->CollateNeeded);
		p->CollateNeeded = Tcl_DuplicateObj((char *)args[2]);
		Tcl_IncrRefCount(p->CollateNeeded);
		DataEx::CollationNeeded(p->Ctx, p, TclCollateNeeded);
		return rc; }

	case DB_COMMIT_HOOK: {
		//    $db commit_hook ?CALLBACK?
		//
		// Invoke the given callback just before committing every SQL transaction. If the callback throws an exception or returns non-zero, then the
		// transaction is aborted.  If CALLBACK is an empty string, the callback is disabled.
		if (argc > 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return TCL_ERROR;
		} else if (argc == 2)
		{
			if (p->Commit)
				Tcl_AppendResult(interp, p->Commit, nullptr);
		}
		else
		{
			if (p->Commit)
				Tcl_Free(p->Commit);
			int len;
			char *commit = Tcl_GetString(interp, args[2], &len);
			if (commit && len > 0)
			{
				p->Commit = (char *)Tcl_Alloc(len + 1);
				_memcpy(p->Commit, commit, len+1);
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
		// built-in "info complete" command of Tcl.
#ifndef OMIT_COMPLETE
		if (argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "SQL");
			return TCL_ERROR;
		}
		bool isComplete = Parse::Complete(args[2]);
		Tcl_SetResult(interp, (isComplete?"1":"0"), TCL_VOLATILE);
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
			Tcl_WrongNumArgs(interp, 2, args, "CONFLICT-ALGORITHM TABLE FILENAME ?SEPARATOR? ?NULLINDICATOR?");
			return TCL_ERROR;
		}
		int i;
		char *sep = (argc >= 6 ? args[5] : "\t");
		char *null = (argc >= 7 ? args[6] : "");
		char *conflict = (char *)args[2]; // The conflict algorithm to use
		char *table = (char *)args[3]; // Insert data into this table
		char *file = (char *)args[4]; // The file from which to extract data
		int sepLength = strlen(sep); // Number of bytes in zSep[]
		int nullLength = strlen(null); // Number of bytes in zNull[]
		if (sepLength == 0)
		{
			Tcl_AppendResult(interp, "Error: non-null separator required for copy", 0);
			return TCL_ERROR;
		}
		if (_strcmp(conflict, "rollback") && _strcmp(conflict, "abort") && _strcmp(conflict, "fail") && _strcmp(conflict, "ignore") && _strcmp(conflict, "replace"))
		{
			Tcl_AppendResult(interp, "Error: \"", conflict, "\", conflict-algorithm must be one of: rollback, abort, fail, ignore, or replace", 0);
			return TCL_ERROR;
		}
		char *sql = _mprintf("SELECT * FROM '%q'", table); // An SQL statement
		if (!sql)
		{
			Tcl_AppendResult(interp, "Error: no such table: ", table, nullptr);
			return TCL_ERROR;
		}
		int bytes = strlen(sql); // Number of bytes in an SQL string
		Vdbe *stmt; // A statement
		rc2 = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
		free(sql);
		int cols;// Number of columns in the table
		if (rc2)
		{
			Tcl_AppendResult(interp, "Error: ", DataEx::ErrMsg(p->Ctx), 0);
			cols = 0;
		}
		else
			cols = Vdbe::Column_Count(stmt);
		Vdbe::Finalize(stmt);
		if (cols == 0)
			return TCL_ERROR;
		sql = (char *)_alloc(bytes + 50 + cols*2);
		if (!sql)
		{
			Tcl_AppendResult(interp, "Error: can't malloc()", nullptr);
			return TCL_ERROR;
		}
		__snprintf(sql, bytes+50, "INSERT OR %q INTO '%q' VALUES(?", conflict, table);
		int j = strlen(sql);
		for (i = 1; i < cols; i++)
		{
			sql[j++] = ',';
			sql[j++] = '?';
		}
		sql[j++] = ')';
		sql[j] = 0;
		rc2 = Prepare::Prepare_(p->Ctx, sql, -1, &stmt, 0);
		free(sql);
		if (rc2)
		{
			Tcl_AppendResult(interp, "Error: ", DataEx::ErrMsg(p->Ctx), nullptr);
			Vdbe::Finalize(stmt);
			return TCL_ERROR;
		}
		FILE *in = _fopen(file, "rb"); // The input file
		if (!in)
		{
			Tcl_AppendResult(interp, "Error: cannot open file: ", file, nullptr);
			Vdbe::Finalize(stmt);
			return TCL_ERROR;
		}
		char **colNames = (char **)_alloc(sizeof(colNames[0]) * (cols+1)); // zLine[] broken up into columns
		if (!colNames)
		{
			Tcl_AppendResult(interp, "Error: can't malloc()", 0);
			_fclose(in);
			return TCL_ERROR;
		}
		DataEx::Exec(p->Ctx, "BEGIN", 0, 0, 0);
		char *commit = "COMMIT"; // How to commit changes
		char *line; // A single line of input from the file
		int lineno = 0; // Line number of input file
		char lineNum[80]; // Line number print buffer
		//Tcl_Obj *result; // interp result
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
				char *err = (char *)_alloc(errLength);
				if (err)
				{
					__snprintf(err, errLength, "Error: %s line %d: expected %d columns of data but found %d", file, lineno, cols, i+1);
					Tcl_AppendResult(interp, err, nullptr);
					free(err);
				}
				commit = "ROLLBACK";
				break;
			}
			for (i = 0; i < cols; i++)
			{
				// check for null data, if so, bind as null
				if ((nullLength > 0 && !_strcmp(colNames[i], null)) || !strlen(colNames[i]))
					Vdbe::Bind_Null(stmt, i+1);
				else
					Vdbe::Bind_Text(stmt, i+1, colNames[i], -1, DESTRUCTOR_STATIC);
			}
			stmt->Step();
			rc2 = Vdbe::Reset(stmt);
			free(line);
			if (rc2 != RC_OK)
			{
				Tcl_AppendResult(interp, "Error: ", DataEx::ErrMsg(p->Ctx), 0);
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
			Tcl_SetResult(interp, _itoa(lineno, b), TCL_VOLATILE);
			rc = TCL_OK;
		}
		else
		{
			// failure, append lineno where failed
			__snprintf(lineNum, sizeof(lineNum), "%d", lineno);
			Tcl_AppendResult(interp, ", failed while processing line: ", lineNum, nullptr);
			rc = TCL_ERROR;
		}
		return rc; }

	case DB_ENABLE_LOAD_EXTENSION: {
		//    $db enable_load_extension BOOLEAN
		//
		// Turn the extension loading feature on or off.  It if off by default.
#ifndef OMIT_LOAD_EXTENSION
		if (argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "BOOLEAN");
			return TCL_ERROR;
		}
		bool onoff;
		if (Tcl_GetBoolean(interp, args[2], &onoff))
			return TCL_ERROR;
		DataEx::EnableLoadExtension(p->Ctx, onoff);
#else
		Tcl_AppendResult(interp, "extension loading is turned off at compile-time", 0);
		return TCL_ERROR;
#endif
		return rc; }

	case DB_ERRORCODE: {
		//    $db errorcode
		//
		// Return the numeric error code that was returned by the most recent call to sqlite3_exec().
		char b[100];
		Tcl_SetResult(interp, _itoa(DataEx::ErrCode(p->Ctx), b), TCL_VOLATILE);
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
			Tcl_WrongNumArgs(interp, 2, args, "SQL");
			return TCL_ERROR;
		}

		DbEvalInit(&sEval, p, (char *)args[2], nullptr);
		rc = DbEvalStep(&sEval);
		if (choice == DB_ONECOLUMN)
		{
			if (rc == TCL_OK)
				Tcl_SetResult(interp, DbEvalColumnValue(&sEval, 0));
			else if (rc == TCL_BREAK)
				Tcl_ResetResult(interp);
		}
		else if (rc == TCL_BREAK || rc == TCL_OK)
			Tcl_SetResult(interp, (rc == TCL_OK?"1":"0"), TCL_VOLATILE);
		DbEvalFinalize(&sEval);

		if (rc == TCL_BREAK)
			rc = TCL_OK;
		return rc; }

	case DB_EVAL: {
		//    $db eval $sql ?array? ?{  ...code... }?
		//
		// The SQL statement in $sql is evaluated.  For each row, the values are placed in elements of the array named "array" and ...code... is executed.
		// If "array" and "code" are omitted, then no callback is every invoked. If "array" is an empty string, then the values are placed in variables
		// that have the same name as the fields extracted by the query.
		if (argc < 3 || argc > 5)
		{
			Tcl_WrongNumArgs(interp, 2, args, "SQL ?ARRAY-NAME? ?SCRIPT?");
			return TCL_ERROR;
		}

		if (argc == 3)
		{
			DbEvalContext sEval;
			Tcl_Obj *ret = Tcl_NewObj();
			Tcl_IncrRefCount(ret);
			DbEvalInit(&sEval, p, args[2], nullptr);

			while ((rc2 = DbEvalStep(&sEval)) == RC_OK)
			{
				int cols;
				DbEvalRowInfo(&sEval, &cols, nullptr);
				for (int i = 0; i < cols; i++)
					Tcl_ListObjAppendElement(interp, ret, DbEvalColumnValue(&sEval, i));
			}
			DbEvalFinalize(&sEval);
			if (rc2 == RC_DONE)
			{
				Tcl_SetObjResult(interp, ret);
				rc2 = RC_OK;
			}
			Tcl_DecrRefCount(ret);
		}
		else
		{
			ClientData cd[2];
			Tcl_Obj *array = (argc == 5 && *args[3] ? args[3] : nullptr);
			Tcl_Obj *script = args[argc-1];
			Tcl_IncrRefCount(script);

			DbEvalContext *p2 = (DbEvalContext *)Tcl_Alloc(sizeof(DbEvalContext));
			DbEvalInit(p2, p, args[2], array);

			cd[0] = (ClientData)p2;
			cd[1] = (ClientData)script;
			rc2 = DbEvalNextCmd(cd, interp, RC_OK);
		}
		return rc; }

	case DB_FUNCTION: {
		//     $db function NAME [-argcount N] SCRIPT
		//
		// Create a new SQL function called NAME.  Whenever that function is called, invoke SCRIPT to evaluate the function.
		Tcl_Obj *script;
		int args4 = -1;
		if (argc == 6)
		{
			const char *z = args[3];
			int n = strlen(z);
			if (n > 2 && !strncmp(z, "-argcount",n))
			{
				if (Tcl_GetInt(interp, args[4], &args4)) return TCL_ERROR;
				if (args4 < 0)
				{
					Tcl_AppendResult(interp, "number of arguments must be non-negative", nullptr);
					return TCL_ERROR;
				}
			}
			script = args[5];
		}
		else if (argc != 4)
		{
			Tcl_WrongNumArgs(interp, 2, args, "NAME [-argcount N] SCRIPT");
			return TCL_ERROR;
		}
		else
			script = args[3];
		char *name = args[2];
		SqlFunc *func = FindSqlFunc(p, name);
		if (!func) return TCL_ERROR;
		if (func->Script)
			Tcl_DecrRefCount(func->Script);
		func->Script = script;
		Tcl_IncrRefCount(script);
		func->UseEvalObjv = SafeToUseEvalObjv(interp, script);
		rc2 = DataEx::CreateFunction(p->Ctx, name, args4, TEXTENCODE_UTF8, func, TclSqlFunc, 0, 0);
		if (rc2 != RC_OK)
		{
			rc = TCL_ERROR;
			Tcl_SetResult(interp, (char *)DataEx::ErrMsg(p->Ctx), TCL_VOLATILE);
		}
		return rc; }

	case DB_INCRBLOB: {
		//     $db incrblob ?-readonly? ?DB? TABLE COLUMN ROWID
#ifdef OMIT_INCRBLOB
		Tcl_AppendResult(interp, "incrblob not available in this build", nullptr);
		return TCL_ERROR;
#else
		// Check for the -readonly option
		int isReadonly = (argc > 3 && !_strcmp(args[2], "-readonly") ? 1 : 0);
		if (argc != (5+isReadonly) && argc != (6+isReadonly))
		{
			Tcl_WrongNumArgs(interp, 2, args, "?-readonly? ?DB? TABLE COLUMN ROWID");
			return TCL_ERROR;
		}

		const char *db = (argc == (6+isReadonly) ? args[2] : "main");
		const char *table = args[argc-3];
		const char *column = args[argc-2];
		int64 rows;
		rc = Tcl_GetWideInt(interp, args[argc-1], &rows);

		if (rc == TCL_OK)
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
			Tcl_WrongNumArgs(interp, 2, args, "NULLVALUE");
			return TCL_ERROR;
		}
		if (argc == 3)
		{
			int len;
			char *null = Tcl_GetString(interp, args[2], &len);
			if (p->NullText)
				Tcl_Free(p->NullText);
			if (null && len > 0)
			{
				p->NullText = (char *)Tcl_Alloc(len + 1);
				_memcpy(p->NullText, null, len);
				p->NullText[len] = '\0';
			}
			else
				p->NullText = nullptr;
		}
		Tcl_SetResult(interp, p->NullText, TCL_VOLATILE);
		return rc; }

	case DB_LAST_INSERT_ROWID: {
		//     $db last_insert_rowid 
		//
		// Return an integer which is the ROWID for the most recent insert.
		if (argc != 2)
		{
			Tcl_WrongNumArgs(interp, 2, args, "");
			return TCL_ERROR;
		}
		int64 rowid = DataEx::CtxLastInsertRowid(p->Ctx);
		Tcl_SetResult(interpt, _itoa64(rowid, b), TCL_VOLATILE);
		return rc; }

							   // The DB_ONECOLUMN method is implemented together with DB_EXISTS.

	case DB_PROGRESS: {
		//    $db progress ?N CALLBACK?
		// 
		// Invoke the given callback every N virtual machine opcodes while executing queries.
		if (argc == 2)
		{
			if (p->Progress)
				Tcl_AppendResult(interp, p->Progress, 0);
		}
		else if (argc == 4)
		{
			int N;
			if (Tcl_GetInt(interp, args[2], &N) != TCL_OK)
				return TCL_ERROR;
			if (p->Progress)
				Tcl_Free(p->Progress);
			int len;
			char *progress = Tcl_GetString(interp, args[3], &len);
			if (progress && len > 0)
			{
				p->Progress = Tcl_Alloc(len + 1);
				_memcpy(p->Progress, progress, len+1);
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
			Tcl_WrongNumArgs(interp, 2, args, "N CALLBACK");
			return TCL_ERROR;
		}
		return rc; }

	case DB_PROFILE: {
		//    $db profile ?CALLBACK?
		//
		// Make arrangements to invoke the CALLBACK routine after each SQL statement that has run.  The text of the SQL and the amount of elapse time are
		// appended to CALLBACK before the script is run.
		if (argc > 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return TCL_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Profile)
				Tcl_AppendResult(interp, p->Profile, nullptr);
		}
		else
		{
			if (p->Profile)
				Tcl_Free(p->Profile);
			int len;
			char *profile = Tcl_GetString(interp, args[2], &len);
			if (profile && len > 0)
			{
				p->Profile = (char *)Tcl_Alloc(len + 1);
				_memcpy(p->Profile, profile, len+1);
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
			Tcl_WrongNumArgs(interp, 2, args, "KEY");
			return TCL_ERROR;
		}
#ifdef HAS_CODEC
		int nKey;
		void *pKey = Tcl_GetByteArray(interp, args[2], &nKey);
		rc2 = sqlite3_rekey(p->Ctx, pKey, nKey);
		if (rc2)
		{
			Tcl_AppendResult(interp, DataEx::ErrStr(rc2), 0);
			rc = TCL_ERROR;
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
			srcFile = args[2];
		}
		else if (argc == 4)
		{
			destDb = args[2];
			srcFile = args[3];
		}
		else
		{
			Tcl_WrongNumArgs(interp, 2, args, "?DATABASE? FILENAME");
			return TCL_ERROR;
		}
		Context *src;
		rc2 = DataEx::Open_v2(srcFile, &src, VSystem::OPEN::OPEN_READONLY, 0);
		if (rc2 != RC_OK)
		{
			Tcl_AppendResult(interp, "cannot open source database: ", DataEx::ErrMsg(src), nullptr);
			DataEx::Close(src);
			return TCL_ERROR;
		}
		Backup *backup = Backup::Init(p->Ctx, destDb, src, "main");
		if (!backup)
		{
			Tcl_AppendResult(interp, "restore failed: ", DataEx::ErrMsg(p->Ctx), nullptr);
			DataEx::Close(src);
			return TCL_ERROR;
		}
		int timeout = 0;
		while ((rc2 = backup->Step(100)) == RC_OK || rc2 == RC_BUSY)
		{
			if (rc2 == RC_BUSY)
			{
				if (timeout++ >= 3) break;
				DataEx::Sleep(100);
			}
		}
		Backup::Finish(backup);
		if (rc2 == RC_DONE)
			rc = TCL_OK;
		else if (rc2 == RC_BUSY || rc2 == RC_LOCKED)
		{
			Tcl_AppendResult(interp, "restore failed: source database busy", nullptr);
			rc = TCL_ERROR;
		}
		else
		{
			Tcl_AppendResult(interp, "restore failed: ", DataEx::ErrMsg(p->Ctx), nullptr);
			rc = TCL_ERROR;
		}
		DataEx::Close(src);
		return rc; }

	case DB_STATUS: {
		//     $db status (step|sort|autoindex)
		//
		// Display SQLITE_STMTSTATUS_FULLSCAN_STEP or SQLITE_STMTSTATUS_SORT for the most recent eval.
		if (argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "(step|sort|autoindex)");
			return TCL_ERROR;
		}
		const char *op = args[2];
		int v;
		if (!_strcmp(op, "step")) v = p->Steps;
		else if (!_strcmp(op, "sort")) v = p->Sorts;
		else if (!_strcmp(op, "autoindex")) v = p->Indexs;
		else
		{
			Tcl_AppendResult(interp, "bad argument: should be autoindex, step, or sort", nullptr);
			return TCL_ERROR;
		}
		Tcl_SetResult(interp, _itoa(v, b), TCL_VOLATILE);
		return rc; }

	case DB_TIMEOUT: {
		//     $db timeout MILLESECONDS
		//
		// Delay for the number of milliseconds specified when a file is locked.
		if (argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "MILLISECONDS");
			return TCL_ERROR;
		}
		int ms;
		if (Tcl_GetInt(interp, args[2], &ms)) return TCL_ERROR;
		DataEx::BusyTimeout(p->Ctx, ms);
		return rc; }

	case DB_TOTAL_CHANGES: {
		//     $db total_changes
		//
		// Return the number of rows that were modified, inserted, or deleted since the database handle was created.
		if (argc != 2)
		{
			Tcl_WrongNumArgs(interp, 2, args, "");
			return TCL_ERROR;
		}
		Tcl_SetResult(interp, _itoa(DataEx::CtxTotalChanges(p->Ctx), b), TCL_VOLATILE);
		return rc; }

	case DB_TRACE: {
		//    $db trace ?CALLBACK?
		//
		// Make arrangements to invoke the CALLBACK routine for each SQL statement that is executed.  The text of the SQL is appended to CALLBACK before
		// it is executed.
		if (argc > 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "?CALLBACK?");
			return TCL_ERROR;
		}
		else if (argc == 2)
		{
			if (p->Trace)
				Tcl_AppendResult(interp, p->Trace, nullptr);
		}
		else
		{
			if (p->Trace)
				Tcl_Free(p->Trace);
			int len;
			char *trace = Tcl_GetString(interp, args[2], &len);
			if (trace && len > 0)
			{
				p->Trace = (char *)Tcl_Alloc(len + 1);
				_memcpy(p->Trace, trace, len+1);
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
		// Start a new transaction (if we are not already in the midst of a transaction) and execute the TCL script SCRIPT.  After SCRIPT
		// completes, either commit the transaction or roll it back if SCRIPT throws an exception.  Or if no new transation was started, do nothing.
		// pass the exception on up the stack.
		//
		// This command was inspired by Dave Thomas's talk on Ruby at the 2005 O'Reilly Open Source Convention (OSCON).
		if (argc != 3 && argc != 4)
		{
			Tcl_WrongNumArgs(interp, 2, args, "[TYPE] SCRIPT");
			return TCL_ERROR;
		}

		const char *begin = "SAVEPOINT _tcl_transaction";
		if (p->Transactions == 0 && argc == 4)
		{
			int ttype;
			if (Tcl_GetIndex(interp, args[2], TTYPE_strs, "transaction type", 0, &ttype))
				return TCL_ERROR;
			switch ((TTYPE_enum)ttype)
			{
			case TTYPE_DEFERRED: break; // no-op
			case TTYPE_EXCLUSIVE: begin = "BEGIN EXCLUSIVE"; break;
			case TTYPE_IMMEDIATE: begin = "BEGIN IMMEDIATE"; break;
			}
		}
		char *script = args[argc-1];

		// Run the SQLite BEGIN command to open a transaction or savepoint.
		p->DisableAuth++;
		rc2 = DataEx::Exec(p->Ctx, begin, 0, 0, 0);
		p->DisableAuth--;
		if (rc2 != RC_OK)
		{
			Tcl_AppendResult(interp, DataEx::ErrMsg(p->Ctx), nullptr);
			return TCL_ERROR;
		}
		p->Transactions++;

		// If using NRE, schedule a callback to invoke the script pScript, then a second callback to commit (or rollback) the transaction or savepoint
		// opened above. If not using NRE, evaluate the script directly, then call function DbTransPostCmd() to commit (or rollback) the transaction 
		// or savepoint.
		//if (DbUseNre())
		//{
		//	Tcl_NRAddCallback(interp, DbTransPostCmd, cd, 0, 0, 0);
		//	Tcl_NREvalObj(interp, pScript, 0);
		//} else
		rc2 = DbTransPostCmd(&cd, interp, Tcl_EvalObj(interp, script, 0));
		return rc; }

	case DB_UNLOCK_NOTIFY: {
		//    $db unlock_notify ?script?
#ifndef ENABLE_UNLOCK_NOTIFY
		Tcl_AppendResult(interp, "unlock_notify not available in this build", nullptr);
		rc = TCL_ERROR;
#else
		if (argc != 2 && argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "?SCRIPT?");
			rc = TCL_ERROR;
		}
		else
		{
			if (p->UnlockNotify)
			{
				Tcl_DecrRefCount(p->UnlockNotify);
				p->UnlockNotify = nullptr;
			}

			void (*notify)(void **, int) = nullptr;
			void *notifyArg = nullptr;
			if (argc == 3)
			{
				notify = DbUnlockNotify;
				notifyArg = (void *)p;
				p->UnlockNotify = args[2];
				Tcl_IncrRefCount(p->UnlockNotify);
			}

			if (DataEx::UnlockNotify(p->Ctx, notify, notifyArg))
			{
				Tcl_AppendResult(interp, DataEx::ErrMsg(p->Ctx), nullptr);
				rc = TCL_ERROR;
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
		Tcl_Obj **hook; 
		if (choice == DB_UPDATE_HOOK) hook = &p->UpdateHook;
		else if (choice == DB_WAL_HOOK) hook = &p->WalHook;
		else hook = &p->RollbackHook;

		if (argc != 2 && argc != 3)
		{
			Tcl_WrongNumArgs(interp, 2, args, "?SCRIPT?");
			return TCL_ERROR;
		}
		if (*hook)
		{
			Tcl_SetObjResult(interp, *hook);
			if (argc == 3)
			{
				Tcl_DecrRefCount(*hook);
				*hook = nullptr;
			}
		}
		if (argc == 3)
		{
			_assert(!(*hook));
			if (strlen(args[2]) > 0)
			{
				*hook = (Tcl_Obj *)args[2];
				Tcl_IncrRefCount(*hook);
			}
		}
		DataEx::UpdateHook(p->Ctx, (p->UpdateHook?DbUpdateHandler:0), p);
		DataEx::RollbackHook(p->Ctx, (p->RollbackHook?DbRollbackHandler:0), p);
		DataEx::WalHook(p->Ctx, (p->WalHook?DbWalHandler:0), p);
		return rc; }
*/

	case DB_VERSION: {
		//    $db version
		//
		// Return the version string for this database.
		Tcl_SetResult(interp, LIBCU_VERSION, TCL_STATIC);
		return rc; }

	} // End of the SWITCH statement
	return rc;
}

#pragma endregion

#pragma region DbMain

//   sqlite3 DBNAME FILENAME ?-vfs VFSNAME? ?-key KEY? ?-readonly BOOLEAN?
//                           ?-create BOOLEAN? ?-nomutex BOOLEAN?
//
// This is the main Tcl command.  When the "sqlite" Tcl command is invoked, this routine runs to process that command.
//
// The first argument, DBNAME, is an arbitrary name for a new database connection.  This command creates a new command named
// DBNAME that is used to control that connection.  The database connection is deleted when the DBNAME command is deleted.
//
// The second argument is the name of the database file.
static __device__ int DbMain(void *cd, Tcl_Interp *interp, int argc, const char *args[])
{
	// In normal use, each TCL interpreter runs in a single thread.  So by default, we can turn of mutexing on SQLite database connections.
	// However, for testing purposes it is useful to have mutexes turned on.  So, by default, mutexes default off.  But if compiled with
	// SQLITE_TCL_DEFAULT_FULLMUTEX then mutexes default on.
#ifdef TCL_DEFAULT_FULLMUTEX
	VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_FULLMUTEX);
#else
	VSystem::OPEN flags = (VSystem::OPEN)(VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_NOMUTEX);
#endif

	const char *arg;
	if (argc == 2)
	{
		arg = args[1];
		if (!_strcmp(arg, "-version"))
		{
			Tcl_AppendResult(interp, LIBCU_VERSION, nullptr);
			return TCL_OK;
		}
		if (!_strcmp(arg, "-has-codec"))
		{
#ifdef HAS_CODEC
			Tcl_AppendResult(interp, "1", nullptr);
#else
			Tcl_AppendResult(interp, "0", nullptr);
#endif
			return TCL_OK;
		}
	}
#ifdef HAS_CODEC
	void *key = nullptr;
	int keyLength = 0;
#endif
	const char *vfsName = nullptr;
	for (int i = 3; i + 1 < argc; i += 2)
	{
		arg = args[i];
		bool b;
		if (!_strcmp(arg, "-key"))
		{
#ifdef HAS_CODEC
			key = Tcl_GetByteArray(args[i+1], &keyLength);
#endif
		}
		else if (!_strcmp(arg, "-vfs"))
			vfsName = args[i+1];
		else if (!_strcmp(arg, "-readonly"))
		{
			if (Tcl_GetBoolean(interp, args[i+1], &b)) return TCL_ERROR;
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
		else if (!_strcmp(arg, "-create"))
		{
			if (Tcl_GetBoolean(interp, args[i+1], &b)) return TCL_ERROR;
			if (b && (flags & VSystem::OPEN_READONLY) == 0)
				flags |= VSystem::OPEN_CREATE;
			else
				flags &= ~VSystem::OPEN_CREATE;
		}
		else if (!_strcmp(arg, "-nomutex"))
		{
			if (Tcl_GetBoolean(interp, args[i+1], &b)) return TCL_ERROR;
			if (b)
			{
				flags |= VSystem::OPEN_NOMUTEX;
				flags &= ~VSystem::OPEN_FULLMUTEX;
			}
			else
				flags &= ~VSystem::OPEN_NOMUTEX;
		}
		else if (!_strcmp(arg, "-fullmutex"))
		{
			if (Tcl_GetBoolean(interp, args[i+1], &b)) return TCL_ERROR;
			if (b)
			{
				flags |= VSystem::OPEN_FULLMUTEX;
				flags &= ~VSystem::OPEN_NOMUTEX;
			}
			else
				flags &= ~VSystem::OPEN_FULLMUTEX;
		}
		else if (!_strcmp(arg, "-uri"))
		{
			if (Tcl_GetBoolean(interp, args[i+1], &b)) return TCL_ERROR;
			if (b)
				flags |= VSystem::OPEN_URI;
			else
				flags &= ~VSystem::OPEN_URI;
		}
		else
		{
			Tcl_AppendResult(interp, "unknown option: ", arg, 0);
			return TCL_ERROR;
		}
	}
	if (argc < 3 || (argc & 1) != 1)
	{
		Tcl_WrongNumArgs(interp, 1, args, "HANDLE FILENAME ?-vfs VFSNAME? ?-readonly BOOLEAN? ?-create BOOLEAN?"
			" ?-nomutex BOOLEAN? ?-fullmutex BOOLEAN? ?-uri BOOLEAN?"
#ifdef HAS_CODEC
			" ?-key CODECKEY?"
#endif
			);
		return TCL_ERROR;
	}
	char *errMsg = nullptr;
	TclContext *p = (TclContext *)Tcl_Alloc(sizeof(*p));
	if (!p)
	{
		Tcl_SetResult(interp, "malloc failed", TCL_STATIC);
		return TCL_ERROR;
	}
	_memset(p, 0, sizeof(*p));
	char *fileName = (char *)args[2];
	//fileName = Tcl_TranslateFileName(interp, fileName, &translatedFilename);
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
		Tcl_SetResult(interp, errMsg, TCL_VOLATILE);
		Tcl_Free(p);
		free(errMsg);
		return TCL_ERROR;
	}
	p->MaxStmt = NUM_PREPARED_STMTS;
	p->Interp = interp;
	arg = args[1];
	Tcl_CreateCommand(interp, (char *)arg, (Tcl_CmdProc *)DbObjCmd, (ClientData)p, DbDeleteCmd);
	return TCL_OK;
}
#pragma endregion

// Initialize this module.
//
// This Tcl module contains only a single new Tcl command named "sqlite". (Hence there is no namespace.  There is no point in using a namespace
// if the extension only supplies one new name!)  The "sqlite" command is used to open a new SQLite database.  See the DbMain() routine above
// for additional information.
__device__ int Main_Init(Tcl_Interp *interp)
{
	Tcl_CreateCommand(interp, "sqlite3", (Tcl_CmdProc *)DbMain, nullptr, nullptr);
	return TCL_OK;
}
__device__ int Main_Shutdown(Tcl_Interp *interp, int flags) { return TCL_OK; }

//// Because it accesses the file-system and uses persistent state, SQLite is not considered appropriate for safe interpreters.  Hence, we deliberately omit the _SafeInit() interfaces.
//#ifndef SQLITE_3_SUFFIX_ONLY
//int Sqlite_Init(Tcl_Interp *interp){ return Sqlite3_Init(interp); }
//int Tclsqlite_Init(Tcl_Interp *interp){ return Sqlite3_Init(interp); }
//int Sqlite_Unload(Tcl_Interp *interp, int flags){ return TCL_OK; }
//int Tclsqlite_Unload(Tcl_Interp *interp, int flags){ return TCL_OK; }
//#endif

#endif
