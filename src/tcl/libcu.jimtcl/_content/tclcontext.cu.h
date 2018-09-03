// A TCL Interface to SQLite
#if HASJIMTCL
#ifndef __TCLCONTEXT_CU_H__
#define __TCLCONTEXT_CU_H__

#include <Core+Vdbe\Core+Vdbe.cu.h>
#include <JimEx.h>
#include <Jim+EventLoop.h>

typedef struct TclContext TclContext;

// New SQL functions can be created as TCL scripts.  Each such function is described by an instance of the following structure.
typedef struct TestSqlFunc TestSqlFunc;
struct SqlFunc
{
	Jim_Interp *Interp;	// The TCL interpret to execute the function
	Jim_Obj *Script;	// The Tcl_Obj representation of the script
	TclContext *Ctx;	// Database connection that owns this function
	bool UseEvalObjv;	// True if it is safe to use Tcl_EvalObjv
	char *Name;			// Name of this function
	SqlFunc *Next;		// Next function on the list of them all
};

// New collation sequences function can be created as TCL scripts.  Each such function is described by an instance of the following structure.
typedef struct SqlCollate SqlCollate;
struct SqlCollate
{
	Jim_Interp *Interp;	// The TCL interpret to execute the function
	char *Script;       // The script to be run
	SqlCollate *Next;   // Next function on the list of them all
};

// Prepared statements are cached for faster execution.  Each prepared statement is described by an instance of the following structure.
typedef struct SqlPreparedStmt SqlPreparedStmt;
struct SqlPreparedStmt
{
	SqlPreparedStmt *Next;  // Next in linked list
	SqlPreparedStmt *Prev;  // Previous on the list
	Vdbe *Stmt;				// The prepared statement
	int SqlLength;          // chars in zSql[]
	const char *Sql;		// Text of the SQL statement
	array_t<Jim_Obj *> Parms; // Array of referenced object pointers
};

typedef struct IncrblobChannel IncrblobChannel;

struct TclContext
{
	Context *Ctx;				// The "real" database structure. MUST BE FIRST
	Jim_Interp *Interp;			// The interpreter used for this database
	char *Busy;					// The busy callback routine
	char *Commit;				// The commit hook callback routine
	char *Trace;				// The trace callback routine
	char *Profile;				// The profile callback routine
	char *Progress;				// The progress callback routine
	char *Auth;					// The authorization callback routine
	int DisableAuth;			// Disable the authorizer if it exists
	char *NullText;				// Text to substitute for an SQL NULL value
	SqlFunc *Funcs;				// List of SQL functions
	Jim_Obj *UpdateHook;		// Update hook script (if any)
	Jim_Obj *RollbackHook;		// Rollback hook script (if any)
	Jim_Obj *WalHook;			// WAL hook script (if any)
	Jim_Obj *UnlockNotify;		// Unlock notify script (if any)
	SqlCollate *Collates;		// List of SQL collation functions
	RC RC;						// Return code of most recent sqlite3_exec()
	Jim_Obj *CollateNeeded;		// Collation needed script
	array_t<SqlPreparedStmt> Stmts;	// List of prepared statements
	SqlPreparedStmt *StmtLast;	// Last statement in the list
	int MaxStmt;				// The next maximum number of stmtList
	IncrblobChannel *Incrblobs;	// Linked list of open incrblob channels
	int Steps, Sorts, Indexs;	// Statistics for most recent operation
	int Transactions;			// Number of nested [transaction] methods
#ifdef _TEST
	bool LegacyPrepare;			// True to use sqlite3_prepare()
#endif
};

struct IncrblobChannel
{
	Blob *Blob;					// sqlite3 blob handle
	TclContext *Ctx;			// Associated database connection
	int Seek;					// Current seek offset
	Jim_Obj *Channel;			// Channel identifier
	IncrblobChannel *Next;		// Linked list of all open incrblob channels
	IncrblobChannel *Prev;		// Linked list of all open incrblob channels
};

#endif // __TCLCONTEXT_CU_H__
#endif