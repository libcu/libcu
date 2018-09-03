// tclExtdInt.h
//
//    Standard internal include file for Extended Tcl library..
//-----------------------------------------------------------------------------
// Copyright 1992 Karl Lehenbauer and Mark Diekhans.
//
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without fee is hereby granted, provided
// that the above copyright notice appear in all copies.  Karl Lehenbauer and Mark Diekhans make no representations about the suitability of this
// software for any purpose.  It is provided "as is" without express or implied warranty.

#ifndef __TCLEXINT_H__
#define __TCLEXINT_H__

#include <timecu.h>
#include "tclEx.h"
#include "tclInt.h"

#ifdef TCL_NEED_SYS_SELECT_H
#include "sys/select.h"
#endif

// If tclUnix.h has already included time.h, don't include it again, some systems don't #ifdef inside of the file.  On some systems, undef
// CLK_TCK (defined in tclUnix.h) to avoid an annoying warning about redefinition.
#ifdef TCL_NEED_TIME_H
#if TCL_SYS_TIME_H
# ifdef TCL_DUP_CLK_TCK
#  undef CLK_TCK
# endif        
# include <time.h>
#endif
#endif

// Precompute milliseconds-per-tick, the " + CLK_TCK / 2" bit gets it to round off instead of truncate.  Take care of defining CLK_TCK if its not defined.
#ifndef CLK_TCK
#ifdef HZ
# define CLK_TCK HZ
#else
# define CLK_TCK 60
#endif
#endif

#define MS_PER_TICK ((1000 + CLK_TCK/2) / CLK_TCK)

// If tclUnix.h did not bring times.h, bring it in here.
#if TCL_GETTOD
#include <sys/times.h>
#endif 

#include <limits.h>
/*#include <grp.h>*/
// On some systems this is not included by tclUnix.h.
// These should be take from an include file, but it got to be such a mess to get the include files right that they are here for good measure.
struct tm *gmtime();
struct tm *localtime();

#ifndef MAXINT
#ifdef INT_MAX
#define MAXINT INT_MAX
#endif
#endif

#ifndef MAXINT
#define BITSPERBYTE   8
#define BITS(type)    (BITSPERBYTE * (int)sizeof(type))
#define HIBITI        ((1 << BITS(int)) - 1)
#define MAXINT        (~HIBITI)
#endif

#ifndef MININT
#ifdef INT_MIN
#define MININT INT_MIN
#else
#define MININT (-MAXINT)-1
#endif
#endif

#ifndef TRUE
#define TRUE   (1)
#define FALSE  (0)
#endif

// Structure to hold a regular expression, plus a Boyer-Moore compiled pattern.
typedef struct regexp_t {
	regex_t progPtr;
	char   *boyerMoorePtr;
	int     noCase;
} regexp_t;
typedef regexp_t *regexp_pt;

// Flags used by RegExpCompile:
#define REXP_NO_CASE         1   // Do matching regardless of case
#define REXP_BOTH_ALGORITHMS 2   // Use boyer-moore along with regexp

// Data structure to control a dynamic buffer.  These buffers are primarly used for reading things from files, were the maximum size is not known
// in advance, and the buffer must grow.  These are used in the case were the value is not to be returned as the interpreter result.
#define INIT_DYN_BUFFER_SIZE 256

typedef struct dynamicBuf_t {
	char  buf[INIT_DYN_BUFFER_SIZE];   // Initial buffer area.             
	char *ptr;                          // Pointer to buffer area.          
	int   size;                         // Current size of buffer.          
	int   len;                          // Current string length (less '\0')
} dynamicBuf_t;

// Used to return argument messages by most commands.
extern char *tclXWrongArgs;

// Macros to do string compares.  They pre-check the first character before checking of the strings are equal.
#define STREQU(str1, str2) (((str1)[0] == (str2)[0]) && (!strcmp(str1, str2)))
#define STRNEQU(str1, str2, cnt) (((str1)[0] == (str2)[0]) && (!strncmp(str1, str2, cnt)))

// Prototypes for utility procedures.
__device__ void Tcl_DynBufInit(dynamicBuf_t *dynBufPtr);
__device__ void Tcl_DynBufFree(dynamicBuf_t *dynBufPtr);
__device__ void Tcl_DynBufReturn(Tcl_Interp *interp, dynamicBuf_t *dynBufPtr);
__device__ void Tcl_DynBufAppend(dynamicBuf_t *dynBufPtr, char *newStr);
__device__ void Tcl_ExpandDynBuf(dynamicBuf_t *dynBufPtr, int appendSize);
__device__ int Tcl_DynamicFgets(dynamicBuf_t *dynBufPtr, FILE *filePtr, int append);
__device__ int Tcl_ConvertFileHandle(Tcl_Interp *interp, char *handle);
__device__ time_t Tcl_GetDate(char *p, time_t now, long zone);
__device__ int Tcl_ProcessSignal(Tcl_Interp *interp, int cmdResultCode);
__device__ void Tcl_RegExpClean(regexp_pt regExpPtr);
__device__ int Tcl_RegExpCompile(Tcl_Interp *interp, regexp_pt regExpPtr, char *expression, int flags);
__device__ int Tcl_RegExpExecute(Tcl_Interp *interp, regexp_pt regExpPtr, char *matchStrIn, char *matchStrLower);
__device__ void Tcl_ResetSignals();
__device__ int Tcl_ReturnDouble(Tcl_Interp *interp, double number);
__device__ int Tcl_SetupFileEntry(Tcl_Interp *interp, int fileNum, int readable, int writable);
__device__ void Tcl_SetupSigInt();

// Definitions required to initialize all extended commands.  These are either the command executors or initialization routines that do the command
// initialization.  The initialization routines are used when there is more to initializing the command that just binding the command name to the
// executor.  Usually, this means initializing some command local data via the ClientData mechanism.  The command executors should be declared to be of
// type `Tcl_CmdProc', but this blows up some compilers, so they are declared with an ANSI prototype.

// from tclXbsearch.c
extern __device__ int Tcl_BsearchCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXchmod.c
extern __device__ int Tcl_ChmodCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_ChownCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_ChgrpCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXclock.c
extern __device__ int Tcl_GetclockCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FmtclockCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXcnvclock.c
extern __device__ int Tcl_ConvertclockCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXcmdloop.c
extern __device__ int Tcl_CommandloopCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXdebug.c
extern __device__ void TclEx_InitDebug(Tcl_Interp *interp);

// from tclXgen.c
extern __device__ void TclEx_InitGeneral(Tcl_Interp *interp);

// from tclXdup.c
extern __device__ int Tcl_DupCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXfcntl.c
extern __device__ int Tcl_FcntlCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXfilecmds.c
extern __device__ int Tcl_PipeCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_CopyfileCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FstatCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LgetsCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FlockCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FunlockCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXfilescan.c
extern __device__ void Tcl_InitFilescan(Tcl_Interp *interp);

// from tclXfmath.c
extern __device__ int Tcl_AcosCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_AsinCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_AtanCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_CosCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_SinCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_TanCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_CoshCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_SinhCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_TanhCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_ExpCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LogCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_Log10Cmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_SqrtCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FabsCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FloorCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_CeilCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_FmodCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_PowCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXgeneral.c
extern __device__ int Tcl_EchoCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_InfoxCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LoopCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXid.c
extern __device__ int Tcl_IdCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXkeylist.c
extern __device__ int Tcl_KeyldelCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_KeylgetCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_KeylkeysCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_KeylsetCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXlist.c
extern __device__ int Tcl_LvarpopCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LvarcatCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LvarpushCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LemptyCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXmath.c
extern __device__ int Tcl_MaxCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_MinCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_RandomCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXmsgcat.c
extern __device__ void Tcl_InitMsgCat(Tcl_Interp *interp);

// from tclXprocess.c
extern __device__ int Tcl_ExeclCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_ForkCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_WaitCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXprofile.c
extern __device__ void Tcl_InitProfile(Tcl_Interp *interp);

// from tclXselect.c
extern __device__ int Tcl_SelectCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXsignal.c
extern __device__ void Tcl_InitSignalHandling(Tcl_Interp *interp);

// from tclXstring.c
extern __device__ int Tcl_CindexCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_ClengthCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_CrangeCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_ReplicateCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_TranslitCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_CtypeCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXlib.c
extern __device__ int Tcl_Demand_loadCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LoadlibindexCmd(ClientData, Tcl_Interp *, int, const char *[]);

// from tclXunixcmds.c
extern __device__ int Tcl_AlarmCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_SleepCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_SystemCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_TimesCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_UmaskCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_LinkCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_UnlinkCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_MkdirCmd(ClientData, Tcl_Interp *, int, const char *[]);
extern __device__ int Tcl_RmdirCmd(ClientData, Tcl_Interp *, int, const char *[]);

#endif /* __TCLEXINT_H__ */
