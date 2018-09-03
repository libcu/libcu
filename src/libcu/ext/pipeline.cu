// https://msdn.microsoft.com/en-us/library/windows/desktop/ms682499(v=vs.85).aspx
#include <ext/pipeline.h>
#include <stddefcu.h>
#include <stdio.h>
#include <stdlib.h>
#include <errnocu.h>

// PORTABILITY
#pragma region PORTABILITY
#if __OS_WIN
#ifndef STRICT
#define STRICT
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <string>

typedef HANDLE FDTYPE;
typedef HANDLE PIDTYPE;
#define __BAD_FD INVALID_HANDLE_VALUE
#define __BAD_PID INVALID_HANDLE_VALUE

#define WIFEXITED(STATUS) 1
#define WEXITSTATUS(STATUS) (STATUS)
#define WIFSIGNALED(STATUS) 0
#define WTERMSIG(STATUS) 0
#define WNOHANG 1
static SECURITY_ATTRIBUTES *WinStdSecAttrs() { static SECURITY_ATTRIBUTES secAtts; secAtts.nLength = sizeof(SECURITY_ATTRIBUTES); secAtts.lpSecurityDescriptor = NULL; secAtts.bInheritHandle = TRUE; return &secAtts; }

static int __Pipe(FDTYPE pipefd[2]) { return CreatePipe(&pipefd[0], &pipefd[1], NULL, 0) ? 0 : -1; }
static FDTYPE __Fileno(FILE *fh) { return (FDTYPE)_get_osfhandle(_fileno(fh)); }
// __Read
#define __Close CloseHandle
static PIDTYPE __WaitPid(PIDTYPE pid, int *status, int nohang) {
	DWORD ret = WaitForSingleObject(pid, nohang ? 0 : INFINITE);
	if (ret == WAIT_TIMEOUT || ret == WAIT_FAILED) return __BAD_PID; // WAIT_TIMEOUT can only happend with WNOHANG
	GetExitCodeProcess(pid, &ret); *status = ret; CloseHandle(pid); return pid;
}
static FDTYPE __Dup(PIDTYPE infd) { FDTYPE dupfd; PIDTYPE pid = GetCurrentProcess(); return (DuplicateHandle(pid, infd, pid, &dupfd, 0, TRUE, DUPLICATE_SAME_ACCESS) ? dupfd : __BAD_FD); }
//static FILE *__Fdopen_r(FDTYPE fd) { return _fdopen(_open_osfhandle((int)fd, _O_RDONLY | _O_TEXT), "r"); }
static FILE *__Fdopen_w(FDTYPE fd) { return _fdopen(_open_osfhandle((int)fd, _O_TEXT), "w"); }
static FDTYPE __Open_r(const char *filename) { return CreateFile(filename, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, WinStdSecAttrs(), OPEN_EXISTING, 0, NULL); }
static FDTYPE __Open_w(const char *filename, int append) { return CreateFile(filename, append ? FILE_APPEND_DATA : GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, WinStdSecAttrs(), append ? OPEN_ALWAYS : CREATE_ALWAYS, 0, (HANDLE)NULL); }
//static int __Rewind(FDTYPE fd) { return SetFilePointer(fd, 0, NULL, FILE_BEGIN) == INVALID_SET_FILE_POINTER ? -1 : 0; }

static HANDLE __CreateTemp(const char *contents, int length) {
	char name[MAX_PATH];
	if (!GetTempPath(MAX_PATH, name) || !GetTempFileName(name, "JIM", 0, name)) return __BAD_FD;
	HANDLE handle = CreateFile(name, GENERIC_READ | GENERIC_WRITE, 0, WinStdSecAttrs(), CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
	if (handle == INVALID_HANDLE_VALUE) goto error;
	if (contents) {
		FILE *fh = __Fdopen_w(__Dup(handle)); // Use fdopen() to get automatic text-mode translation
		if (!fh) goto error;
		if (fwrite(contents, length, 1, fh) != 1) { fclose(fh); goto error; }
		fseek(fh, 0, SEEK_SET);
		fclose(fh);
	}
	return handle;
error:
	printf("failed to create temp file\n");
	CloseHandle(handle);
	DeleteFile(name);
	return __BAD_FD;
}

static int WinFindExecutable(const char *originalName, char fullPath[MAX_PATH]) {
	static char extensions[][5] = { ".exe", "", ".bat" };
	for (int i = 0; i < ARRAYSIZE_(extensions); i++) {
		lstrcpyn(fullPath, originalName, MAX_PATH - 5);
		lstrcat(fullPath, extensions[i]);
		if (!SearchPath(NULL, fullPath, NULL, MAX_PATH, fullPath, NULL) || (GetFileAttributes(fullPath) & FILE_ATTRIBUTE_DIRECTORY))
			continue;
		return 0;
	}
	return -1;
}

static char *WinBuildCommandLine(char **argv) {
	std::string str;
	bool quote = false;
	char *start;
	for (int i = 0; argv[i]; i++) {
		if (i > 0) str.append(" ", 1);
		if (argv[i][0] == '\0') quote = true;
		else {
			quote = false;
			for (start = argv[i]; *start != '\0'; start++)
				if (isspace(UCHAR(*start))) { quote = true; break; }
		}
		if (quote) str.append("\"", 1);

		start = argv[i];
		char *special; for (special = argv[i]; ; ) {
			if (*special == '\\' && (special[1] == '\\' || special[1] == '"' || (quote && special[1] == '\0'))) {
				str.append(start, special - start);
				start = special;
				while (1) {
					special++;
					if (*special == '"' || (quote && *special == '\0')) {
						// N backslashes followed a quote -> insert
						// N * 2 + 1 backslashes then a quote.
						str.append(start, special - start);
						break;
					}
					if (*special != '\\')
						break;
				}
				str.append(start, special - start);
				start = special;
			}
			if (*special == '"') {
				if (special == start) str.append("\"", 1);
				else str.append(start, special - start);
				str.append("\\\"", 2);
				start = special + 1;
			}
			if (*special == '\0')
				break;
			special++;
		}
		str.append(start, special - start);
		if (quote) str.append("\"", 1);
	}

	char *cstr = new char[str.length() + 1];
	std::strcpy(cstr, str.c_str());
	return cstr;
}

static PIDTYPE __StartProcess(char **argv, char *env, FDTYPE inputId, FDTYPE outputId, FDTYPE errorId) {
	char execPath[MAX_PATH];
	if (WinFindExecutable(argv[0], execPath) < 0)
		return __BAD_PID;
	argv[0] = execPath;
	HANDLE hProcess = GetCurrentProcess();
	char *cmdLine = WinBuildCommandLine(argv);
	PIDTYPE pid = __BAD_PID;

	// STARTF_USESTDHANDLES must be used to pass handles to child process. Using SetStdHandle() and/or dup2() only works when a console mode
	// parent process is spawning an attached console mode child process.
	STARTUPINFO startInfo;
	ZeroMemory(&startInfo, sizeof(startInfo));
	startInfo.cb = sizeof(startInfo);
	startInfo.dwFlags = STARTF_USESTDHANDLES;
	startInfo.hStdInput = INVALID_HANDLE_VALUE;
	startInfo.hStdOutput = INVALID_HANDLE_VALUE;
	startInfo.hStdError = INVALID_HANDLE_VALUE;

	// Duplicate all the handles which will be passed off as stdin, stdout and stderr of the child process. The duplicate handles are set to
	// be inheritable, so the child process can use them.
	if (inputId == __BAD_FD) {
		HANDLE h;
		if (CreatePipe(&startInfo.hStdInput, &h, WinStdSecAttrs(), 0) != FALSE) CloseHandle(h);
	}
	else DuplicateHandle(hProcess, inputId, hProcess, &startInfo.hStdInput, 0, TRUE, DUPLICATE_SAME_ACCESS);
	if (startInfo.hStdInput == __BAD_FD) goto end;
	if (outputId == __BAD_FD) {
		startInfo.hStdOutput = CreateFile("NUL:", GENERIC_WRITE, 0, WinStdSecAttrs(), OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	}
	else DuplicateHandle(hProcess, outputId, hProcess, &startInfo.hStdOutput, 0, TRUE, DUPLICATE_SAME_ACCESS);
	if (startInfo.hStdOutput == __BAD_FD) goto end;
	if (errorId == __BAD_FD) { // If handle was not set, errors should be sent to an infinitely deep sink.
		startInfo.hStdError = CreateFile("NUL:", GENERIC_WRITE, 0, WinStdSecAttrs(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	}
	else DuplicateHandle(hProcess, errorId, hProcess, &startInfo.hStdError, 0, TRUE, DUPLICATE_SAME_ACCESS);
	if (startInfo.hStdError == __BAD_FD) goto end;

	// "When an application spawns a process repeatedly, a new thread instance will be created for each process but the previous
	// instances may not be cleaned up.  This results in a significant virtual memory loss each time the process is spawned.  If there
	// is a WaitForInputIdle() call between CreateProcess() and CloseHandle(), the problem does not occur." PSS ID Number: Q124121
	PROCESS_INFORMATION procInfo;
	if (!CreateProcess(NULL, cmdLine, NULL, NULL, TRUE, 0, env, NULL, &startInfo, &procInfo)) goto end;
	WaitForInputIdle(procInfo.hProcess, 5000);
	CloseHandle(procInfo.hThread);
	pid = procInfo.hProcess;
end:
	free(cmdLine);
	if (startInfo.hStdInput != __BAD_FD) CloseHandle(startInfo.hStdInput);
	if (startInfo.hStdOutput != __BAD_FD) CloseHandle(startInfo.hStdOutput);
	if (startInfo.hStdError != __BAD_FD) CloseHandle(startInfo.hStdError);
	return pid;
}

#elif __OS_UNIX
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/stat.h>

typedef int FDTYPE;
typedef int PIDTYPE;
#define __BAD_FD -1
#define __BAD_PID -1

#define __Pipe pipe
#define __Fileno fileno
#define __Read read
#define __Close close
#define __WaitPid waitpid
#define __Dup dup
#define __Fdopen_r(FD) fdopen((FD), "r")
#define __Open_r(NAME) open((NAME), O_RDONLY, 0)
static int __Open_w(const char *filename, int append) { return open(filename, O_WRONLY | O_CREAT | (append ? O_APPEND : O_TRUNC), 0666); }
static int __Rewind(FDTYPE fd) { return lseek(fd, 0L, SEEK_SET); }

static int __CreateTemp(const char *contents, int length) {
#define TMP_STDIN_NAME "/tmp/tcl.in.XXXXXX"
	char inName[sizeof(TMP_STDIN_NAME) + 1];
	strcpy(inName, TMP_STDIN_NAME);
	int fd;
#ifdef HAVE_MKSTEMP
	fd = mkstemp(inName);
#else
	mktemp(inName);
	fd = open(inName, O_RDWR | O_CREAT | O_TRUNC, 0600);
#endif
	if (fd < 0) {
		printf("couldn't create input file for command: %s\n", strerror(errno));
		return -1;
	}
	if (contents) {
		if (write(fd, input, length) != length) {
			printf("couldn't write file input for command:  %s\n", strerror(errno));
			close(fd);
			return -1;
		}
		if ((lseek(fd, 0L, SEEK_SET) == -1) || (unlink(inName) == -1)) {
			printf("couldn't reset or remove input file for command:  %s\n", strerror(errno));
			close(fd);
			return -1;
		}
	}
}

#ifndef HAVE_EXECVPE
#define execvpe(ARG0, ARGV, ENV) execvp(ARG0, ARGV)
#endif

static PIDTYPE __StartProcess(char **argv, char *env, FDTYPE inputId, FDTYPE outputId, FDTYPE errorId) {
	PDTYPE pid = vfork();
	if (pid < 0) return __BAD_PID;
	if (pid == 0) {
		// Child
		if (inputId != __BAD_FD) dup2(inputId, 0);
		if (outputId != __BAD_FD) dup2(outputId, 1);
		if (errorId != __BAD_FD) dup2(errorId, 2);
		for (int i = 3; i <= outputId || i <= inputId || i <= errorId; i++)
			close(i);
		// Restore SIGPIPE behaviour
		signal(SIGPIPE, SIG_DFL);
		execvpe(argv[0], &argv[0], env);
		// Need to prep an error message before vfork(), just in case
		fprintf(stderr, "couldn't exec \"%s\"\n", argv[0]);
		_exit(127);
	}
}

#endif
#pragma endregion

/*
* Flag bits in WaitInfo structures:
*
* WI_READY -			Non-zero means process has exited or suspended since it was forked or last returned by Tcl_WaitPids.
* WI_DETACHED -		Non-zero means no-one cares about the process anymore.  Ignore it until it exits, then forget about it.
*/
#define WI_READY	1
#define WI_DETACHED	2
#define WAIT_TABLE_GROW_BY 4

// Data structures of the following type are used by Tcl_Fork and Tcl_WaitPids to keep track of child processes.
struct WaitInfo_t {
	PIDTYPE pid;                // Process id of child.
	int status;                 // Status returned when child exited or suspended.
	int flags;                  // Various flag bits;  see below for definitions.
};

static struct WaitInfoTable_t {
	struct WaitInfo_t *table;   // Table of outstanding processes
	int size;                   // Size of the allocated table
	int used;                   // Number of entries in use
} _waitInfo = { nullptr, 0, 0 };

typedef struct OpenFile {
	FILE *f;			// Stdio file to use for reading and/or writing.
	FILE *f2;			// Normally NULL.  In the special case of a command pipeline with pipes for both input and output, this is a stdio file to use for writing to the pipeline.
	int readable;		// Non-zero means file may be read.
	int writable;		// Non-zero means file may be written.
	int numPids;		// If this is a connection to a process pipeline, gives number of processes in pidPtr array below;  otherwise it is 0.
	int *pidPtr;		// Pointer to malloc-ed array of child process ids (numPids of them), or NULL if this isn't a connection to a process pipeline.
	int errorId;		// File id of file that receives error output from pipeline.  -1 means not used (i.e. this is a normal file).
} OpenFile_t;

/* Create a new process using the vfork system call, and keep track of it for "safe" waiting with Tcl_WaitPids. */
PIDTYPE StartProcess(char **argv, char *env, FDTYPE inputId, FDTYPE outputId, FDTYPE errorId)
{
#if __OS_UNIX
	// Disable SIGPIPE signals:  if they were allowed, this process might go away unexpectedly if children misbehave.  This code
	// can potentially interfere with other application code that expects to handle SIGPIPEs;  what's really needed is an
	// arbiter for signals to allow them to be "shared".
	if (!_waitInfo.table)
		signal(SIGPIPE, SIG_IGN);
#endif

	// Enlarge the wait table if there isn't enough space for a new entry.
	if (_waitInfo.used == _waitInfo.size) {
		int newSize = _waitInfo.size + WAIT_TABLE_GROW_BY;
		WaitInfo_t *newTable = (WaitInfo_t *)malloc(newSize * sizeof(WaitInfo_t));
		memcpy(newTable, _waitInfo.table, (_waitInfo.size * sizeof(WaitInfo_t)));
		if (newTable)
			free(_waitInfo.table);
		_waitInfo.table = newTable;
		_waitInfo.size = newSize;
	}

	// Make a new process and enter it into the table if the fork is successful.
	WaitInfo_t *info = &_waitInfo.table[_waitInfo.used];
	PIDTYPE pid = __StartProcess(&argv[0], env, inputId, outputId, errorId);
	if (pid == __BAD_PID) {
		printf("couldn't exec \"%s\"", argv[0]);
		return __BAD_PID;
	}
	info->pid = pid;
	info->flags = 0;
	_waitInfo.used++;
	return pid;
}

static void ReapDetachedPids()
{
	int count, dest = 0;
	register WaitInfo_t *info;
	for (info = _waitInfo.table, count = _waitInfo.used; count > 0; info++, count--) {
		if (info->flags & WI_DETACHED) {
			int status;
			PIDTYPE pid = __WaitPid(info->pid, &status, WNOHANG);
			if (pid == info->pid) { _waitInfo.used--; continue; } // Process has exited, so remove it from the table
		}
		if (info != &_waitInfo.table[dest]) _waitInfo.table[dest] = *info;
		dest++;
	}
}

static PIDTYPE WaitForProcess(PIDTYPE pid, int *statusPtr)
{
	for (int i = 0; i < _waitInfo.used; i++) {
		if (pid == _waitInfo.table[i].pid) {
			__WaitPid(pid, statusPtr, 0); // wait for it
			if (i != _waitInfo.used - 1) // Remove it from the table
				_waitInfo.table[i] = _waitInfo.table[_waitInfo.used - 1];
			_waitInfo.used--;
			return pid;
		}
	}
	return __BAD_PID;
}

static void DetachPids(int numPids, const PIDTYPE *pids)
{
	for (int i = 0; i < _waitInfo.used; i++)
		for (int j = 0; j < numPids; j++) {
			if (pids[j] == _waitInfo.table[i].pid) {
				_waitInfo.table[i].flags |= WI_DETACHED;
				break;
			}
		}
}

static int CleanupChildren(int numPids, PIDTYPE *pids, int child_siginfo)
{
	int result = 0;
	for (int i = 0; i < numPids; i++) {
		int waitStatus = 0;
		if (WaitForProcess(pids[i], &waitStatus) != __BAD_PID) {
			if (WIFEXITED(waitStatus) && !WEXITSTATUS(waitStatus)) {
				continue;
			}
			if (WIFEXITED(waitStatus)) {
				printf("child:exit: %d\n", WEXITSTATUS(waitStatus));
			}
			else {
				if (WIFSIGNALED(waitStatus)) printf("child: killed\n");
				else printf("child: suspended\n");
			}
			result = -1;
		}
	}
	return result;
}

//
//int TclGetOpenFile(char *string, OpenFile **filePtrPtr)
//{
//	int fd = 0; // Initial value needed only to stop compiler warnings.
//	if (string[0] == 'f' && string[1] == 'i' && string[2] == 'l' && string[3] == 'e') {
//		char *end;
//		fd = strtoul(string+4, &end, 10);
//		if (end == string+4 || *end != 0)
//			goto badId;
//	}
//	else if (string[0] == 's' && string[1] == 't' && string[2] == 'd') {
//		if (!strcmp(string+3, "in")) fd = fileno(stdin);
//		else if (!strcmp(string+3, "out")) fd = fileno(stdout);
//		else if (!strcmp(string+3, "err")) fd = fileno(stderr);
//		else goto badId;
//	}
//	else {
//badId:
//		printf("bad file identifier \"%s\"\n", string);
//		return -1;
//	}
//#ifdef DEBUG_FDS
//	syslog(LOG_INFO, "TclGetOpenFile(%s), fd=%d, numFiles=%d", string, fd, iPtr->numFiles);
//#endif
//	if (iPtr->numFiles == 0) {
//		TclMakeFileTable(iPtr, fd);
//	}
//	if (fd >= iPtr->numFiles || iPtr->filePtrArray[fd] == NULL) {
//		printf("file \"%s\" isn't open\n", string);
//		return -1;
//	}
//#ifdef DEBUG_FDS
//	syslog(LOG_INFO, "TclGetOpenFile(%s): filePtrArray[%d]=%p", string, fd, iPtr->filePtrArray[fd]);
//#endif
//	*filePtrPtr = iPtr->filePtrArray[fd];
//	return -1;
//}

static FILE *GetAioFilehandle(const char *input) {
	return nullptr;
}

#define FILE_NAME   0           /* input/output: filename */
#define FILE_APPEND 1           /* output only:  filename, append */
#define FILE_HANDLE 2           /* input/output: filehandle */
#define FILE_TEXT   3           /* input only:   input is actual text */

static int CreatePipeline(int argc, char **argv, PIDTYPE **pidsPtr, FDTYPE *inPipePtr, FDTYPE *outPipePtr, FDTYPE *errFilePtr)
{
	ReapDetachedPids();
	if (inPipePtr != NULL) *inPipePtr = __BAD_FD;
	if (outPipePtr != NULL) *outPipePtr = __BAD_FD;
	if (errFilePtr != NULL) *errFilePtr = __BAD_FD;

	// First, scan through all the arguments to figure out the structure of the pipeline.  Count the number of distinct processes (it's the
	// number of "|" arguments).  If there are "<", "<<", or ">" arguments then make note of input and output redirection and remove these
	// arguments and the arguments that follow them.
	if (!argc) {
		printf("didn't specify command to execute", -1);
		return -1;
	}
	const char *input = nullptr;	// Describes input for pipeline, depending on "inputFile".  NULL means take input from stdin/pipe.
	int inputFile = FILE_NAME;		// 1 means input is name of input file.
	// 2 means input is filehandle name.
	// 0 means input holds actual text to be input to command. */
	const char *output = nullptr;	// Holds name of output file to pipe to, or NULL if output goes to stdout/pipe.
	int outputFile = FILE_NAME;		// 0 means output is the name of output file.
	// 1 means output is the name of output file, and append.
	// 2 means output is filehandle name.
	// All this is ignored if output is NULL */
	const char *error = nullptr;	// Holds name of stderr file to pipe to, or NULL if stderr goes to stderr/pipe.
	int errorFile = FILE_NAME;		// 0 means error is the name of error file.
	// 1 means error is the name of error file, and append.
	// 2 means error is filehandle name.
	// All this is ignored if error is NULL */
	int cmdCount = 1; // Count of number of distinct commands found in argc/argv.
	int lastBar = -1;
	for (int i = 0; i < argc; i++) {
		const char *arg = argv[i];
		if ((arg[0] == '|' && !arg[1]) || (arg[0] == '|' && arg[1] == '&' && !arg[2])) {
			if (i == lastBar + 1 || i == argc - 1) { printf("illegal use of | or |& in command"); return -1; }
			lastBar = i;
			cmdCount++;
			continue;
		}
		else if (arg[0] == '<') {
			input = arg + 1;
			inputFile = FILE_NAME;
			if (*input == '<') { input++; inputFile = FILE_TEXT; }
			else if (*input == '@') { input++; inputFile = FILE_HANDLE; }
			if (!*input && ++i < argc) { input = argv[i]; }
		}
		else if (arg[0] == '>') {
			bool dup_error = false;
			output = arg + 1;
			outputFile = FILE_NAME;
			if (*output == '>') { outputFile = FILE_APPEND; output++; }
			if (*output == '&') { output++; dup_error = true; } // Redirect stderr too 
			if (*output == '@') { outputFile = FILE_HANDLE; output++; }
			if (!*output && ++i < argc) { output = argv[i]; }
			if (dup_error) { errorFile = outputFile; error = output; }
		}
		else if (arg[0] == '2' && arg[1] == '>') {
			error = arg + 2;
			errorFile = FILE_NAME;
			if (*error == '@') { error++; errorFile = FILE_HANDLE; }
			else if (*error == '>') { error++; errorFile = FILE_APPEND; }
			if (!*error && ++i < argc) { error = argv[i]; }
		}
		else continue;
		if (i >= argc) { printf("can't specify \"%s\" as last word in command", arg); return -1; }
	}

	/* Must do this before vfork(), so do it now */
	//save_environ = JimSaveEnv(JimBuildEnv(interp));

	FDTYPE pipeIds[2] = { __BAD_FD, __BAD_FD }; // File ids for pipe that's being created.
	FDTYPE inputId = __BAD_FD;			// Readable file id input to current command in pipeline (could be file or pipe).  JIM_BAD_FD means use stdin.
	FDTYPE outputId = __BAD_FD;			// Writable file id for output from current command in pipeline (could be file or pipe). JIM_BAD_FD means use stdout.
	FDTYPE errorId = __BAD_FD;			// Writable file id for all standard error output from all commands in pipeline.  JIM_BAD_FD means use stderr.
	FDTYPE lastOutputId = __BAD_FD;		// Write file id for output from last command in pipeline (could be file or pipe). -1 means use stdout.

	// Set up the redirected input source for the pipeline, if so requested.
	if (input) {
		if (inputFile == FILE_TEXT) { // Immediate data in command.  Create temporary file and put data into file.
			inputId = __CreateTemp(input, (int)strlen(input));
			if (inputId == __BAD_FD) goto error;
		}
		else if (inputFile == FILE_HANDLE) { // Should be a file descriptor
			FILE *fh = GetAioFilehandle(input);
			if (!fh) goto error;
			inputId = __Dup(__Fileno(fh));
		}
		else { // File redirection.  Just open the file.
			inputId = __Open_r(input);
			if (inputId == __BAD_FD) { printf("couldn't read file \"%s\": %s\n", input, __Strerror()); goto error; }
		}
	}
	else if (inPipePtr) {
		if (__Pipe(pipeIds) != 0) { printf("couldn't create input pipe for command"); goto error; }
		inputId = pipeIds[0];
		*inPipePtr = pipeIds[1];
		pipeIds[0] = pipeIds[1] = __BAD_FD;
	}

	// Set up the redirected output sink for the pipeline from one of two places, if requested.
	if (output) {
		if (outputFile == FILE_HANDLE) {
			FILE *fh = GetAioFilehandle(output);
			if (!fh) goto error;
			fflush(fh);
			lastOutputId = __Dup(__Fileno(fh));
		}
		else { // Output is to go to a file.
			lastOutputId = __Open_w(output, outputFile == FILE_APPEND);
			if (lastOutputId == __BAD_FD) { printf("couldn't write file \"%s\": %s\n", output, __Strerror()); goto error; }
		}
	}
	else if (outPipePtr) { // Output is to go to a pipe.
		if (__Pipe(pipeIds) != 0) { printf("couldn't create output pipe"); goto error; }
		lastOutputId = pipeIds[1];
		*outPipePtr = pipeIds[0];
		pipeIds[0] = pipeIds[1] = __BAD_FD;
	}

	// If we are redirecting stderr with 2>filename or 2>@fileId, then we ignore errFilePtr
	if (error) {
		if (errorFile == FILE_HANDLE) {
			if (error[0] == '1' && !error[1]) {
				if (lastOutputId != __BAD_FD) errorId = __Dup(lastOutputId); // Special 2>@1
				else error = "stdout"; // No redirection of stdout, so just use 2>@stdout 
			}
			if (errorId == __BAD_FD) {
				FILE *fh = GetAioFilehandle(error);
				if (!fh)
					goto error;
				fflush(fh);
				errorId = __Dup(__Fileno(fh));
			}
		}
		else { // Output is to go to a file.
			errorId = __Open_w(error, errorFile == FILE_APPEND);
			if (errorId == __BAD_FD) {
				printf("couldn't write file \"%s\": %s\n", error, __Strerror());
				goto error;
			}
		}
	}
	else if (errFilePtr) {
		// Set up the standard error output sink for the pipeline, if requested.  Use a temporary file which is opened, then deleted.
		// Could potentially just use pipe, but if it filled up it could cause the pipeline to deadlock:  we'd be waiting for processes
		// to complete before reading stderr, and processes couldn't complete because stderr was backed up.
		errorId = __CreateTemp(nullptr, 0);
		if (errorId == __BAD_FD) goto error;
		*errFilePtr = __Dup(errorId);
	}

	// Scan through the argc array, forking off a process for each group of arguments between "|" arguments.
	PIDTYPE *pids = (PIDTYPE *)malloc(cmdCount * sizeof(PIDTYPE)); // Points to malloc-ed array holding all the pids of child processes.
	for (int i = 0; i < cmdCount; i++)
		pids[i] = __BAD_PID;
	int numPids = 0; // Actual number of processes that exist at *pids right now.
	int firstArg, lastArg; // Indexes of first and last arguments in current command.
	for (firstArg = 0; firstArg < argc; numPids++, firstArg = lastArg + 1) {
		bool pipe_dup_err = false;
		FDTYPE origErrorId = errorId;
		for (lastArg = firstArg; lastArg < argc; lastArg++) {
			if (argv[lastArg][0] == '|') {
				if (argv[lastArg][1] == '&') pipe_dup_err = true;
				break;
			}
		}
		argv[lastArg] = 0; // Replace | with NULL for execv()
		if (lastArg == argc) {
			outputId = lastOutputId;
		}
		else {
			if (__Pipe(pipeIds) != 0) { printf("couldn't create pipe"); goto error; }
			outputId = pipeIds[1];
		}
		if (pipe_dup_err) errorId = outputId;

		PIDTYPE pid = StartProcess(argv, nullptr, inputId, outputId, errorId);
		pids[numPids] = pid;

		// Restore in case of pipe_dup_err
		errorId = origErrorId;

		// Close off our copies of file descriptors that were set up for this child, then set up the input for the next child.
		if (inputId != __BAD_FD) __Close(inputId);
		if (outputId != __BAD_FD) __Close(outputId);
		inputId = pipeIds[0];
		pipeIds[0] = pipeIds[1] = __BAD_FD;
	}
	*pidsPtr = pids;

	// All done.  Cleanup open files lying around and then return.
cleanup:
	if (inputId != __BAD_FD) __Close(inputId);
	if (lastOutputId != __BAD_FD) __Close(lastOutputId);
	if (errorId != __BAD_FD) __Close(errorId);
	//JimRestoreEnv(save_environ);
	return numPids;

	// An error occurred.  There could have been extra files open, such as pipes between children.  Clean them all up.  Detach any child processes that have been created.
error:
	if (inPipePtr && *inPipePtr != __BAD_FD) { __Close(*inPipePtr); *inPipePtr = __BAD_FD; }
	if (outPipePtr && *outPipePtr != __BAD_FD) { __Close(*outPipePtr); *outPipePtr = __BAD_FD; }
	if (errFilePtr && *errFilePtr != __BAD_FD) { __Close(*errFilePtr); *errFilePtr = __BAD_FD; }
	if (pipeIds[0] != __BAD_FD) { __Close(pipeIds[0]); }
	if (pipeIds[1] != __BAD_FD) { __Close(pipeIds[1]); }
	if (pids) {
		for (int i = 0; i < numPids; i++)
			if (pids[i] != __BAD_PID)
				DetachPids(1, &pids[i]);
		free(pids);
	}
	numPids = -1;
	goto cleanup;
}

#pragma region tinytcl

///* This procedure is used to wait for one or more processes created by Tcl_Fork to exit or suspend. */
//PIDTYPE Tcl_WaitPids(int numPids, PIDTYPE *pids, int *statusPtr)
//{
//	while (1) {
//		// Scan the table of child processes to see if one of the specified children has already exited or suspended.  If so,
//		// remove it from the table and return its status.
//		PIDTYPE pid;
//		bool anyProcesses = false;
//		register WaitInfo_t *info; int count;
//		for (info = _waitInfo.table, count = _waitInfo.used; count > 0; info++, count--) {
//			for (int i = 0; i < numPids; i++) {
//				if (pids[i] != info->pid)
//					continue;
//				anyProcesses = true;
//				if (info->flags & WI_READY) {
//					*statusPtr = *((int *)&info->status);
//					pid = info->pid;
//					if (WIFEXITED(info->status) || WIFSIGNALED(info->status)) {
//						*info = _waitInfo.table[_waitInfo.used - 1];
//						_waitInfo.used--;
//					}
//					else info->flags &= ~WI_READY;
//					return pid;
//				}
//			}
//		}
//
//		// Make sure that the caller at least specified one valid process to wait for.
//		if (!anyProcesses) {
//			errno = ECHILD;
//			return __BAD_PID;
//		}
//
//		// Wait for a process to exit or suspend, then update its entry in the table and go back to the beginning of the
//		// loop to see if it's one of the desired processes.
//		int status;
//		pid = wait(&status);
//		if (pid < 0)
//			return pid;
//		for (info = _waitInfo.table, count = _waitInfo.used; ; info++, count--) {
//			if (count == 0)
//				break; // Ignore unknown processes.
//			if (pid != info->pid)
//				continue;
//			// If the process has been detached, then ignore anything other than an exit, and drop the entry on exit.
//			if (info->flags & WI_DETACHED) {
//				if (WIFEXITED(status) || WIFSIGNALED(status)) {
//					*info = _waitInfo.table[_waitInfo.used - 1];
//					_waitInfo.used--;
//				}
//			}
//			else {
//				info->status = status;
//				info->flags |= WI_READY;
//			}
//			break;
//		}
//	}
//}

///* This procedure is called to indicate that one or more child processes have been placed in background and are no longer cared about. */
//void Tcl_DetachPids(int numPids, PIDTYPE *pids)
//{
//	for (int i = 0; i < numPids; i++) {
//		PIDTYPE pid = pids[i];
//		register WaitInfo_t *info;
//		int count;
//		for (info = _waitInfo.table, count = _waitInfo.used; count > 0; info++, count--) {
//			if (pid != info->pid)
//				continue;
//			// If the process has already exited then destroy its table entry now.
//			if ((info->flags & WI_READY) && (WIFEXITED(info->status) || WIFSIGNALED(info->status))) {
//				*info = _waitInfo.table[_waitInfo.used - 1];
//				_waitInfo.used--;
//			}
//			else info->flags |= WI_DETACHED;
//			goto nextPid;
//		}
//		panic("Tcl_Detach couldn't find process");
//nextPid:
//		continue;
//	}
//}

#pragma endregion
