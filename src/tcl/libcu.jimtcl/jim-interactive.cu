#include <cuda_runtimecu.h>
#include <errnocu.h>
#include <stringcu.h>

#include "jimautoconf.h"
#include "jim.h"
#ifdef USE_LINENOISE
#include <unistdcu.h>
#include "linenoise.h"
#else
#define MAX_LINE_LEN 512
#endif

// Returns an allocated line, or NULL if EOF.
char *Jim_HistoryGetline(const char *prompt) {
#ifdef USE_LINENOISE
	return linenoise(prompt);
#else
	char *line = (char *)malloc(MAX_LINE_LEN);
	fputs(prompt, stdout);
	fflush(stdout);
	if (fgets(line, MAX_LINE_LEN, stdin) == NULL) {
		free(line);
		return NULL;
	}
	int len = (int)strlen(line);
	if (len && line[len - 1] == '\n')
		line[len - 1] = '\0';
	return line;
#endif
}

void Jim_HistoryLoad(const char *filename) {
#ifdef USE_LINENOISE
	linenoiseHistoryLoad(filename);
#endif
}

void Jim_HistoryAdd(const char *line) {
#ifdef USE_LINENOISE
	linenoiseHistoryAdd(line);
#endif
}

void Jim_HistorySave(const char *filename) {
#ifdef USE_LINENOISE
	linenoiseHistorySave(filename);
#endif
}

void Jim_HistoryShow() {
#ifdef USE_LINENOISE
	// built-in history command
	int len;
	char **history = linenoiseHistory(&len);
	for (int i = 0; i < len; i++)
		printf("%4d %s\n", i + 1, history[i]);
#endif
}

#pragma region Jim_InteractivePrompt
#if __CUDACC__

struct InteractiveData {
	Jim_Interp *interp;
	int retcode;
	char *history_file;
	Jim_Obj *scriptObjPtr;
	char OP;
	char prompt[20];
};
struct InteractiveData h_dataI;

__device__ struct InteractiveData d_dataI;
void D_DATAI() { cudaErrorCheck(cudaMemcpyToSymbol(d_dataI, &h_dataI, sizeof(h_dataI))); }
void H_DATAI() { cudaErrorCheck(cudaMemcpyFromSymbol(&h_dataI, d_dataI, sizeof(h_dataI))); }

__global__ void g_InteractivePromptBegin() {
	printf("Welcome to Jim version %d.%d\n", JIM_VERSION / 100, JIM_VERSION % 100);
	Jim_SetVariableStrWithStr(d_dataI.interp, JIM_INTERACTIVE, "1");
}
void Jim_InteractivePromptBegin(Jim_Interp *interp) {
	memset(&h_dataI, 0, sizeof(h_dataI));
	h_dataI.interp = interp;
	h_dataI.retcode = JIM_OK;
#ifdef USE_LINENOISE
	const char *home;
	home = getenv("HOME");
	if (home && isatty(STDIN_FILENO)) {
		int history_len = strlen(home) + sizeof("/.jim_history");
		char *history_file = d_dataI.history_file = malloc(history_len);
		snprintf(history_file, history_len, "%s/.jim_history", home);
		Jim_HistoryLoad(history_file);
	}
#endif
	D_DATAI(); g_InteractivePromptBegin << <1, 1 >> > (); cudaErrorCheck(cudaDeviceSynchronize()); H_DATAI();
}

__global__ void g_InteractivePromptBodyBegin() {
	Jim_Interp *interp = d_dataI.interp;
	char *prompt = d_dataI.prompt;
	int retcode = d_dataI.retcode;
	if (retcode != 0) {
		const char *retcodestr = Jim_ReturnCode(retcode);
		if (*retcodestr == '?')
			snprintf(prompt, sizeof(d_dataI.prompt) - 3, "[%d] ", retcode);
		else
			snprintf(prompt, sizeof(d_dataI.prompt) - 3, "[%s] ", retcodestr);
	}
	else
		prompt[0] = '\0';
	strcat(prompt, ". ");
	Jim_Obj *scriptObjPtr = d_dataI.scriptObjPtr = Jim_NewStringObj(interp, "", 0);
	Jim_IncrRefCount(scriptObjPtr);
}
void Jim_InteractivePromptBodyBegin() {
	D_DATAI(); g_InteractivePromptBodyBegin << <1, 1 >> > (); cudaErrorCheck(cudaDeviceSynchronize()); H_DATAI();
}

__global__ void g_InteractivePromptBodyMiddle(char *line) {
	Jim_Interp *interp = d_dataI.interp;
	Jim_Obj *scriptObjPtr = d_dataI.scriptObjPtr;
	if (line == NULL) {
		Jim_DecrRefCount(interp, scriptObjPtr);
		d_dataI.retcode = JIM_OK;
		d_dataI.OP = -1; //: goto out;
		return;
	}
	if (Jim_Length(scriptObjPtr) != 0)
		Jim_AppendString(interp, scriptObjPtr, "\n", 1);
	Jim_AppendString(interp, scriptObjPtr, line, -1);
	int len;
	const char *str = Jim_GetString(scriptObjPtr, &len);
	if (len == 0) {
		d_dataI.OP = 0; //: continue;
		return;
	}
	char state;
	if (Jim_ScriptIsComplete(str, len, &state)) {
		d_dataI.OP = 1; //: break;
		return;
	}
	char *prompt = d_dataI.prompt;
	snprintf(prompt, sizeof(d_dataI.prompt), "%c> ", state);
	d_dataI.OP = 0; //: continue;
}
int Jim_InteractivePromptBodyMiddle(char *line) {
	char *d_line;
	int lineLength = (int)strlen(line) + 1;
	cudaMalloc((void**)&d_line, lineLength);
	cudaMemcpy(d_line, line, lineLength, cudaMemcpyHostToDevice);
	D_DATAI(); g_InteractivePromptBodyMiddle << <1, 1 >> > (d_line); cudaErrorCheck(cudaDeviceSynchronize()); H_DATAI();
	cudaFree(d_line);
	return h_dataI.OP;
}

__global__ void g_InteractivePromptBodyEnd() {
	Jim_Interp *interp = d_dataI.interp;
	Jim_Obj *scriptObjPtr = d_dataI.scriptObjPtr;
	int retcode = d_dataI.retcode = Jim_EvalObj(interp, scriptObjPtr);
	Jim_DecrRefCount(interp, scriptObjPtr);
	if (retcode == JIM_EXIT) {
		d_dataI.OP = 1; //: break;
		return;
	}
	if (retcode == JIM_ERROR)
		Jim_MakeErrorMessage(interp);
	int reslen;
	const char *result = Jim_GetString(Jim_GetResult(interp), &reslen);
	if (reslen)
		printf("%s\n", result);
	d_dataI.OP = 0; //: continue;
}
int Jim_InteractivePromptBodyEnd() {
#ifdef USE_LINENOISE
	if (!strcmp(str, "h")) {
		// built-in history command
		Jim_HistoryShow();
		Jim_DecrRefCount(interp, scriptObjPtr);
		return 0; //: continue;
	}
	Jim_HistoryAdd(Jim_String(scriptObjPtr));
	if (history_file)
		Jim_HistorySave(history_file);
#endif
	D_DATAI(); g_InteractivePromptBodyEnd << <1, 1 >> > (); cudaErrorCheck(cudaDeviceSynchronize()); H_DATAI();
	return h_dataI.OP;
}

int Jim_InteractivePromptEnd() {
	free(h_dataI.history_file);
	return h_dataI.retcode;
}

int Jim_InteractivePrompt(Jim_Interp *interp) {
	Jim_InteractivePromptBegin(interp);
	while (1) {
		Jim_InteractivePromptBodyBegin();
		int op;
		while (1) {
			char *line = Jim_HistoryGetline(h_dataI.prompt);
			if (line == NULL)
				if (errno == EINTR)
					continue;
			op = Jim_InteractivePromptBodyMiddle(line);
			if (op == -1) goto out;
			free(line);
			if (op == 0) continue;
			else if (op == 1) break;
		}
		op = Jim_InteractivePromptBodyEnd();
		if (op == -1) goto out;
		else if (op == 0) continue;
		else if (op == 1) break;
	}
out:
	return Jim_InteractivePromptEnd();
}

#else

int Jim_InteractivePrompt(Jim_Interp *interp) {
	int retcode = JIM_OK;
	char *history_file = NULL;
#ifdef USE_LINENOISE
	const char *home;
	home = getenv("HOME");
	if (home && isatty(STDIN_FILENO)) {
		int history_len = strlen(home) + sizeof("/.jim_history");
		history_file = Jim_Alloc(history_len);
		snprintf(history_file, history_len, "%s/.jim_history", home);
		Jim_HistoryLoad(history_file);
	}
#endif
	printf("Welcome to Jim version %d.%d\n", JIM_VERSION / 100, JIM_VERSION % 100);
	Jim_SetVariableStrWithStr(interp, JIM_INTERACTIVE, "1");
	while (1) {
		char prompt[20];
		if (retcode != 0) {
			const char *retcodestr = Jim_ReturnCode(retcode);
			if (*retcodestr == '?')
				snprintf(prompt, sizeof(prompt) - 3, "[%d] ", retcode);
			else
				snprintf(prompt, sizeof(prompt) - 3, "[%s] ", retcodestr);
		}
		else
			prompt[0] = '\0';
		strcat(prompt, ". ");
		Jim_Obj *scriptObjPtr = Jim_NewStringObj(interp, "", 0);
		Jim_IncrRefCount(scriptObjPtr);
		while (1) {
			char *line = Jim_HistoryGetline(prompt);
			if (line == NULL) {
				if (_errno == EINTR)
					continue;
				Jim_DecrRefCount(interp, scriptObjPtr);
				retcode = JIM_OK;
				goto out;
			}
			if (Jim_Length(scriptObjPtr) != 0)
				Jim_AppendString(interp, scriptObjPtr, "\n", 1);
			Jim_AppendString(interp, scriptObjPtr, line, -1);
			free(line);
			int len;
			const char *str = Jim_GetString(scriptObjPtr, &len);
			if (len == 0)
				continue;
			char state;
			if (Jim_ScriptIsComplete(str, len, &state))
				break;
			snprintf(prompt, sizeof(prompt), "%c> ", state);
		}
#ifdef USE_LINENOISE
		if (!strcmp(str, "h")) {
			// built-in history command
			Jim_HistoryShow();
			Jim_DecrRefCount(interp, scriptObjPtr);
			continue;
		}
		Jim_HistoryAdd(Jim_String(scriptObjPtr));
		if (history_file)
			Jim_HistorySave(history_file);
#endif
		retcode = Jim_EvalObj(interp, scriptObjPtr);
		Jim_DecrRefCount(interp, scriptObjPtr);
		if (retcode == JIM_EXIT)
			break;
		if (retcode == JIM_ERROR)
			Jim_MakeErrorMessage(interp);
		int reslen;
		const char *result = Jim_GetString(Jim_GetResult(interp), &reslen);
		if (reslen)
			printf("%s\n", result);
	}
out:
	Jim_Free(history_file);
	return retcode;
}

#endif
#pragma endregion
