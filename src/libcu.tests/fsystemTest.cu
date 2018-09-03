#include <stdiocu.h>
#include <stringcu.h>
#include "../libcu/fsystem.h"
#include <assert.h>

static __global__ void g_fsystem_test1() {
	printf("fsystem_test1\n");
	char newPath[MAX_PATH];
	// ABSOLUTE
	strcpy(__cwd, ":\\test");
	expandPath(":\\", newPath); int a0a = !strcmp(newPath, ":");
	expandPath(":/one", newPath); int a1a = !strcmp(newPath, ":\\one");
	expandPath(":\\one", newPath); int a2a = !strcmp(newPath, ":\\one");
	expandPath(":\\one\\", newPath); int a3a = !strcmp(newPath, ":\\one");
	assert(a0a && a1a && a2a && a3a);

	expandPath(":\\.", newPath); int b0a = !strcmp(newPath, ":");
	expandPath(":\\one\\.", newPath); int b1a = !strcmp(newPath, ":\\one");
	expandPath(":\\one\\.\\", newPath); int b2a = !strcmp(newPath, ":\\one");
	expandPath(":\\one\\.\\two", newPath); int b3a = !strcmp(newPath, ":\\one\\two");
	assert(b0a && b1a && b2a && b3a);

	expandPath(":\\one\\..\\two", newPath); int c0a = !strcmp(newPath, ":\\two");
	expandPath(":\\one\\..\\two\\three", newPath); int c1a = !strcmp(newPath, ":\\two\\three");
	assert(c0a && c1a);

	// ROOT
	strcpy(__cwd, ":\\test");
	expandPath("\\.", newPath); int d0a = !strcmp(newPath, ":");
	expandPath("\\one", newPath); int d1a = !strcmp(newPath, ":\\one");
	assert(d0a && d1a);

	// RELATIVE
	strcpy(__cwd, ":\\test");
	expandPath(".", newPath); int e0a = !strcmp(newPath, ":\\test"); //printf("%s\n", newPath);
	expandPath("one", newPath); int e1a = !strcmp(newPath, ":\\test\\one"); //printf("%s\n", newPath);
	//assert(e0a && e1a);

	// CHDIR
	strcpy(__cwd, ":\\test");
	int f0a = fsystemChdir(":\\"); int f0b = !strcmp(__cwd, ":\\");
	//assert(f0a);

	// OPENDIR
	dirEnt_t *g0a = fsystemOpendir(":\\"); int g0b = !strcmp(__cwd, ":\\");
	//assert(g0a);

	// RENAME
	int h0a = fsystemRename(":\\", ":\\"); int h0b = !strcmp(__cwd, ":\\");
	//assert(h0a);

	// UNLINK
	int i0a = fsystemUnlink(":\\", false); int i0b = !strcmp(__cwd, ":\\");
	//assert(i0a);

	// MKDIR
	int r;
	dirEnt_t *j0a = fsystemMkdir(":\\", 0, &r); int j0b = !strcmp(__cwd, ":\\");
	//assert(j0a);

	// OPEN
	int fd;
	dirEnt_t *k0a = fsystemOpen(":\\", 0, &fd); int k0b = !strcmp(__cwd, ":\\");
	//assert(k0a);

	// RESET
	fsystemReset();
}
cudaError_t fsystem_test1() { g_fsystem_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
