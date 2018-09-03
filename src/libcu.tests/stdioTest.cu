#include <stdiocu.h>
#include <stringcu.h>
#include <sys/statcu.h>
#include <unistdcu.h>
#include <assert.h>

#ifndef HostDir
#define HostDir "C:\\T_\\"
#endif
#ifndef DeviceDir
#define DeviceDir ":\\"
#endif

#ifndef MAKEAFILE
#define MAKEAFILE
static __device__ void makeAFile(char *file) {
	FILE *fp = fopen(file, "w");
	fprintf_(fp, "test");
	fclose(fp);
}
#endif

extern __constant__ cuFILE __iob_streams[LIBCU_MAXFILESTREAM + 3];
static __global__ void g_stdio_test1() {
	printf("stdio_test1\n");
	goto here;

	//// STDIN/STDOUT/STDERR ////
	//#define stdin  ((FILE*)&__iob_streams[0]) /* Standard input stream.  */
	//#define stdout ((FILE*)&__iob_streams[1]) /* Standard output stream.  */
	//#define stderr ((FILE*)&__iob_streams[2]) /* Standard error output stream.  */
	bool a0 = (stdin == (FILE*)&__iob_streams[0] && stdout == (FILE*)&__iob_streams[1] && stderr == (FILE*)&__iob_streams[2]); assert(a0);

	//// REMOVE FILE ////
	//extern __device__ int remove_(const char *filename); #sentinel-branch
	/* Host Absolute */
	int a0a = remove(HostDir"missing.txt"); assert(a0a < 0);
	makeAFile(HostDir"test.txt");
	int a1a = remove(HostDir"test.txt"); assert(!a1a);

	/* Device Absolute */
	int b0a = remove(DeviceDir"missing.txt"); assert(b0a < 0);
	makeAFile(DeviceDir"test.txt");
	int b1a = remove(DeviceDir"test.txt"); assert(!b1a);

	/* Host Relative */
	chdir(HostDir);
	int c0a = remove("missing.txt"); assert(c0a < 0);
	makeAFile("test.txt");
	int c1a = remove("test.txt"); assert(!c1a);

	/* Device Relative */
	chdir(DeviceDir);
	int d0a = remove("missing.txt"); assert(d0a < 0);
	makeAFile("test.txt");
	int d1a = remove("test.txt"); assert(!d1a);

	//// RENAME FILE ////
	//extern __device__ int rename_(const char *old, const char *new_); #sentinel-branch
	/* Host Absolute */
	int e0a = rename(HostDir"missing.txt", "missing2.txt"); assert(e0a < 0);
	makeAFile(HostDir"test.txt");
	int e1a = rename(HostDir"test.txt", "test2.txt"); int e1b = remove(HostDir"test2.txt"); assert(!e1a && !e1b);
	makeAFile(HostDir"test.txt");
	int e2a = rename(HostDir"test.txt", "missing\\test2.txt"); int e2b = remove(HostDir"test.txt"); assert(e2a < 0 && !e2b);
	makeAFile(HostDir"test.txt");
	mkdir(HostDir"_dir", 0);
	int e3a = rename(HostDir"test.txt", "_dir\\test2.txt"); int e3b = remove(HostDir"_dir\\test2.txt"); assert(!e3a && !e3b);
	rmdir(HostDir"_dir");

	/* Device Absolute */
	int f0a = rename(DeviceDir"missing.txt", "missing2.txt"); assert(f0a < 0);
	makeAFile(DeviceDir"test.txt");
	int f1a = rename(DeviceDir"test.txt", "test2.txt"); int f1b = remove(DeviceDir"test2.txt"); assert(!f1a && !f1b);
	makeAFile(DeviceDir"test.txt");
	int f2a = rename(DeviceDir"test.txt", "missing\\test2.txt"); int f2b = remove(DeviceDir"test.txt"); assert(f2a < 0 && !f2b);
	makeAFile(DeviceDir"test.txt");
	mkdir(DeviceDir"_dir", 0);
	int f3a = rename(DeviceDir"test.txt", "_dir\\test2.txt"); int f3b = remove(DeviceDir"_dir\\test2.txt"); assert(!f3a && !f3b);
	rmdir(DeviceDir"_dir");

	/* Host Relative */
	chdir(HostDir);
	int g0a = rename("missing.txt", "missing2.txt"); assert(g0a < 0);
	makeAFile("test.txt");
	int g1a = rename("test.txt", "test2.txt"); int g1b = remove("test2.txt"); assert(!g1a && !g1b);

	/* Device Relative */
	chdir(DeviceDir);
	int h0a = rename("missing.txt", "missing2.txt"); assert(h0a < 0);
	makeAFile("test.txt");
	int h1a = rename("test.txt", "test2.txt"); int h1b = remove("test2.txt"); assert(!h1a && !h1b);

	//// TMPFILE ////
	//extern __device__ FILE *tmpfile_(void);
	FILE *i0a = tmpfile();
	fclose(i0a);

here:
	//// FCLOSE, FFLUSH, FREOPEN, FOPEN, FPRINTF ////
	//extern __device__ int fclose_(FILE *stream, bool wait = true); #sentinel-branch
	//extern __device__ int fflush_(FILE *stream); #sentinel-branch
	//extern __device__ FILE *freopen_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream) #sentinel-branch
	//extern __device__ FILE *fopen_(const char *__restrict filename, const char *__restrict modes); #sentinel-branch
	//moved: extern __device__ int fprintf(FILE *__restrict stream, const char *__restrict format, ...); //extern __device__ int vfprintf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);
	char buf[100];
	/* Host Absolute */
	FILE *j0a = fopen(HostDir"missing.txt", "r"); assert(!j0a);
	makeAFile(HostDir"test.txt");
	FILE *j1a = fopen(HostDir"test.txt", "r"); int j1b = fread(buf, 1, 4, j1a); FILE *j1c = freopen(HostDir"test.txt", "r", j1a); int j1d = fread(buf, 1, 4, j1c); int j1e = fclose(j1c); assert(j1a && j1b == 4 && j1c && j1d == 4 && !j1e);
	FILE *j2a = fopen(HostDir"test.txt", "w"); int j2b = fprintf_(j2a, "test"); FILE *j2c = freopen(HostDir"test.txt", "w", j2a); int j2d = fprintf_(j2c, "test"); int j2e = fflush(j2c); int j2f = fclose(j2c); assert(j2a && j2b == 4 && j2c && j2d == 4 && !j2e && !j2f);
	FILE *j3a = fopen(HostDir"test.txt", "w"); int j3b = fprintf_(j3a, "%03000d", 1234); FILE *j3c = freopen(HostDir"test.txt", "w", j3a); int j3d = fprintf_(j3c, "%03000d", 1234); int j3e = fflush(j3c); int j3f = fclose(j3c); assert(j3a && j3b == 3000 && j3c && j3d == 3000 && !j3e && !j3f);

	/* Device Absolute */
	FILE *k0a = fopen(DeviceDir"missing.txt", "r"); assert(!k0a);
	makeAFile(DeviceDir"test.txt");
	FILE *k1a = fopen(DeviceDir"test.txt", "r"); int k1b = fread(buf, 1, 4, k1a); FILE *k1c = freopen(DeviceDir"test.txt", "r", k1a); int k1d = fread(buf, 1, 4, k1c); int k1e = fclose(k1c); assert(k1a && k1b == 4 && k1c && k1d == 4 && !k1e);
	FILE *k2a = fopen(DeviceDir"test.txt", "w"); int k2b = fprintf_(k2a, "test"); FILE *k2c = freopen(DeviceDir"test.txt", "w", k2a); int k2d = fprintf_(k2c, "test"); int k2e = fflush(k2c); int k2f = fclose(k2c); assert(k2a && k2b == 4 && k2c && k2d == 4 && !k2e && !k2f);
	FILE *k3a = fopen(DeviceDir"test.txt", "w"); int k3b = fprintf_(k3a, "%03000d", 1234); FILE *k3c = freopen(DeviceDir"test.txt", "w", k3a); int k3d = fprintf_(k3c, "%03000d", 1234); int k3e = fflush(k3c); int k3f = fclose(k3c); assert(k3a && k3b == 3000 && k3c && k3d == 3000 && !k3e && !k3f);

	/* Host Relative */
	chdir(HostDir);
	FILE *l0a = fopen("missing.txt", "r"); assert(!l0a);
	makeAFile("test.txt");
	FILE *l1a = fopen("test.txt", "r"); int l1b = fread(buf, 4, 1, l1a); FILE *l1c = freopen("test.txt", "r", l1a); int l1d = fread(buf, 4, 1, l1c); int l1e = fclose(l1c); assert(l1a);
	FILE *l2a = fopen("test.txt", "w"); int l2b = fprintf_(l2a, "test"); FILE *l2c = freopen("test.txt", "w", l2a); int l2d = fprintf_(l2c, "test"); int l2e = fflush(l2c); int l2f = fclose(l2c); assert(l2a);

	/* Device Relative */
	chdir(DeviceDir);
	FILE *m0a = fopen("missing.txt", "r"); assert(!m0a);
	makeAFile("test.txt");
	FILE *m1a = fopen("test.txt", "r"); int m1b = fread(buf, 4, 1, m1a); FILE *m1c = freopen("test.txt", "r", m1a); int m1d = fread(buf, 4, 1, m1c); int m1e = fclose(m1c); assert(m1a);
	FILE *m2a = fopen("test.txt", "w"); int m2b = fprintf_(m2a, "test"); FILE *m2c = freopen("test.txt", "w", m2a); int m2d = fprintf_(m2c, "test"); int m2e = fflush(m2c); int m2f = fclose(m2c); assert(m2a);

	//// SETVBUF, SETBUF ////
	//extern __device__ int setvbuf_(FILE *__restrict stream, char *__restrict buf, int modes, size_t n); #sentinel-branch
	//extern __device__ void setbuf_(FILE *__restrict stream, char *__restrict buf); #sentinel-branch
	FILE *n0a = fopen(HostDir"test.txt", "w"); int n0b = setvbuf(n0a, nullptr, 0, 10); int n0c = fclose(n0a); assert(n0a && n0b && n0c);
	FILE *n1a = fopen(HostDir"test.txt", "w"); setbuf(n1a, nullptr); int n1b = fclose(n0a); assert(n1a && n1b);
	FILE *n2a = fopen(DeviceDir"test.txt", "w"); int n2b = setvbuf(n2a, nullptr, 0, 10); int n2c = fclose(n2a); assert(n2a && n2b && n2c);
	FILE *n3a = fopen(DeviceDir"test.txt", "w"); setbuf(n3a, nullptr); int n3b = fclose(n3a); assert(n3a && n3b);

	//// SNPRINTF, PRINTF, SPRINTF ////
	//#define sprintf(s, format, ...) snprintf_(s, 0xffffffff, format, __VA_ARGS__)
	//moved: extern __device__ int snprintf(char *__restrict s, size_t maxlen, const char *__restrict format, ...); //extern __device__ int vsnprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, va_list va);
	////moved: extern __device__ int printf(const char *__restrict format, ...);
	////moved: extern __device__ int sprintf(char *__restrict s, const char *__restrict format, ...); //__forceinline__ __device__ int vsprintf_(char *__restrict s, const char *__restrict format, va_list va);
	int o0a = snprintf(buf, sizeof(buf), "%d", 1); bool o0b = !strcmp(buf, "1"); assert(o0a && o0b);
	//skipped: printf("%d", 1);
	int o1a = sprintf(buf, "%d", 1); bool o1b = !strcmp(buf, "1"); assert(o1a && o1b);

	//// FSCANF, SCANF, SSCANF ////
	//moved: extern __device__ int fscanf(FILE *__restrict stream, const char *__restrict format, ...); //extern __device__ int vfscanf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);
	//moved: extern __device__ int scanf(const char *__restrict format, ...); //__forceinline__ __device__ int vscanf_(const char *__restrict format, va_list va);
	//moved: extern __device__ int sscanf(const char *__restrict s, const char *__restrict format, ...); //extern __device__ int vsscanf_(const char *__restrict s, const char *__restrict format, va_list va);
	FILE *p0a = fopen(HostDir"test.txt", "r"); int p0b = fscanf(p0a, "%s", buf); int p0c = fclose(p0a); bool p0d = !strcmp(buf, "1"); assert(p0a && p0b && p0c && p0d);
	FILE *p1a = fopen(DeviceDir"test.txt", "r"); int p1b = fscanf(p1a, "%s", buf); int p1c = fclose(p1a); bool p1d = !strcmp(buf, "1"); assert(p1a && p1b && p1c && p1d);
	//skipped: scanf("%s", buf);
	int p2a = sscanf("test", "%s", buf); bool p2b = !strcmp(buf, "1"); assert(p2a && p2b);

	//// FGETC, GETCHAR, GETC, FPUTC, PUTCHAR, PUTC, UNGETC ////
	//extern __device__ int fgetc_(FILE *stream); #sentinel-branch
	//__forceinline__ __device__ int getchar_(void);
	////sky: #define getc(fp) __GETC(fp)
	//extern __device__ int fputc_(int c, FILE *stream, bool wait = true); #sentinel-branch
	//__forceinline__ __device__ int putchar_(int c);
	////sky: #define putc(ch, fp) __PUTC(ch, fp)
	//extern __device__ int ungetc_(int c, FILE *stream, bool wait = true); #sentinel-branch
	//skipped: getchar();
	//skipped: getc(fp);
	/* Host Absolute */
	FILE *q0a = fopen(HostDir"test.txt", "w"); int q0b = fputc('a', q0a); int q0c = fputc('b', q0a); int q0d = fputc('c', q0a); fclose(q0a); assert(q0b && q0c && q0d);
	FILE *q1a = fopen(HostDir"test.txt", "r"); int q1b = fgetc(q1a); int q1c = fgetc(q1a); int q1d = fgetc(q1a); fclose(q1a); assert(q1b == 'a' && q1c == 'b' && q1d == 'c');
	//?? ungetc

	/* Device Absolute */
	FILE *r0a = fopen(DeviceDir"test.txt", "w"); int r0b = fputc('a', r0a); int r0c = fputc('b', r0a); int r0d = fputc('c', r0a); fclose(r0a); assert(r0b && r0c && r0d);
	FILE *r1a = fopen(DeviceDir"test.txt", "r"); int r1b = fgetc(r1a); int r1c = fgetc(r1a); int r1d = fgetc(r1a); fclose(r1a); assert(r1b == 'a' && r1c == 'b' && r1d == 'c');
	//?? ungetc

	//// FGETS, FPUTS, PUTS ////
	//extern __device__ char *fgets_(char *__restrict s, int n, FILE *__restrict stream); #sentinel-branch
	//extern __device__ int fputs_(const char *__restrict s, FILE *__restrict stream, bool wait = true); #sentinel-branch
	//__forceinline__ __device__ int puts_(const char *s);
	//skipped: puts(s);
	/* Host Absolute */
	FILE *s0a = fopen(HostDir"test.txt", "w"); int s0b = fputs("abc", s0a); fclose(s0a); assert(s0b);
	FILE *s1a = fopen(HostDir"test.txt", "r"); char *s1b = fgets(buf, 3, s1a); fclose(q1a); assert(!strncmp(s1b, "abc", 3));

	/* Device Absolute */
	FILE *t0a = fopen(DeviceDir"test.txt", "w"); int t0b = fputs("abc", t0a); fclose(t0a); assert(t0b);
	FILE *t1a = fopen(DeviceDir"test.txt", "r"); char *t1b = fgets(buf, 3, t1a); fclose(t1a); assert(!strncmp(t1b, "abc", 3));

	//extern __device__ size_t fread_(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true); #sentinel-branch
	//extern __device__ size_t fwrite_(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true); #sentinel-branch
	/* Host Absolute */
	FILE *u1a = fopen(HostDir"test.txt", "w"); int u1b = fwrite("test", 4, 1, u1a); fclose(u1a); assert(u1b == 4);
	FILE *u2a = fopen(HostDir"test.txt", "r"); int u2b = fread(buf, 4, 1, u2a); fclose(u2a); assert(u2b == 4 && !strncmp(buf, "test", 4));

	/* Device Absolute */
	FILE *v1a = fopen(DeviceDir"test.txt", "w"); int v1b = fwrite("test", 4, 1, v1a); fclose(v1a); assert(v1b == 4);
	FILE *v2a = fopen(DeviceDir"test.txt", "r"); int v2b = fread(buf, 4, 1, v2a); fclose(v2a); assert(v2b == 4 && !strncmp(buf, "test", 4));

	//// FSEEK, FTELL, REWING, FSEEKO, FGETPOS, FSETPOS ///
	//extern __device__ int fseek_(FILE *stream, long int off, int whence); #sentinel-branch
	//extern __device__ long int ftell_(FILE *stream); #sentinel-branch
	//extern __device__ void rewind_(FILE *stream); #sentinel-branch
	//extern __device__ int fseeko_(FILE *stream, __off_t off, int whence);
	//extern __device__ __off_t ftello_(FILE *stream);
	//extern __device__ int fgetpos_(FILE *__restrict stream, fpos_t *__restrict pos); #sentinel-branch
	//extern __device__ int fsetpos_(FILE *stream, const fpos_t *pos); #sentinel-branch
	//skipped: fseeko(s, 0, 0);
	//skipped: ftello(s);
	//skipped: fseeko64(s, 0, 0);
	//skipped: ftello64(s);
	//skipped: fgetpos64_(s);
	//skipped: fsetpos64_(s);
	/* Host Absolute */
	makeAFile(HostDir"test.txt");
	FILE *w0a = fopen(HostDir"test.txt", "r"); long int w0b = ftell(w0a); int w0c = fseek(w0a, 2, 0); long int w0d = ftell(w0a); rewind(w0a); long int w0e = ftell(w0a); fclose(w0a); assert(w0b == 0 && w0c && w0d == 2 && w0e == 0);
	FILE *w1a = fopen(HostDir"test.txt", "r"); fpos_t w1b; int w1c = fgetpos(w1a, &w1b); fseek(w1a, 2, 0); fpos_t w1d; fgetpos(w1a, &w1d); int w1e = fsetpos(w1a, &w1d); fpos_t w1f; int w1g = fgetpos(w1a, &w1f); fclose(w1a); assert(w1b == 0 && w1c && w1d == 2 && w1e && w1g == 2 && w1g);
	// TODO: skipped

	/* Device Absolute */
	makeAFile(DeviceDir"test.txt");
	FILE *x0a = fopen(DeviceDir"test.txt", "r"); long int x0b = ftell(x0a); int x0c = fseek(x0a, 2, 0); long int x0d = ftell(x0a); rewind(x0a); long int x0e = ftell(x0a); fclose(x0a); assert(x0b == 0 && x0c && x0d == 2 && x0e == 0);
	FILE *x1a = fopen(DeviceDir"test.txt", "r"); fpos_t x1b; int x1c = fgetpos(x1a, &x1b); fseek(x1a, 2, 0); fpos_t x1d; fgetpos(x1a, &x1d); int x1e = fsetpos(x1a, &x1d); fpos_t x1f; int x1g = fgetpos(x1a, &x1f); fclose(x1a); assert(x1b == 0 && x1c && x1d == 2 && x1e && x1g == 2 && x1g);

#if defined(__USE_LARGEFILE)
#endif

	//// CLEARERR, FERROR, PERROR ////
	//extern __device__ void clearerr_(FILE *stream); #sentinel-branch
	//extern __device__ int ferror_(FILE *stream); #sentinel-branch
	//extern __device__ void perror_(const char *s);
	//skipped: perror(s);
	/* Host Absolute */
	FILE *y0a = fopen(HostDir"test.txt", "r"); int y0b = ferror(y0a); clearerr(y0a); int y0c = ferror(y0a); fclose(y0a); assert(y0b == 0 && y0c == 0);

	/* Device Absolute */
	FILE *z0a = fopen(DeviceDir"test.txt", "r"); int z0b = ferror(z0a); clearerr(z0a); int z0c = ferror(z0a); fclose(z0a); assert(z0b == 0 && z0c == 0);

	//// FEOF, FILENO ////
	//extern __device__ int feof_(FILE *stream); #sentinel-branch
	//extern __device__ int fileno_(FILE *stream); #sentinel-branch
	/* Host Absolute */
	FILE *A0a = fopen(HostDir"test.txt", "r"); int A0b = feof(A0a); fseek(A0a, 4, 0); int A0c = feof(A0a); fclose(A0a); assert(!A0b && A0c);
	FILE *A1a = fopen(HostDir"test.txt", "r"); int A1b = fileno(A0a); fclose(A1a); assert(A1b);

	/* Device Absolute */
	FILE *B0a = fopen(DeviceDir"test.txt", "r"); int B0b = feof(B0a); fseek(B0a, 4, 0); int B0c = feof(B0a); fclose(B0a); assert(!B0b && B0c);
	FILE *B1a = fopen(DeviceDir"test.txt", "r"); int B1b = fileno(B0a); fclose(B1a); assert(B1b);
}
cudaError_t stdio_test1() { g_stdio_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma region _64bit

static __global__ void g_stdio_64bit() {
	printf("stdio_64bit\n");
	/*
	unsigned long long val = -1;
	void *ptr = (void *)-1;
	printf("%p\n", ptr);

	sscanf("123456789", "%Lx", &val);
	printf("val = %Lx\n", val);
	*/
}
cudaError_t stdio_64bit() { g_stdio_64bit<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma endregion

#pragma region Ganging

static __global__ void g_stdio_ganging() {
	printf("stdio_ganging\n");
}
cudaError_t stdio_ganging() { g_stdio_ganging<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma endregion

#pragma region scanf

static __global__ void g_stdio_scanf() {
	printf("stdio_scanf\n");
	/*
	const char *buf = "hello world";
	char *ps = NULL, *pc = NULL;
	char s[6], c;

	/ Check that %[...]/%c work. /
	sscanf(buf, "%[a-z] %c", s, &c);
	/ Check that %m[...]/%mc work. /
	sscanf(buf, "%m[a-z] %mc", &ps, &pc);

	if (strcmp(ps, "hello") != 0 || *pc != 'w' || strcmp(s, "hello") != 0 || c != 'w')
	return 1;

	free(ps);
	free(pc);

	return 0;
	*/
}
cudaError_t stdio_scanf() { g_stdio_scanf<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#pragma endregion
