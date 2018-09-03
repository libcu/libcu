#include "fsystem.h"
#include <stdiocu.h>
#include <sentinel-stdiomsg.h>
#include <stdlibcu.h>
#include <ctypecu.h>
#include <assert.h>
#include <unistdcu.h>
#include <fcntlcu.h>
#include <errnocu.h>

#define LIBCU_MAXLENGTH 1000000000

#if __OS_UNIX
#define _base _IO_buf_base
#define _flag _flags
#define _file _fileno
#endif

__BEGIN_DECLS;

// STREAMS
#pragma region STREAMS

typedef struct __align__(8) {
	cuFILE *file;			// reference
	unsigned short id;		// ID of author
	unsigned short threadid;// thread ID of author
} streamRef;

__device__ streamRef __iob_streamRefs[LIBCU_MAXFILESTREAM]; // Start of circular buffer (set up by host)
volatile __device__ streamRef *__iob_freeStreamPtr = __iob_streamRefs; // Current atomically-incremented non-wrapped offset
volatile __device__ streamRef *__iob_retnStreamPtr = __iob_streamRefs; // Current atomically-incremented non-wrapped offset
__constant__ cuFILE __iob_streams[LIBCU_MAXFILESTREAM+3];

static __forceinline__ __device__ void writeStreamRef(streamRef *ref, cuFILE *s) {
	ref->file = s;
	ref->id = gridDim.x*blockIdx.y + blockIdx.x;
	ref->threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

static __device__ cuFILE *streamGet(int fd = 0) {
	// advance circular buffer
	size_t offset = atomicAdd((_uintptr_t *)&__iob_freeStreamPtr, sizeof(streamRef)) - (size_t)&__iob_streamRefs;
	offset %= (sizeof(streamRef)*LIBCU_MAXFILESTREAM);
	int offsetId = offset / sizeof(streamRef);
	streamRef *ref = (streamRef *)((char *)&__iob_streamRefs + offset);
	cuFILE *s = (cuFILE *)ref->file;
	if (!s) {
		s = &__iob_streams[offsetId+3];
		writeStreamRef(ref, s);
	}
	s->_file = fd;
	return s;
}

static __device__ void streamFree(cuFILE *s) {
	if (!s) return;
	//if (s->_file != -1)
	//	close(s->_file);
	// advance circular buffer
	size_t offset = atomicAdd((_uintptr_t *)&__iob_retnStreamPtr, sizeof(streamRef)) - (size_t)&__iob_streamRefs;
	offset %= (sizeof(streamRef)*LIBCU_MAXFILESTREAM);
	streamRef *ref = (streamRef *)((char *)&__iob_streamRefs + offset);
	writeStreamRef(ref, s);
}

#pragma endregion

/* Remove file FILENAME.  */
__device__ int remove_(const char *filename) {
	if (ISHOSTPATH(filename)) { stdio_remove msg(filename); return msg.RC; }
	int saved_errno = errno;
	int rv = fsystemUnlink(filename, true); //: rmdir(filename);
	if (rv < 0 && errno == ENOTDIR) {
		_set_errno(saved_errno); // Need to restore errno.
		rv = fsystemUnlink(filename, false); //: unlink(filename);
	}
	return rv;
}

/* Rename file OLD to NEW.  */
__device__ int rename_(const char *old, const char *new_) {
	if (ISHOSTPATH(old)) { stdio_rename msg(old, new_); return msg.RC; }
	return fsystemRename(old, new_);
}

/* Create a temporary file and open it read/write. */
#ifndef __USE_FILE_OFFSET64
#define TEMP_DIR ":\\_temp\\"
#define TEMP_DIRLENGTH sizeof(TEMP_DIR)-1
__device__ FILE *tmpfile_(void) {
	char newPath[50] = TEMP_DIR;
	dirEnt_t *ent = fsystemOpendir(newPath);
	int r; if (!ent) fsystemMkdir(newPath, 0666, &r);
	newPath[TEMP_DIRLENGTH+6] = 0;
	int i; for (i = 0; i < 10; i++) {
		for (int n = 0; n < 6; n++) {
			int x = rand() % (10+26);
			char c = x < 10 ? x+'0' : x < 10+26 ? x-10+'a' : 'Z';
			newPath[TEMP_DIRLENGTH+n] = c;
		}
		if (!fsystemAccess(newPath, 0, &r)) break;
	}
	if (i == 10) {
		_set_errno(EINVAL);
		return nullptr;
	}
	FILE *file = freopen(newPath, "wb+", nullptr);
	register cuFILE *s = (cuFILE *)file;
	fsystemSetFlag(s->_file, DELETE);
	return file;
}
#endif

/* Close STREAM. */
__device__ int fclose_(FILE *stream, bool wait) {
	if (ISHOSTFILE(stream)) { 
		stdio_fclose msg(wait, stream); return msg.RC; }
	register cuFILE *s = (cuFILE *)stream;
	dirEnt_t *f; UNUSED_SYMBOL(f);
	if (!s || !(f = (dirEnt_t *)s->_base))
		panic("fclose: !stream");
	if (s->_file != -1)
		close(s->_file);
	streamFree(s);
	return 0;
}

/* Flush STREAM, or all streams if STREAM is NULL. */
__device__ int fflush_(FILE *stream, bool wait) {
	if (ISHOSTFILE(stream)) { stdio_fflush msg(wait, stream); return msg.RC; }
	return 0;
}

/* Open a file, replacing an existing stream with it. */
__device__ FILE *freopen_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream) {
	if (ISHOSTPATH(filename)) { stdio_freopen msg(filename, modes, stream); return msg.RC; }
	register cuFILE *s = (cuFILE *)stream;
	if (s)
		streamFree(s);
	// Parse the specified mode.
	unsigned short openMode = O_RDONLY;
	if (*modes != 'r') { // Not read...
		openMode = (O_WRONLY|O_CREAT|O_TRUNC);
		if (*modes != 'w') { // Not write (create or truncate)...
			openMode = (O_WRONLY|O_CREAT|O_APPEND);
			if (*modes != 'a') {	// Not write (create or append)...
				_set_errno(EINVAL); // So illegal mode.
				streamFree(s);
				return nullptr;
			}
		}
	}
	if (modes[1] == 'b') // Binary mode (NO-OP currently).
		++modes;
	if (modes[1] == '+') { // Read and Write.
		++modes;
		openMode |= (O_RDONLY|O_WRONLY);
		openMode += (O_RDWR - (O_RDONLY|O_WRONLY));
	}

	// Need to allocate a FILE (not freopen).
	if (!s) {
		s = streamGet();
		if (!s)
			return nullptr;
	}
	s->_flag = openMode;
	s->_base = (char *)fsystemOpen(filename, openMode, &s->_file);
	if (!s->_base) {
		_set_errno(EINVAL);
		streamFree(s);
		return nullptr;
	}
	return (FILE *)s;
}

/* Open a file and create a new stream for it. */
__device__ FILE *fopen_(const char *__restrict filename, const char *__restrict modes) {
	if (ISHOSTPATH(filename)) { stdio_freopen msg(filename, modes, nullptr); return msg.RC; }
	return freopen_(filename, modes, nullptr); 
}

#ifdef __USE_LARGEFILE64
/* Open a file, replacing an existing stream with it. */
__device__ FILE *freopen64_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream) {
	if (ISHOSTPATH(filename)) { stdio_freopen msg(filename, modes, stream); return msg.RC; }
	register cuFILE *s = (cuFILE *)stream;
	if (s)
		streamFree(s);
	// Parse the specified mode.
	unsigned short openMode = O_RDONLY;
	if (*modes != 'r') { // Not read...
		openMode = (O_WRONLY|O_CREAT|O_TRUNC);
		if (*modes != 'w') { // Not write (create or truncate)...
			openMode = (O_WRONLY|O_CREAT|O_APPEND);
			if (*modes != 'a') {	// Not write (create or append)...
				_set_errno(EINVAL); // So illegal mode.
				streamFree(s);
				return nullptr;
			}
		}
	}
	if (modes[1] == 'b') // Binary mode (NO-OP currently).
		++modes;
	if (modes[1] == '+') { // Read and Write.
		++modes;
		openMode |= (O_RDONLY|O_WRONLY);
		openMode += (O_RDWR - (O_RDONLY|O_WRONLY));
	}

	// Need to allocate a FILE (not freopen).
	if (!s) {
		s = streamGet();
		if (!s)
			return nullptr;
	}
	s->_flag = openMode;
	s->_base = (char *)fsystemOpen(filename, openMode, &s->_file);
	return (FILE *)s;
}

/* Open a file and create a new stream for it. */
__device__ FILE *fopen64_(const char *__restrict filename, const char *__restrict modes)
{
	if (ISHOSTPATH(filename)) { stdio_freopen msg(filename, modes, nullptr); return msg.RC; }
	return freopen64_(filename, modes, nullptr); 
}
#endif

/* Make STREAM use buffering mode MODE. If BUF is not NULL, use N bytes of it for buffering; else allocate an internal buffer N bytes long.  */
__device__ int setvbuf_(FILE *__restrict stream, char *__restrict buf, int modes, size_t n) {
	if (ISHOSTFILE(stream)) { stdio_setvbuf msg(stream, buf, modes, n); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* If BUF is NULL, make STREAM unbuffered. Else make it use buffer BUF, of size BUFSIZ.  */
__device__ void setbuf_(FILE *__restrict stream, char *__restrict buf) {
	if (ISHOSTFILE(stream)) { stdio_setvbuf msg(stream, buf, -1, 0); return; }
	setvbuf_(stream, buf, buf ? _IOFBF : _IONBF, BUFSIZ);
}

/* Write formatted output to S from argument list ARG.  */
#ifdef __CUDA_ARCH__
__device__ int snprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, ...) { va_list va; va_start(va, format); int r = vsnprintf_(s, maxlen, format, va); va_end(va); return r; }
__device__ int vsnprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, va_list va) {
	if (maxlen <= 0) return -1;
	strbld_t b;
	strbldInit(&b, nullptr, (char *)s, (int)maxlen, 0);
	strbldAppendFormatv(&b, format, va);
	strbldToString(&b);
	return b.index;
}
#endif

/* Write formatted output to S.  */
// moved: extern __device__ int sprintf_(char *__restrict s, const char *__restrict format, ...);

/* Write formatted output to S from argument list ARG. */
#ifdef __CUDA_ARCH__
__device__ int fprintf_(FILE *__restrict s, const char *__restrict format, ...) { va_list va; va_start(va, format); int r = vfprintf_(s, format, va, true); va_end(va); return r; }
__device__ int vfprintf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait) {
	char base[PRINT_BUF_SIZE];
	strbld_t b;
	strbldInit(&b, nullptr, base, sizeof(base), LIBCU_MAXLENGTH);
	strbldAppendFormatv(&b, format, va);
	const char *v = strbldToString(&b);
	int size = b.index + 1;
	// chunk results
	int rc = 1, offset = 0;
	if (!ISHOSTFILE(s)) while (size > 0 && rc > 0) { rc = fwrite_(v + offset, 1, size > 1024 ? 1024 : size, s); size -= 1024; offset += rc; }
	else while (size > 0 && rc > 0) { stdio_fwrite msg(true, v + offset, 1, size > 1024 ? 1024 : size, s); rc = msg.RC; size -= 1024; offset += rc; }
	free((void *)v);
	return offset - 1; // remove null termination, returns number of characters written
}
#endif

/* Read formatted input from STREAM.  */
// moved: extern __device__ int fscanf_(FILE *__restrict stream, const char *__restrict format, ...);
/* Read formatted input from stdin.  */
// moved: extern __device__ int scanf_(const char *__restrict format, ...);
/* Read formatted input from S.  */
// moved: extern __device__ int sscanf_(const char *__restrict s, const char *__restrict format, ...);

/* Read a character from STREAM.  */
__device__ int fgetc_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_fgetc msg(stream); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Write a character to STREAM.  */
__device__ int fputc_(int c, FILE *stream, bool wait) {
	if (ISHOSTFILE(stream)) { stdio_fputc msg(wait, c, stream); return msg.RC; }
	if (stream == stdout || stream == stderr)
		printf("%c", c);
	return 0;
}

/* Get a newline-terminated string of finite length from STREAM.  */
__device__ char *fgets_(char *__restrict s, int n, FILE *__restrict stream) {
	if (ISHOSTFILE(stream)) { stdio_fgets msg(s, n, stream); return msg.RC; }
	panic("Not Implemented");
	return nullptr;
}

/* Write a string to STREAM.  */
__device__ int fputs_(const char *__restrict s, FILE *__restrict stream, bool wait) {
	if (ISHOSTFILE(stream)) { stdio_fputs msg(wait, s, stream); return msg.RC; }
	if (stream == stdout || stream == stderr)
		printf(s);
	return 0;
}

/* Push a character back onto the input buffer of STREAM.  */
__device__ int ungetc_(int c, FILE *stream, bool wait) {
	if (ISHOSTFILE(stream)) { stdio_ungetc msg(wait, c, stream); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Read chunks of generic data from STREAM.  */
__device__ size_t fread_(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait) {
	if (ISHOSTFILE(stream)) { stdio_fread msg(wait, ptr, size, n, stream); return msg.RC; }
	register cuFILE *s = (cuFILE *)stream;
	dirEnt_t *f;
	if (!s || !(f = (dirEnt_t *)s->_base))
		panic("fwrite: !stream");
	if (f->dir.d_type != 2)
		panic("fwrite: stream !file");
	size *= n;
	memfileRead(f->u.file, ptr, size, 0);
	return n;
}

/* Write chunks of generic data to STREAM.  */
__device__ size_t fwrite_(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait) {
	if (ISHOSTFILE(stream)) { stdio_fwrite msg(wait, ptr, size, n, stream); return msg.RC; }
	register cuFILE *s = (cuFILE *)stream;
	dirEnt_t *f;
	if (!s || !(f = (dirEnt_t *)s->_base))
		panic("fwrite: !stream");
	if (f->dir.d_type != 2)
		panic("fwrite: stream !file");
	size *= n;
	memfileWrite(f->u.file, ptr, size, 0);
	return n;
}

/* Seek to a certain position on STREAM.  */
__device__ int fseek_(FILE *stream, long int off, int whence) {
	if (ISHOSTFILE(stream)) { stdio_fseek msg(true, stream, off, whence); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Return the current position of STREAM.  */
__device__ long int ftell_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_ftell msg(stream); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Rewind to the beginning of STREAM.  */
__device__ void rewind_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_rewind msg(stream); return; }
	panic("Not Implemented");
	return;
}

#if defined(__USE_LARGEFILE)
#ifndef __USE_FILE_OFFSET64
/* Seek to a certain position on STREAM.   */
__device__ int fseeko_(FILE *stream, __off_t off, int whence) {
	if (ISHOSTFILE(stream)) { stdio_fseeko msg(true, stream, off, 0, whence, false); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Return the current position of STREAM.  */
__device__ __off_t ftello_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_ftello msg(stream, false); return msg.RC; }
	panic("Not Implemented");
	return 0;
}
#endif
#endif

#ifndef __USE_FILE_OFFSET64
/* Get STREAM's position.  */
__device__ int fgetpos_(FILE *__restrict stream, fpos_t *__restrict pos) {
	if (ISHOSTFILE(stream)) { stdio_fgetpos msg(stream, pos, nullptr, false); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Set STREAM's position.  */
__device__ int fsetpos_(FILE *stream, const fpos_t *pos) {
	if (ISHOSTFILE(stream)) { stdio_fsetpos msg(stream, pos, nullptr, false); return msg.RC; }
	panic("Not Implemented");
	return 0;
}
#endif

#ifdef __USE_LARGEFILE64
/* Seek to a certain position on STREAM.   */
__device__ int fseeko64_(FILE *stream, __off64_t off, int whence) {
	if (ISHOSTFILE(stream)) { stdio_fseeko msg(true, stream, 0, off, whence, true); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Return the current position of STREAM.  */
__device__ __off64_t ftello64_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_ftello msg(stream, true); return msg.RC64; }
	panic("Not Implemented");
	return 0;
}

/* Get STREAM's position.  */
__device__ int fgetpos64_(FILE *__restrict stream, fpos64_t *__restrict pos) {
	if (ISHOSTFILE(stream)) { stdio_fgetpos msg(stream, nullptr, pos, true); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Set STREAM's position.  */
__device__ int fsetpos64_(FILE *stream, const fpos64_t *pos) {
	if (ISHOSTFILE(stream)) { stdio_fsetpos msg(stream, nullptr, pos, true); return msg.RC; }
	panic("Not Implemented");
	return 0;
}
#endif

/* Clear the error and EOF indicators for STREAM.  */
__device__ void clearerr_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_clearerr msg(stream); return; }
	panic("Not Implemented");
}

/* Return the EOF indicator for STREAM.  */
__device__ int feof_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_feof msg(stream); return msg.RC; }
	panic("Not Implemented");
	return 0;
}

/* Return the error indicator for STREAM.  */
__device__ int ferror_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_ferror msg(stream); return msg.RC; }
	if (stream == stdout || stream == stderr)
		return 0; 
	return 0;
}

/* Print a message describing the meaning of the value of errno.  */
__device__ void perror_(const char *s) {
	printf(s);
}

/* Return the system file descriptor for STREAM.  */
__device__ int fileno_(FILE *stream) {
	if (ISHOSTFILE(stream)) { stdio_fileno msg(stream); return msg.RC; }
	register cuFILE *s = (cuFILE *)stream;
	return stream == stdin ? 0 : stream == stdout ? 1 : stream == stderr ? 2 : s->_file;
}

// sscanf
#pragma region sscanf

#define	BUF		32 	// Maximum length of numeric string.

// Flags used during conversion.
#define	LONG		0x01	// l: long or double
#define	SHORT		0x04	// h: short
#define	SUPPRESS	0x08	// *: suppress assignment
#define	POINTER		0x10	// p: void * (as hex)
#define	NOSKIP		0x20	// [ or c: do not skip blanks
#define	LONGLONG	0x400	// ll: long long (+ deprecated q: quad)
#define	SHORTSHORT	0x4000	// hh: char
#define	UNSIGNED	0x8000	// %[oupxX] conversions

// The following are used in numeric conversions only:
// SIGNOK, NDIGITS, DPTOK, and EXPOK are for floating point;
// SIGNOK, NDIGITS, PFXOK, and NZDIGITS are for integral.
#define	SIGNOK		0x40	// +/- is (still) legal
#define	NDIGITS		0x80	// no digits detected
#define	DPTOK		0x100	// (float) decimal point is still legal
#define	EXPOK		0x200	// (float) exponent (e+3, etc) still legal
#define	PFXOK		0x100	// 0x prefix is (still) legal
#define	NZDIGITS	0x200	// no zero digits detected

// Conversion types.
#define	CT_CHAR		0	// %c conversion
#define	CT_CCL		1	// %[...] conversion
#define	CT_STRING	2	// %s conversion
#define	CT_INT		3	// %[dioupxX] conversion

static __device__ const char *__sccl(char *tab, const char *fmt) {
	// first 'clear' the whole table
	int c, n, v;
	c = *fmt++; // first char hat => negated scanset
	if (c == '^') {
		v = 1; // default => accept
		c = *fmt++; // get new first char
	} else
		v = 0; // default => reject 
	memset(tab, v, 256); // XXX: Will not work if sizeof(tab*) > sizeof(char)
	if (c == 0)
		return (fmt - 1); // format ended before closing ]
	// Now set the entries corresponding to the actual scanset to the opposite of the above.
	// The first character may be ']' (or '-') without being special; the last character may be '-'.
	v = 1 - v;
	for (;;) {
		tab[c] = v; // take character c
doswitch:
		n = *fmt++; // and examine the next
		switch (n) {
		case 0: // format ended too soon
			return (fmt - 1);
		case '-':
			// A scanset of the form [01+-]
			// is defined as `the digit 0, the digit 1, the character +, the character -', but
			// the effect of a scanset such as [a-zA-Z0-9]
			// is implementation defined.  The V7 Unix scanf treats `a-z' as `the letters a through
			// z', but treats `a-a' as `the letter a, the character -, and the letter a'.
			//
			// For compatibility, the `-' is not considerd to define a range if the character following
			// it is either a close bracket (required by ANSI) or is not numerically greater than the character
			// we just stored in the table (c).
			n = *fmt;
			if (n == ']' || n < c) {
				c = '-';
				break; // resume the for(;;)
			}
			fmt++;
			// fill in the range
			do {
				tab[++c] = v;
			} while (c < n);
			c = n;
			// Alas, the V7 Unix scanf also treats formats such as [a-c-e] as `the letters a through e'. This too is permitted by the standard....
			goto doswitch;
			//break;
		case ']': // end of scanset
			return fmt;
		default:
			// just another character
			c = n;
			break;
		}
	}
}

/* Read formatted input from S into argument list ARG.  */
#if __CUDA_ARCH__
__device__ int fscanf_(FILE *__restrict s, const char *__restrict format, ...) { va_list va; va_start(va, format); int r = vfscanf_(s, format, va, true); va_end(va); return r; }
#endif
__device__ int vfscanf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait) {
	panic("Not Implemented");
	return 0;
}

/* Read formatted input from stdin into argument list ARG. */
#if __CUDA_ARCH__
__device__ int scanf_(const char *__restrict format, ...) { va_list va; va_start(va, format); int r = vfscanf_(stdin, format, va, true); va_end(va); return r; }
#endif
__device__ int vscanf_(const char *__restrict format, va_list va) { return vfscanf_(stdin, format, va, true); }

/* Read formatted input from S.  */
static __constant__ const short _basefix[17] = { 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; // 'basefix' is used to avoid 'if' tests in the integer scanner
#if __CUDA_ARCH__
__device__ int sscanf_(const char *__restrict s, const char *__restrict format, ...) { va_list va; va_start(va, format); int r = vsscanf_(s, format, va); va_end(va); return r; }
#endif
__device__ int vsscanf_(const char *__restrict str, const char *__restrict fmt, va_list va) {
	int c; // character from format, or conversion
	size_t width; // field width, or 0
	char *p; // points into all kinds of strings
	int n; // handy integer
	int flags; // flags as defined above
	char *p0; // saves original value of p when necessary
	char ccltab[256]; // character class table for %[...]
	char buf[BUF]; // buffer for numeric conversions

	int nassigned = 0; // number of fields assigned
	int nconversions = 0; // number of conversions
	int nread = 0; // number of characters consumed from fp
	int base = 0; // base argument to conversion function

	int inr = strlen(str);
	for (;;) {
		c = *fmt++;
		if (c == 0)
			return nassigned;
		if (isspace(c)) {
			while (inr > 0 && isspace(*str)) nread++, inr--, str++;
			continue;
		}
		if (c != '%')
			goto literal_;
		width = 0;
		flags = 0;
		// switch on the format.  continue if done; break once format type is derived.
again:	c = *fmt++;
		switch (c) {
		case '%':
literal_:
			if (inr <= 0)
				goto input_failure;
			if (*str != c)
				goto match_failure;
			inr--, str++;
			nread++;
			continue;
		case '*':
			flags |= SUPPRESS;
			goto again;
		case 'l':
			if (flags & LONG) {
				flags &= ~LONG;
				flags |= LONGLONG;
			} else
				flags |= LONG;
			goto again;
		case 'q':
			flags |= LONGLONG; // not quite
			goto again;
		case 'h':
			if (flags & SHORT) {
				flags &= ~SHORT;
				flags |= SHORTSHORT;
			} else
				flags |= SHORT;
			goto again;

		case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
			width = width * 10 + c - '0';
			goto again;

			// Conversions.
		case 'd':
			c = CT_INT;
			base = 10;
			break;
		case 'i':
			c = CT_INT;
			base = 0;
			break;
		case 'o':
			c = CT_INT;
			flags |= UNSIGNED;
			base = 8;
			break;
		case 'u':
			c = CT_INT;
			flags |= UNSIGNED;
			base = 10;
			break;
		case 'X':
		case 'x':
			flags |= PFXOK;	// enable 0x prefixing
			c = CT_INT;
			flags |= UNSIGNED;
			base = 16;
			break;
		case 's':
			c = CT_STRING;
			break;
		case '[':
			fmt = __sccl(ccltab, fmt);
			flags |= NOSKIP;
			c = CT_CCL;
			break;
		case 'c':
			flags |= NOSKIP;
			c = CT_CHAR;
			break;
		case 'p': // pointer format is like hex
			flags |= POINTER|PFXOK;
			c = CT_INT;
			flags |= UNSIGNED;
			base = 16;
			break;
		case 'n':
			nconversions++;
			if (flags & SUPPRESS) continue; // ??? 
			if (flags & SHORTSHORT) *va_arg(va, char *) = nread;
			else if (flags & SHORT) *va_arg(va, short *) = nread;
			else if (flags & LONG) *va_arg(va, long *) = nread;
			else if (flags & LONGLONG) *va_arg(va, long long *) = nread;
			else *va_arg(va, int *) = nread;
			continue;
		}

		// We have a conversion that requires input.
		if (inr <= 0)
			goto input_failure;

		// Consume leading white space, except for formats that suppress this.
		if ((flags & NOSKIP) == 0) {
			while (isspace(*str)) {
				nread++;
				if (--inr > 0) str++;
				else goto input_failure;
			}
			// Note that there is at least one character in the buffer, so conversions that do not set NOSKIP
			// can no longer result in an input failure.
		}

		// Do the conversion.
		switch (c) {
		case CT_CHAR: // scan arbitrary characters (sets NOSKIP)
			if (width == 0)
				width = 1;
			if (flags & SUPPRESS) {
				size_t sum = 0;
				for (;;) {
					if ((n = inr) < (int)width) {
						sum += n;
						width -= n;
						str += n;
						if (sum == 0)
							goto input_failure;
						break;
					}
					else {
						sum += width;
						inr -= width;
						str += width;
						break;
					}
				}
				nread += sum;
			}
			else {
				memcpy(va_arg(va, char *), str, width);
				inr -= width;
				str += width;
				nread += width;
				nassigned++;
			}
			nconversions++;
			break;
		case CT_CCL: // scan a (nonempty) character class (sets NOSKIP)
			if (width == 0)
				width = (size_t)~0;	// 'infinity'
			// take only those things in the class
			if (flags & SUPPRESS) {
				n = 0;
				while (ccltab[(unsigned char)*str]) {
					n++, inr--, str++;
					if (--width == 0) break;
					if (inr <= 0) {
						if (n == 0)
							goto input_failure;
						break;
					}
				}
				if (n == 0)
					goto match_failure;
			}
			else {
				p0 = p = va_arg(va, char *);
				while (ccltab[(unsigned char)*str]) {
					inr--;
					*p++ = *str++;
					if (--width == 0) break;
					if (inr <= 0) {
						if (p == p0)
							goto input_failure;
						break;
					}
				}
				n = p - p0;
				if (n == 0)
					goto match_failure;
				*p = 0;
				nassigned++;
			}
			nread += n;
			nconversions++;
			break;
		case CT_STRING: // like CCL, but zero-length string OK, & no NOSKIP
			if (width == 0)
				width = (size_t)~0;
			if (flags & SUPPRESS) {
				n = 0;
				while (!isspace(*str)) {
					n++, inr--, str++;
					if (--width == 0) break;
					if (inr <= 0) break;
				}
				nread += n;
			}
			else {
				p0 = p = va_arg(va, char *);
				while (!isspace(*str)) {
					inr--;
					*p++ = *str++;
					if (--width == 0) break;
					if (inr <= 0) break;
				}
				*p = 0;
				nread += p - p0;
				nassigned++;
			}
			nconversions++;
			continue;
		case CT_INT: // scan an integer as if by the conversion function
#ifdef hardway
			if (width == 0 || width > sizeof(buf) - 1)
				width = sizeof(buf) - 1;
#else
			// size_t is unsigned, hence this optimisation
			if (--width > sizeof(buf) - 2)
				width = sizeof(buf) - 2;
			width++;
#endif
			flags |= SIGNOK|NDIGITS|NZDIGITS;
			for (p = buf; width; width--) {
				c = *str;
				// Switch on the character; `goto ok' if we accept it as a part of number.
				switch (c) {
				case '0':
					// The digit 0 is always legal, but is special.  For %i conversions, if no digits (zero or nonzero) have been
					// scanned (only signs), we will have base==0.  In that case, we should set it to 8 and enable 0x prefixing.
					// Also, if we have not scanned zero digits before this, do not turn off prefixing (someone else will turn it off if we
					// have scanned any nonzero digits).
					if (base == 0) {
						base = 8;
						flags |= PFXOK;
					}
					if (flags & NZDIGITS) flags &= ~(SIGNOK|NZDIGITS|NDIGITS);
					else flags &= ~(SIGNOK|PFXOK|NDIGITS);
					goto ok;
				case '1': case '2': case '3': // 1 through 7 always legal
				case '4': case '5': case '6': case '7':
					base = _basefix[base];
					flags &= ~(SIGNOK|PFXOK|NDIGITS);
					goto ok;
				case '8': case '9': // digits 8 and 9 ok iff decimal or hex
					base = _basefix[base];
					if (base <= 8) break; // not legal here
					flags &= ~(SIGNOK|PFXOK|NDIGITS);
					goto ok;
				case 'A': case 'B': case 'C': // letters ok iff hex
				case 'D': case 'E': case 'F':
				case 'a': case 'b': case 'c':
				case 'd': case 'e': case 'f':
					// no need to fix base here
					if (base <= 10) break; // not legal here
					flags &= ~(SIGNOK|PFXOK|NDIGITS);
					goto ok;
				case '+': case '-': // sign ok only as first character
					if (flags & SIGNOK) {
						flags &= ~SIGNOK;
						goto ok;
					}
					break;
				case 'x': case 'X': // x ok iff flag still set & 2nd char
					if (flags & PFXOK && p == buf + 1) {
						base = 16; // if %i
						flags &= ~PFXOK;
						goto ok;
					}
					break;
				}
				// If we got here, c is not a legal character for a number.  Stop accumulating digits.
				break;
ok:
				// c is legal: store it and look at the next.
				*p++ = c;
				if (--inr > 0)
					str++;
				else 
					break; // end of input
			}
			// If we had only a sign, it is no good; push back the sign.  If the number ends in `x',
			// it was [sign] '0' 'x', so push back the x and treat it as [sign] '0'.
			if (flags & NDIGITS) {
				if (p > buf) {
					str--;
					inr++;
				}
				goto match_failure;
			}
			c = ((char *)p)[-1];
			if (c == 'x' || c == 'X') {
				--p;
				str--;
				inr++;
			}
			if (!(flags & SUPPRESS)) {
				quad_t res;
				*p = 0;
				if ((flags & UNSIGNED) == 0) res = strtoq(buf, (char **)NULL, base);
				else res = strtouq(buf, (char **)NULL, base);
				if (flags & POINTER) *va_arg(va, void **) = (void *)(intptr_t)res;
				else if (flags & SHORTSHORT) *va_arg(va, char *) = res;
				else if (flags & SHORT) *va_arg(va, short *) = res;
				else if (flags & LONG) *va_arg(va, long *) = res;
				else if (flags & LONGLONG) *va_arg(va, long long *) = res;
				else *va_arg(va, int *) = res;
				nassigned++;
			}
			nread += p - buf;
			nconversions++;
			break;
		}
	}
input_failure:
	return nconversions != 0 ? nassigned : -1;
match_failure:
	return nassigned;
}

#pragma endregion

__END_DECLS;
