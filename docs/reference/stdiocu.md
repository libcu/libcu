## #include <stdiocu.h>

Also includes:
```
#include <stdio.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```#define stdin``` | Standard input stream
```#define stdout``` | Standard output stream.
```#define stderr``` | Standard error output stream.
```__device__ int remove(const char *filename);``` | Remove file FILENAME. | #sentinel-isdevicepath
```__device__ int rename(const char *old, const char *new_);``` | Rename file OLD to NEW. | #sentinel-isdevicepath
```__device__ FILE *tmpfile(void);``` | Create a temporary file and open it read/write.
```__device__ int fclose(FILE *stream, bool wait = true);``` | Close STREAM. | #sentinel-isdevicefile
```__device__ int fflush(FILE *stream);``` | Flush STREAM, or all streams if STREAM is NULL. | #sentinel-isdevicefile
```__device__ FILE *freopen(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);``` | Open a file, replacing an existing stream with it. | #sentinel-isdevicepath
```__device__ FILE *fopen(const char *__restrict filename, const char *__restrict modes);``` | Open a file and create a new stream for it. | #sentinel-isdevicepath
```__device__ FILE *freopen64(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream)``` | Open a file, replacing an existing stream with it. | #sentinel-isdevicepath #file64
```__device__ FILE *fopen64(const char *__restrict filename, const char *__restrict modes)``` | Open a file and create a new stream for it. | #sentinel-isdevicepath #file64
```__device__ int setvbuf(FILE *__restrict stream, char *__restrict buf, int modes, size_t n);``` | Make STREAM use buffering mode MODE. If BUF is not NULL, use N bytes of it for buffering; else allocate an internal buffer N bytes long. | #sentinel-isdevicefile
```__device__ void setbuf(FILE *__restrict stream, char *__restrict buf);``` | If BUF is NULL, make STREAM unbuffered. Else make it use buffer BUF, of size BUFSIZ. | #sentinel-isdevicefile
```__device__ int snprintf(char *__restrict s, size_t maxlen, const char *__restrict format, ...);``` | Maximum chars of output to write in MAXLEN. | #stdarg1
```__device__ int vsnprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, va_list va);``` | Maximum chars of output to write in MAXLEN.
```__device__ int fprintf(FILE *__restrict stream, const char *__restrict format, ...);``` | Write formatted output to STREAM. | #stdarg1
```__device__ int printf(const char *__restrict format, ...);``` | Write formatted output to stdout.
```__device__ int sprintf(char *__restrict s, const char *__restrict format, ...);``` | Write formatted output to S.
```__device__ int vfprintf(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);``` | Write formatted output to S from argument list ARG.
```__device__ int vprintf(const char *__restrict format, va_list va);``` | Write formatted output to stdout from argument list ARG.
```__device__ int vsprintf(char *__restrict s, const char *__restrict format, va_list va);``` | Write formatted output to S from argument list ARG.
```__device__ int fscanf(FILE *__restrict stream, const char *__restrict format, ...);``` | Read formatted input from STREAM. | #stdarg1
```__device__ int scanf(const char *__restrict format, ...);``` | Read formatted input from stdin | #stdarg1
```__device__ int sscanf(const char *__restrict s, const char *__restrict format, ...);``` | Read formatted input from S. | #stdarg123
```__device__ int vfscanf(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);``` | Read formatted input from S into argument list ARG.
```__device__ int vscanf(const char *__restrict format, va_list va)``` | Read formatted input from stdin into argument list ARG.
```__device__ int vsscanf(const char *__restrict s, const char *__restrict format, va_list va);``` | Read formatted input from S into argument list ARG.
```__device__ int fgetc(FILE *stream);``` | Read a character from STREAM. | #sentinel-isdevicefile
```#define getc(stream)``` | Read a character from STREAM.
```__device__ int getchar(void);``` | Read a character from stdin.
```__device__ int fputc(int c, FILE *stream, bool wait = true);``` | Write a character to STREAM. | #sentinel-isdevicefile
```__device__ int putchar(int c);``` | Write a character to stdout. | #sentinel-isdevicefile
```__device__ char *fgets(char *__restrict s, int n, FILE *__restrict stream);``` | Get a newline-terminated string of finite length from STREAM.
```__device__ int fputs(const char *__restrict s, FILE *__restrict stream, bool wait = true);``` | Write a string to STREAM. | #sentinel-isdevicefile
```__device__ int puts(const char *s);``` | Write a string, followed by a newline, to stdout. | #sentinel-isdevicefile
```__device__ int ungetc(int c, FILE *stream, bool wait = true);``` | Push a character back onto the input buffer of STREAM. | #sentinel-isdevicefile
```__device__ size_t fread(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true);``` | xxxx | #sentinel-isdevicefile
```__device__ size_t fwrite(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true);``` | xxxx | #sentinel-isdevicefile
```__device__ int fseek(FILE *stream, long int off, int whence);``` | xxxx | #sentinel-isdevicefile
```__device__ long int ftell(FILE *stream);``` | xxxx | #sentinel-isdevicefile
```__device__ void rewind(FILE *stream);``` | xxxx | #sentinel-isdevicefile
```__device__ int fseeko(FILE *stream, __off_t off, int whence);``` | Seek to a certain position on STREAM.
```__device__ __off_t ftello(FILE *stream);``` | Return the current position of STREAM.
```__device__ int fgetpos(FILE *__restrict stream, fpos_t *__restrict pos);``` | Get STREAM's position
```__device__ int fsetpos(FILE *stream, const fpos_t *pos);``` | Set STREAM's position.
```__device__ int fseeko64(FILE *stream, __off64_t off, int whence);``` | xxxx | #file64
```__device__ __off64_t ftello64(FILE *stream);``` | xxxx | #file64
```__device__ int fgetpos64(FILE *__restrict stream, fpos64_t *__restrict pos);``` | xxxx | #file64
```__device__ int fsetpos64(FILE *stream, const fpos64_t *pos);``` | xxxx | #file64
```__device__ void clearerr(FILE *stream);``` | Clear the error and EOF indicators for STREAM. | #sentinel-isdevicefile
```__device__ int feof(FILE *stream);``` | Return the EOF indicator for STREAM. | #sentinel-isdevicefile
```__device__ int ferror(FILE *stream);``` | Return the error indicator for STREAM. | #sentinel-isdevicefile
```__device__ void perror(const char *s);``` | Print a message describing the meaning of the value of errno.
```__device__ int fileno(FILE *stream);``` | Return the system file descriptor for STREAM. | #sentinel-isdevicefile
```__device__ char *vmtagprintf_(void *tag, const char *format, va_list va);``` | xxxx
```__device__ char *vmprintf_(const char *format, va_list va);``` | xxxx
```__device__ char *vmnprintf_(char *__restrict s, size_t maxlen, const char *format, va_list va);``` | xxxx
