## #include <ctypecu.h>

Also includes:
```
#include <ctype.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ int isctype(int c, int type);``` | Is ctype based on "type"
```__device__ int isalnum(int c);``` | Is alphabet/numeric character
```__device__ int isalpha(int c);``` | Is alphabet character
```__device__ int iscntrl(int c);``` | Is control character
```__device__ int isdigit(int c);``` | Is digit character
```__device__ int islower(int c);``` | Is lowercase alphabet character
```__device__ int isgraph(int c);``` | Is graph character
```__device__ int isprint(int c);``` | Is print character
```__device__ int ispunct(int c);``` | Is punctiation character
```__device__ int isspace(int c);``` | Is white space character
```__device__ int isupper(int c);``` | Is uppercase alphabet character
```__device__ int isxdigit(int c);``` | Is xdigit character
```__device__ int tolower(int c);``` | Return the lowercase version of C.
```__device__ int toupper(int c);``` | Return the uppercase version of C.
```#define _tolower(c)``` | | #existing
```#define _toupper(c)``` | | #existing
```__device__ int isblank(int c);``` | Is blank character | #portable
```__device__ int isidchar(int c);``` | Is id character | #portable
