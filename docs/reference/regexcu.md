## #include <regexcu.h>

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ int regcomp_(regex_t *preg, const char *regex, int cflags);``` | xxxx
```__device__ int regexec_(regex_t *preg, const char *string, size_t nmatch, regmatch_t pmatch[], int eflags);``` | xxxx
```__device__ size_t regerror_(int errcode, const regex_t *preg, char *errbuf, size_t errbuf_size);``` | xxxx
```__device__ void regfree_(regex_t *preg);``` | xxxx

## Compile Flags
Define | Description
--- | ---
```REG_EXTENDED``` | xxxx
```REG_NEWLINE``` | xxxx
```REG_ICASE``` | xxxx
```REG_NOTBOL``` | xxxx


## Error Codes
Define | Description
--- | ---
```REG_NOERROR``` | xxxx
```REG_NOMATCH``` | xxxx
```REG_BADPAT``` | xxxx
```REG_ERR_NULL_ARGUMENT``` | xxxx
```REG_ERR_UNKNOWN``` | xxxx
```REG_ERR_TOO_BIG``` | xxxx
```REG_ERR_NOMEM``` | xxxx
```REG_ERR_TOO_MANY_PAREN``` | xxxx
```REG_ERR_UNMATCHED_PAREN``` | xxxx
```REG_ERR_UNMATCHED_BRACES``` | xxxx
```REG_ERR_BAD_COUNT``` | xxxx
```REG_ERR_JUNK_ON_END``` | xxxx
```REG_ERR_OPERAND_COULD_BE_EMPTY``` | xxxx
```REG_ERR_NESTED_COUNT``` | xxxx
```REG_ERR_INTERNAL``` | xxxx
```REG_ERR_COUNT_FOLLOWS_NOTHING``` | xxxx
```REG_ERR_TRAILING_BACKSLASH``` | xxxx
```REG_ERR_CORRUPTED``` | xxxx
```REG_ERR_NULL_CHAR``` | xxxx
```REG_ERR_NUM``` | xxxx