/*
regex.h - defines regex functions
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//#pragma once
#ifndef _REGEXCU_H
#define _REGEXCU_H
#define _REGEX_H

#include <crtdefscu.h>

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct {
		int rm_so;
		int rm_eo;
	} regmatch_t;

	typedef struct regexp {
		int re_nsub;		// number of parenthesized subexpressions

		// private
		int cflags;			// Flags used when compiling
		int err;			// Any error which occurred during compile
		int regstart;		// Internal use only
		int reganch;		// Internal use only
		int regmust;		// Internal use only
		int regmlen;		// Internal use only
		int *program;		// Allocated
		// working state - compile
		const char *regparse;	// Input-scan pointer
		int p;					// Current output pos in program
		int proglen;			// Allocated program size
		// working state - exec
		int eflags;				// Flags used when executing
		const char *start;		// Initial string pointer
		const char *reginput;	// Current input pointer
		const char *regbol;		// Beginning of input, for ^ check
		// Input to regexec()
		regmatch_t *pmatch;		// submatches will be stored here
		int nmatch;				// size of pmatch[]
	} regex_t;

#define REG_EXTENDED 0
#define REG_NEWLINE 1
#define REG_ICASE 2
#define REG_NOTBOL 16

	enum {
		REG_NOERROR,      // Success
		REG_NOMATCH,      // Didn't find a match (for regexec)
		REG_BADPAT,		  // >= REG_BADPAT is an error
		REG_ERR_NULL_ARGUMENT,
		REG_ERR_UNKNOWN,
		REG_ERR_TOO_BIG,
		REG_ERR_NOMEM,
		REG_ERR_TOO_MANY_PAREN,
		REG_ERR_UNMATCHED_PAREN,
		REG_ERR_UNMATCHED_BRACES,
		REG_ERR_BAD_COUNT,
		REG_ERR_JUNK_ON_END,
		REG_ERR_OPERAND_COULD_BE_EMPTY,
		REG_ERR_NESTED_COUNT,
		REG_ERR_INTERNAL,
		REG_ERR_COUNT_FOLLOWS_NOTHING,
		REG_ERR_TRAILING_BACKSLASH,
		REG_ERR_CORRUPTED,
		REG_ERR_NULL_CHAR,
		REG_ERR_NUM
	};

	extern __device__ int regcomp_(regex_t *preg, const char *regex, int cflags);
#define regcomp regcomp_
	extern __device__ int regexec_(regex_t *preg, const char *string, size_t nmatch, regmatch_t pmatch[], int eflags);
#define regexec regexec_
	extern __device__ size_t regerror_(int errcode, const regex_t *preg, char *errbuf, size_t errbuf_size);
#define regerror regerror_
	extern __device__ void regfree_(regex_t *preg);
#define regfree regfree_

#ifdef __cplusplus
}
#endif

#endif /* _REGEXCU_H */
