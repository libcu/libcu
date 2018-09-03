// tclParse.c --
//
//	This file contains a collection of procedures that are used to parse Tcl commands or parts of commands (like quoted
//	strings or nested sub-commands).
//
// Copyright 1991 Regents of the University of California.
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

#include "tclInt.h"

// The following table assigns a type to each character.  Only types meaningful to Tcl parsing are represented here.  The table indexes
// all 256 characters, with the negative ones first, then the positive ones.

__constant__ char _tclTypeTable[] = {
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_COMMAND_END,   TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_SPACE,         TCL_COMMAND_END,   TCL_SPACE,
	TCL_SPACE,         TCL_SPACE,         TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_SPACE,         TCL_NORMAL,        TCL_QUOTE,         TCL_NORMAL,
	TCL_DOLLAR,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_COMMAND_END,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_OPEN_BRACKET,
	TCL_BACKSLASH,     TCL_COMMAND_END,   TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,
	TCL_NORMAL,        TCL_NORMAL,        TCL_NORMAL,        TCL_OPEN_BRACE,
	TCL_NORMAL,        TCL_CLOSE_BRACE,   TCL_NORMAL,        TCL_NORMAL,
};

// Function prototypes for procedures local to this file:
static __device__ char *QuoteEnd(char *string, int term);
static __device__ char *VarNameEnd(char *string);

/*
*----------------------------------------------------------------------
*
* Tcl_Backslash --
*	Figure out how to handle a backslash sequence.
*
* Results:
*	The return value is the character that should be substituted in place of the backslash sequence that starts at src, or 0
*	if the backslash sequence should be replace by nothing (e.g. backslash followed by newline).  If readPtr isn't NULL then
*	it is filled in with a count of the number of characters in the backslash sequence.  Note:  if the backslash isn't followed
*	by characters that are understood here, then the backslash sequence is only considered to be one character long, and it
*	is replaced by a backslash char.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char Tcl_Backslash(const char *src, int *readPtr)
{
	register const char *p = src+1;
	char result;
	int count = 2;
	switch (*p) {
	case 'b':
		result = '\b';
		break;
	case 'e':
		result = 033;
		break;
	case 'f':
		result = '\f';
		break;
	case 'n':
		result = '\n';
		break;
	case 'r':
		result = '\r';
		break;
	case 't':
		result = '\t';
		break;
	case 'v':
		result = '\v';
		break;
	case 'C':
		p++;
		if (isspace(*p) || *p == 0) {
			result = 'C';
			count = 1;
			break;
		}
		count = 3;
		if (*p == 'M') {
			p++;
			if (isspace(*p) || *p == 0) {
				result = 'M' & 037;
				break;
			}
			count = 4;
			result = (*p & 037) | '\200';
			break;
		}
		count = 3;
		result = *p & 037;
		break;
	case 'M':
		p++;
		if (isspace(*p) || *p == 0) {
			result = 'M';
			count = 1;
			break;
		}
		count = 3;
		result = *p + '\200';
		break;
	case '}':
	case '{':
	case ']':
	case '[':
	case '$':
	case ' ':
	case ';':
	case '"':
	case '\\':
		result = *p;
		break;
	case '\n':
		result = 0;
		break;
	default:
		if (isdigit(*p)) {
			result = *p - '0';
			p++;
			if (!isdigit(*p)) {
				break;
			}
			count = 3;
			result = (result << 3) + (*p - '0');
			p++;
			if (!isdigit(*p)) {
				break;
			}
			count = 4;
			result = (result << 3) + (*p - '0');
			break;
		}
		result = '\\';
		count = 1;
		break;
	}
	if (readPtr != NULL) {
		*readPtr = count;
	}
	return result;
}

/*
*--------------------------------------------------------------
* TclParseQuotes --
*
*	This procedure parses a double-quoted string such as a quoted Tcl command argument or a quoted value in a Tcl
*	expression.  This procedure is also used to parse array element names within parentheses, or anything else that
*	needs all the substitutions that happen in quotes.
*
* Results:
*	The return value is a standard Tcl result, which is TCL_OK unless there was an error while parsing the
*	quoted string.  If an error occurs then interp->result contains a standard error message.  *TermPtr is filled
*	in with the address of the character just after the last one successfully processed;  this is usually the
*	character just after the matching close-quote.  The fully-substituted contents of the quotes are stored in
*	standard fashion in *pvPtr, null-terminated with pvPtr->next pointing to the terminating null character.
*
* Side effects:
*	The buffer space in pvPtr may be enlarged by calling its expandProc.
*
*--------------------------------------------------------------
*/
__device__ int TclParseQuotes(Tcl_Interp *interp, char *string, int termChar, int flags, char **termPtr, ParseValue *pvPtr)
{
	register char *src = string;
	register char *dst = pvPtr->next;
	while (true) {
		if (dst == pvPtr->end) {
			// Target buffer space is about to run out.  Make more space.
			pvPtr->next = dst;
			(*pvPtr->expandProc)(pvPtr, 1);
			dst = pvPtr->next;
		}
		int c = *src;
		src++;
		if (c == termChar) {
			*dst = '\0';
			pvPtr->next = dst;
			*termPtr = src;
			return TCL_OK;
		} else if (CHAR_TYPE(c) == TCL_NORMAL) {
copy:
			*dst = c;
			dst++;
			continue;
		} else if (c == '$') {
			char *value = Tcl_ParseVar(interp, src-1, termPtr);
			if (value == NULL) {
				return TCL_ERROR;
			}
			src = *termPtr;
			int length = strlen(value);
			if ((pvPtr->end - dst) <= length) {
				pvPtr->next = dst;
				(*pvPtr->expandProc)(pvPtr, length);
				dst = pvPtr->next;
			}
			strcpy(dst, value);
			dst += length;
			continue;
		} else if (c == '[') {
			pvPtr->next = dst;
			int result = TclParseNestedCmd(interp, src, flags, termPtr, pvPtr);
			if (result != TCL_OK) {
				return result;
			}
			src = *termPtr;
			dst = pvPtr->next;
			continue;
		} else if (c == '\\') {
			src--;
			int numRead;
			*dst = Tcl_Backslash(src, &numRead);
			if (*dst != 0) {
				dst++;
			}
			src += numRead;
			continue;
		} else if (c == '\0') {
			Tcl_ResetResult(interp);
			sprintf(interp->result, "missing %c", termChar);
			*termPtr = string-1;
			return TCL_ERROR;
		} else {
			goto copy;
		}
	}
}

/*
*--------------------------------------------------------------
*
* TclParseNestedCmd --
*	This procedure parses a nested Tcl command between brackets, returning the result of the command.
*
* Results:
*	The return value is a standard Tcl result, which is TCL_OK unless there was an error while executing the
*	nested command.  If an error occurs then interp->result contains a standard error message.  *TermPtr is filled
*	in with the address of the character just after the last one processed;  this is usually the character just
*	after the matching close-bracket, or the null character at the end of the string if the close-bracket was missing
*	(a missing close bracket is an error).  The result returned by the command is stored in standard fashion in *pvPtr,
*	null-terminated, with pvPtr->next pointing to the null character.
*
* Side effects:
*	The storage space at *pvPtr may be expanded.
*
*--------------------------------------------------------------
*/
__device__ int TclParseNestedCmd(Tcl_Interp *interp, char *string, int flags, char **termPtr, register ParseValue *pvPtr)
{
	Interp *iPtr = (Interp *) interp;
	int result = Tcl_Eval(interp, string, flags | TCL_BRACKET_TERM, termPtr);
	if (result != TCL_OK) {
		// The increment below results in slightly cleaner message in the errorInfo variable (the close-bracket will appear).
		if (**termPtr == ']') {
			*termPtr += 1;
		}
		return result;
	}
	(*termPtr) += 1;
	int length = strlen(iPtr->result);
	int shortfall = length + 1 - (int)(pvPtr->end - pvPtr->next);
	if (shortfall > 0) {
		(*pvPtr->expandProc)(pvPtr, shortfall);
	}
	strcpy(pvPtr->next, iPtr->result);
	pvPtr->next += length;
	Tcl_FreeResult(iPtr);
	iPtr->result = iPtr->resultSpace;
	iPtr->resultSpace[0] = '\0';
	return TCL_OK;
}

/*
*--------------------------------------------------------------
*
* TclParseBraces --
*	This procedure scans the information between matching curly braces.
*
* Results:
*	The return value is a standard Tcl result, which is TCL_OK unless there was an error while parsing string.
*	If an error occurs then interp->result contains a standard error message.  *TermPtr is filled
*	in with the address of the character just after the last one successfully processed;  this is usually the
*	character just after the matching close-brace.  The information between curly braces is stored in standard
*	fashion in *pvPtr, null-terminated with pvPtr->next pointing to the terminating null character.
*
* Side effects:
*	The storage space at *pvPtr may be expanded.
*
*--------------------------------------------------------------
*/
__device__ int TclParseBraces(Tcl_Interp *interp, char *string, char **termPtr, register ParseValue *pvPtr)
{
	register char *src = string;
	register char *dst = pvPtr->next;
	register char *end = pvPtr->end;
	int level = 1;
	// Copy the characters one at a time to the result area, stopping when the matching close-brace is found.
	while (true) {
		register int c = *src;
		src++;
		if (dst == end) {
			pvPtr->next = dst;
			(*pvPtr->expandProc)(pvPtr, 20);
			dst = pvPtr->next;
			end = pvPtr->end;
		}
		*dst = c;
		dst++;
		if (CHAR_TYPE(c) == TCL_NORMAL) {
			continue;
		} else if (c == '{') {
			level++;
		} else if (c == '}') {
			level--;
			if (level == 0) {
				dst--; // Don't copy the last close brace.
				break;
			}
		} else if (c == '\\') {
			// Must always squish out backslash-newlines, even when in braces.  This is needed so that this sequence can appear anywhere in a command, such as the middle of an expression.
			if (*src == '\n') {
				dst--;
				src++;
			} else {
				int count;
				Tcl_Backslash(src-1, &count);
				while (count > 1) {
					if (dst == end) {
						pvPtr->next = dst;
						(*pvPtr->expandProc)(pvPtr, 20);
						dst = pvPtr->next;
						end = pvPtr->end;
					}
					*dst = *src;
					dst++;
					src++;
					count--;
				}
			}
		} else if (c == '\0') {
			Tcl_SetResult(interp, "missing close-brace", TCL_STATIC);
			*termPtr = string-1;
			return TCL_ERROR;
		}
	}
	*dst = '\0';
	pvPtr->next = dst;
	*termPtr = src;
	return TCL_OK;
}

/*
*--------------------------------------------------------------
*
* TclParseWords --
*	This procedure parses one or more words from a command string and creates args-style pointers to fully-substituted
*	copies of those words.
*
* Results:
*	The return value is a standard Tcl result.
*	
*	*argcPtr is modified to hold a count of the number of words successfully parsed, which may be 0.  At most maxWords words
*	will be parsed.  If 0 <= *argcPtr < maxWords then it means that a command separator was seen.  If *argcPtr
*	is maxWords then it means that a command separator was not seen yet.
*
*	*TermPtr is filled in with the address of the character just after the last one successfully processed in the
*	last word.  This is either the command terminator (if *argcPtr < maxWords), the character just after the last
*	one in a word (if *argcPtr is maxWords), or the vicinity of an error (if the result is not TCL_OK).
*	
*	The pointers at *args are filled in with pointers to the fully-substituted words, and the actual contents of the
*	words are copied to the buffer at pvPtr.
*
*	If an error occurrs then an error message is left in interp->result and the information at *args, *argcPtr,
*	and *pvPtr may be incomplete.
*
* Side effects:
*	The buffer space in pvPtr may be enlarged by calling its expandProc.
*
*--------------------------------------------------------------
*/
__device__ int TclParseWords(Tcl_Interp *interp, char *string, int flags, int maxWords, char **termPtr, int *argcPtr, const char *args[], register ParseValue *pvPtr)
{
	int result;
	register char *src = string;
	char *oldBuffer = pvPtr->buffer; // Used to detect when pvPtr's buffer gets reallocated, so we can adjust all of the args pointers.
	register char *dst = pvPtr->next;
	int argc;
	for (argc = 0; argc < maxWords; argc++) {
		args[argc] = dst;
		// Skip leading space.
skipSpace:
		register int c = *src;
		int type = CHAR_TYPE(c);
		while (type == TCL_SPACE) { src++; c = *src; type = CHAR_TYPE(c); }
		// Handle the normal case (i.e. no leading double-quote or brace).
		if (type == TCL_NORMAL) {
normalArg:
			do {
				if (dst == pvPtr->end) {
					// Target buffer space is about to run out.  Make more space.
					pvPtr->next = dst;
					(*pvPtr->expandProc)(pvPtr, 1);
					dst = pvPtr->next;
				}
				if (type == TCL_NORMAL) {
copy:
					*dst = c; dst++; src++;
				} else if (type == TCL_SPACE) {
					goto wordEnd;
				} else if (type == TCL_DOLLAR) {
					char *value = Tcl_ParseVar(interp, src, termPtr);
					if (value == NULL) {
						return TCL_ERROR;
					}
					src = *termPtr;
					int length = strlen(value);
					if ((pvPtr->end - dst) <= length) {
						pvPtr->next = dst;
						(*pvPtr->expandProc)(pvPtr, length);
						dst = pvPtr->next;
					}
					strcpy(dst, value);
					dst += length;
				} else if (type == TCL_COMMAND_END) {
					if (c == ']' && !(flags & TCL_BRACKET_TERM)) {
						goto copy;
					}
					// End of command;  simulate a word-end first, so that the end-of-command can be processed as the first thing in a new word.
					goto wordEnd;
				} else if (type == TCL_OPEN_BRACKET) {
					pvPtr->next = dst;
					result = TclParseNestedCmd(interp, src+1, flags, termPtr, pvPtr);
					if (result != TCL_OK) {
						return result;
					}
					src = *termPtr;
					dst = pvPtr->next;
				} else if (type == TCL_BACKSLASH) {
					int numRead;
					*dst = Tcl_Backslash(src, &numRead);
					if (*dst != 0) {
						dst++;
					}
					src += numRead;
				} else {
					goto copy;
				}
				c = *src; type = CHAR_TYPE(c);
			} while (true);
		} else {
			// Check for the end of the command.
			if (type == TCL_COMMAND_END) {
				if (flags & TCL_BRACKET_TERM) {
					if (c == '\0') {
						Tcl_SetResult(interp, "missing close-bracket", TCL_STATIC);
						return TCL_ERROR;
					}
				} else {
					if (c == ']') {
						goto normalArg;
					}
				}
				goto done;
			}
			// Now handle the special cases: open braces, double-quotes, and backslash-newline.
			pvPtr->next = dst;
			if (type == TCL_QUOTE) {
				result = TclParseQuotes(interp, src+1, '"', flags, termPtr, pvPtr);
			} else if (type == TCL_OPEN_BRACE) {
				result = TclParseBraces(interp, src+1, termPtr, pvPtr);
			} else if (type == TCL_BACKSLASH && src[1] == '\n') {
				src += 2;
				goto skipSpace;
			} else {
				goto normalArg;
			}
			if (result != TCL_OK) {
				return result;
			}
			// Back from quotes or braces;  make sure that the terminating character was the end of the word.  Have to be careful here
			// to handle continuation lines (i.e. lines ending in backslash).
			c = **termPtr;
			if (c == '\\' && (*termPtr)[1] == '\n') {
				c = (*termPtr)[2];
			}
			type = CHAR_TYPE(c);
			if (type != TCL_SPACE && type != TCL_COMMAND_END) {
				if (*src == '"') {
					Tcl_SetResult(interp, "extra characters after close-quote", TCL_STATIC);
				} else {
					Tcl_SetResult(interp, "extra characters after close-brace", TCL_STATIC);
				}
				return TCL_ERROR;
			}
			src = *termPtr;
			dst = pvPtr->next;
		}
		// We're at the end of a word, so add a null terminator.  Then see if the buffer was re-allocated during this word.  If so, update all of the args pointers.
wordEnd:
		*dst = '\0';
		dst++;
		if (oldBuffer != pvPtr->buffer) {
			for (int i = 0; i <= argc; i++) {
				args[i] = pvPtr->buffer + (args[i] - oldBuffer);
			}
			oldBuffer = pvPtr->buffer;
		}
	}
done:
	pvPtr->next = dst;
	*termPtr = src;
	*argcPtr = argc;
	return TCL_OK;
}

/*
*--------------------------------------------------------------
*
* TclExpandParseValue --
*	This procedure is commonly used as the value of the expandProc in a ParseValue.  It uses malloc to allocate
*	more space for the result of a parse.
*
* Results:
*	The buffer space in *pvPtr is reallocated to something larger, and if pvPtr->clientData is non-zero the old
*	buffer is freed.  Information is copied from the old buffer to the new one.
*
* Side effects:
*	None.
*
*--------------------------------------------------------------
*/
__device__ void TclExpandParseValue(register ParseValue *pvPtr, int needed)
{
	// Either double the size of the buffer or add enough new space to meet the demand, whichever produces a larger new buffer.
	int newSpace = (int)(pvPtr->end - pvPtr->buffer) + 1;
	if (newSpace < needed) {
		newSpace += needed;
	} else {
		newSpace += newSpace;
	}
	char *new_ = (char *)_allocFast((unsigned)newSpace);

	// Copy from old buffer to new, free old buffer if needed, and mark new buffer as malloc-ed.
	memcpy(new_, pvPtr->buffer, pvPtr->next - pvPtr->buffer);
	pvPtr->next = new_ + (pvPtr->next - pvPtr->buffer);
	if (pvPtr->clientData != 0) {
		_freeFast(pvPtr->buffer);
	}
	pvPtr->buffer = new_;
	pvPtr->end = new_ + newSpace - 1;
	pvPtr->clientData = (ClientData)1;
}

/*
*----------------------------------------------------------------------
*
* TclWordEnd --
*	Given a pointer into a Tcl command, find the end of the next word of the command.
*
* Results:
*	The return value is a pointer to the last character that's part of the word pointed to by "start".  If the word doesn't end
*	properly within the string then the return value is the address of the null character at the end of the string.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *TclWordEnd(char *start, int nested)
{
	int count;
	register char *p = start;
	while (isspace(*p)) {
		p++;
	}
	// Handle words beginning with a double-quote or a brace.
	if (*p == '"') {
		p = QuoteEnd(p+1, '"');
		if (*p == 0) {
			return p;
		}
		p++;
	} else if (*p == '{') {
		int braces = 1;
		while (braces != 0) {
			p++;
			while (*p == '\\') {
				Tcl_Backslash(p, &count);
				p += count;
			}
			if (*p == '}') {
				braces--;
			} else if (*p == '{') {
				braces++;
			} else if (*p == 0) {
				return p;
			}
		}
		p++;
	}

	// Handle words that don't start with a brace or double-quote. This code is also invoked if the word starts with a brace or
	// double-quote and there is garbage after the closing brace or quote.  This is an error as far as Tcl_Eval is concerned, but
	// for here the garbage is treated as part of the word.
	while (true) {
		if (*p == '[') {
			for (p++; *p != ']'; p++) {
				p = TclWordEnd(p, 1);
				if (*p == 0) {
					return p;
				}
			}
			p++;
		} else if (*p == '\\') {
			Tcl_Backslash(p, &count);
			p += count;
			if (*p == 0 && count == 2 && p[-1] == '\n') {
				return p;
			}
		} else if (*p == '$') {
			p = VarNameEnd(p);
			if (*p == 0) {
				return p;
			}
			p++;
		} else if (*p == ';') {
			// Include the semi-colon in the word that is returned.
			return p;
		} else if (isspace(*p)) {
			return p-1;
		} else if (*p == ']' && nested) {
			return p-1;
		} else if (*p == 0) {
			if (nested) {
				// Nested commands can't end because of the end of the string.
				return p;
			}
			return p-1;
		} else {
			p++;
		}
	}
}

/*
*----------------------------------------------------------------------
*
* QuoteEnd --
*	Given a pointer to a string that obeys the parsing conventions for quoted things in Tcl, find the end of that quoted thing.
*	The actual thing may be a quoted argument or a parenthesized index name.
*
* Results:
*	The return value is a pointer to the last character that is part of the quoted string (i.e the character that's equal to
*	term).  If the quoted string doesn't terminate properly then the return value is a pointer to the null character at the end of the string.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ char *QuoteEnd(char *string, int term)
{
	register char *p = string;
	int count;
	while (*p != term) {
		if (*p == '\\') {
			Tcl_Backslash(p, &count);
			p += count;
		} else if (*p == '[') {
			for (p++; *p != ']'; p++) {
				p = TclWordEnd(p, 1);
				if (*p == 0) {
					return p;
				}
			}
			p++;
		} else if (*p == '$') {
			p = VarNameEnd(p);
			if (*p == 0) {
				return p;
			}
			p++;
		} else if (*p == 0) {
			return p;
		} else {
			p++;
		}
	}
	return p-1;
}

/*
*----------------------------------------------------------------------
*
* VarNameEnd --
*	Given a pointer to a variable reference using $-notation, find the end of the variable name spec.
*
* Results:
*	The return value is a pointer to the last character that is part of the variable name.  If the variable name doesn't
*	terminate properly then the return value is a pointer to the null character at the end of the string.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ char *VarNameEnd(char *string)
{
	register char *p = string+1;
	if (*p == '{') {
		for (p++; (*p != '}') && (*p != 0); p++) { } // Empty loop body.
		return p;
	}
	// Two leading colons are OK
	if (p[0] == ':' && p[1] == ':') {
		p += 2;
	}
	while (isalnum(*p) || *p == '_') {
		p++;
	}
	if (*p == '(' && p != string+1) {
		return QuoteEnd(p+1, ')');
	}
	return p-1;
}

/*
*----------------------------------------------------------------------
*
* Tcl_ParseVar --
*	Given a string starting with a $ sign, parse off a variable name and return its value.
*
* Results:
*	The return value is the contents of the variable given by the leading characters of string.  If termPtr isn't NULL,
*	*termPtr gets filled in with the address of the character just after the last one in the variable specifier.  If the
*	variable doesn't exist, then the return value is NULL and an error message will be left in interp->result.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
__device__ char *Tcl_ParseVar(Tcl_Interp *interp, register char *string, char **termPtr)
{
#define NUM_CHARS 200
	char copyStorage[NUM_CHARS];
	/*
	* There are three cases:
	* 1. The $ sign is followed by an open curly brace.  Then the variable name is everything up to the next close curly brace, and the variable is a scalar variable.
	* 2. The $ sign is not followed by an open curly brace.  Then the variable name is everything up to the next character that isn't
	*    a letter, digit, or underscore.  If the following character is an open parenthesis, then the information between parentheses is
	*    the array element name, which can include any of the substitutions permissible between quotes.
	* 3. The $ sign is followed by something that isn't a letter, digit, colon or underscore:  in this case, there is no variable name, and "$" is returned.
	*/
	ParseValue pv;
	char *name1, *name1End, *result;
	register char *name2 = NULL;
	string++;
	if (*string == '{') {
		string++;
		name1 = string;
		while (*string != '}') {
			if (*string == 0) {
				Tcl_SetResult(interp, "missing close-brace for variable name", TCL_STATIC);
				if (termPtr != 0) {
					*termPtr = string;
				}
				return NULL;
			}
			string++;
		}
		name1End = string;
		string++;
	} else {
		name1 = string;
		// Two leading colons are OK
		if (string[0] == ':' && string[1] == ':') {
			string += 2;
		}
		while (isalnum(*string) || *string == '_') {
			string++;
		}
		if (string == name1) {
			if (termPtr != 0) {
				*termPtr = string;
			}
			return "$";
		}
		name1End = string;
		if (*string == '(') {
			// Perform substitutions on the array element name, just as is done for quotes.
			pv.buffer = pv.next = copyStorage;
			pv.end = copyStorage + NUM_CHARS - 1;
			pv.expandProc = TclExpandParseValue;
			pv.clientData = (ClientData)NULL;
			char *end;
			if (TclParseQuotes(interp, string+1, ')', 0, &end, &pv) != TCL_OK) {
				char msg[100];
				sprintf(msg, "\n    (parsing index for array \"%.*s\")", (int)(string-name1), name1);
				Tcl_AddErrorInfo(interp, msg);
				result = NULL;
				name2 = pv.buffer;
				if (termPtr != 0) {
					*termPtr = end;
				}
				goto done;
			}
			string = end;
			name2 = pv.buffer;
		}
	}
	if (termPtr != 0) {
		*termPtr = string;
	}
	if (((Interp *)interp)->noEval) {
		return "";
	}
	char c = *name1End;
	*name1End = 0;
	result = Tcl_GetVar2(interp, name1, name2, TCL_LEAVE_ERR_MSG);
	*name1End = c;
done:
	if (name2 != NULL && pv.buffer != copyStorage) {
		_freeFast(pv.buffer);
	}
	return result;
}
