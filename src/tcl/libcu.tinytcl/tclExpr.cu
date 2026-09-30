// tclExpr.c --
//
//	This file contains the code to evaluate expressions for Tcl.
//
//	This implementation of floating-point support was modelled after an initial implementation by Bill Carpenter.
//
// Copyright 1987-1991 Regents of the University of California
// Permission to use, copy, modify, and distribute this software and its documentation for any purpose and without
// fee is hereby granted, provided that the above copyright notice appear in all copies.  The University of California
// makes no representations about the suitability of this software for any purpose.  It is provided "as is" without
// express or implied warranty.

//#include <errnocu.h>
#include "tclInt.h"

// The stuff below is a bit of a hack so that this file can be used in environments that include no UNIX, i.e. no errno.  Just define errno here.
//extern int errno;
//#include <errno.h>
#define ERANGE 34

// The data structure below is used to describe an expression value, which can be either an integer (the usual case), a double-precision
// floating-point value, or a string.  A given number has only one value at a time.
#define STATIC_STRING_SPACE 150

typedef struct {
	long intValue;			// Integer value, if any.
	double doubleValue;	// Floating-point value, if any.
	ParseValue pv;			// Used to hold a string value, if any.
	char staticSpace[STATIC_STRING_SPACE];
	// Storage for small strings;  large ones are malloc-ed.
	int type;				// Type of value:  TYPE_INT, TYPE_DOUBLE, or TYPE_STRING.
} Value;

// Valid values for type:
#define TYPE_INT	0
#define TYPE_DOUBLE	1
#define TYPE_STRING	2

// The data structure below describes the state of parsing an expression. It's passed among the routines in this module.
typedef struct {
	char *originalExpr;		// The entire expression, as originally passed to Tcl_Expr.
	char *expr;				// Position to the next character to be scanned from the expression string.
	int token;				// Type of the last token to be parsed from expr.  See below for definitions. Corresponds to the characters just before expr. */
} ExprInfo;

// The token types are defined below.  In addition, there is a table associating a precedence with each operator.  The order of types is important.  Consult the code before changing it.
#define VALUE		0
#define OPEN_PAREN	1
#define CLOSE_PAREN	2
#define END		3
#define UNKNOWN		4

// Binary operators:
#define MULT		8
#define DIVIDE		9
#define MOD		10
#define PLUS		11
#define MINUS		12
#define LEFT_SHIFT	13
#define RIGHT_SHIFT	14
#define LESS		15
#define GREATER		16
#define LEQ		17
#define GEQ		18
#define EQUAL		19
#define NEQ		20
#define BIT_AND		21
#define BIT_XOR		22
#define BIT_OR		23
#define AND		24
#define OR		25
#define QUESTY		26
#define COLON		27

// Unary operators:
#define	UNARY_MINUS	28
#define NOT		29
#define BIT_NOT		30

// Precedence table.  The values for non-operator token types are ignored.
__constant__ int _precTable[] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	11, 11, 11,			// MULT, DIVIDE, MOD
	10, 10,				// PLUS, MINUS
	9, 9,				// LEFT_SHIFT, RIGHT_SHIFT
	8, 8, 8, 8,			// LESS, GREATER, LEQ, GEQ
	7, 7,				// EQUAL, NEQ
	6,					// BIT_AND
	5,					// BIT_XOR
	4,					// BIT_OR
	3,					// AND
	2,					// OR
	1, 1,				// QUESTY, COLON
	12, 12, 12			// UNARY_MINUS, NOT, BIT_NOT
};

// Mapping from operator numbers to strings;  used for error messages.
__constant__ const char *_operatorStrings[] = {
	"VALUE", "(", ")", "END", "UNKNOWN", "5", "6", "7",
	"*", "/", "%", "+", "-", "<<", ">>", "<", ">", "<=",
	">=", "==", "!=", "&", "^", "|", "&&", "||", "?", ":",
	"-", "!", "~"
};

// Declarations for local procedures to this file:
static __device__ int ExprGetValue(Tcl_Interp *interp, ExprInfo *infoPtr, int prec, Value *valuePtr);
static __device__ int ExprLex(Tcl_Interp *interp, ExprInfo *infoPtr, Value *valuePtr);
static __device__ void ExprMakeString(Value *valuePtr);
static __device__ int ExprParseString(Tcl_Interp *interp, char *string, Value *valuePtr);
static __device__ int ExprTopLevel(Tcl_Interp *interp, char *string, Value *valuePtr);

/*
*--------------------------------------------------------------
*
* ExprParseString --
*	Given a string (such as one coming from command or variable substitution), make a Value based on the string.  The value
*	will be a floating-point or integer, if possible, or else it will just be a copy of the string.
*
* Results:
*	TCL_OK is returned under normal circumstances, and TCL_ERROR is returned if a floating-point overflow or underflow occurred
*	while reading in a number.  The value at *valuePtr is modified to hold a number, if possible.
*
* Side effects:
*	None.
*
*--------------------------------------------------------------
*/
static __device__ int ExprParseString(Tcl_Interp *interp, char *string, Value *valuePtr) {
	// Try to convert the string to a number.
	register char c = *string;
	if ((c >= '0' && c <= '9') || c == '-' || c == '.') {
		char *term;
		valuePtr->type = TYPE_INT;
		errno = 0;
		valuePtr->intValue = (int)strtol(string, &term, 0);
		c = *term;
		if (c == '\0' && errno != ERANGE) {
			return TCL_OK;
		}
		if (c == '.' || c == 'e' || c == 'E' || errno == ERANGE) {
			errno = 0;
			valuePtr->doubleValue = strtod(string, &term);
			if (errno == ERANGE) {
				Tcl_ResetResult(interp);
				if (valuePtr->doubleValue == 0.0) {
					Tcl_AppendResult(interp, "floating-point value \"", string, "\" too small to represent", (char *)NULL);
				}
				else {
					Tcl_AppendResult(interp, "floating-point value \"", string, "\" too large to represent", (char *)NULL);
				}
				return TCL_ERROR;
			}
			if (*term == '\0') {
				valuePtr->type = TYPE_DOUBLE;
				return TCL_OK;
			}
		}
	}
	// Not a valid number.  Save a string value (but don't do anything if it's already the value).
	valuePtr->type = TYPE_STRING;
	if (string != valuePtr->pv.buffer) {
		int length = strlen(string);
		valuePtr->pv.next = valuePtr->pv.buffer;
		int shortfall = length - (int)(valuePtr->pv.end - valuePtr->pv.buffer);
		if (shortfall > 0) {
			(*valuePtr->pv.expandProc)(&valuePtr->pv, shortfall);
		}
		strcpy(valuePtr->pv.buffer, string);
	}
	return TCL_OK;
}

/*
*----------------------------------------------------------------------
*
* ExprLex --
*	Lexical analyzer for expression parser:  parses a single value, operator, or other syntactic element from an expression string.
*
* Results:
*	TCL_OK is returned unless an error occurred while doing lexical analysis or executing an embedded command.  In that case a
*	standard Tcl error is returned, using interp->result to hold an error message.  In the event of a successful return, the token
*	and field in infoPtr is updated to refer to the next symbol in the expression string, and the expr field is advanced past that
*	token;  if the token is a value, then the value is stored at valuePtr.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ int ExprLex(Tcl_Interp *interp, register ExprInfo *infoPtr, register Value *valuePtr) {
	int result;
	register char *p = infoPtr->expr;
	register char c = *p;
	while (isspace(c)) {
		p++;
		c = *p;
	}
	infoPtr->expr = p + 1;
	char *var;
	switch (c) {
	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	case '8':
	case '9':
	case '.':
		// Number.  First read an integer.  Then if it looks like there's a floating-point number (or if it's too big a
		// number to fit in an integer), parse it as a floating-point number.
		infoPtr->token = VALUE;
		valuePtr->type = TYPE_INT;
		errno = 0;
		char *term;
		valuePtr->intValue = strtoul(p, &term, 0);
		c = *term;
		if (c == '.' || c == 'e' || c == 'E' || errno == ERANGE) {
			char *term2;
			errno = 0;
			valuePtr->doubleValue = strtod(p, &term2);
			if (errno == ERANGE) {
				Tcl_ResetResult(interp);
				if (valuePtr->doubleValue == 0.0) {
					interp->result = (char *)"floating-point value too small to represent";
				}
				else {
					interp->result = (char *)"floating-point value too large to represent";
				}
				return TCL_ERROR;
			}
			if (term2 == infoPtr->expr) {
				interp->result = (char *)"poorly-formed floating-point value";
				return TCL_ERROR;
			}
			valuePtr->type = TYPE_DOUBLE;
			infoPtr->expr = term2;
		}
		else {
			infoPtr->expr = term;
		}
		return TCL_OK;
	case '$':
		// Variable.  Fetch its value, then see if it makes sense as an integer or floating-point number.
		infoPtr->token = VALUE;
		var = Tcl_ParseVar(interp, p, &infoPtr->expr);
		if (var == NULL) {
			return TCL_ERROR;
		}
		if (((Interp *)interp)->noEval) {
			valuePtr->type = TYPE_INT;
			valuePtr->intValue = 0;
			return TCL_OK;
		}
		return ExprParseString(interp, var, valuePtr);
	case '[':
		infoPtr->token = VALUE;
		result = Tcl_Eval(interp, p + 1, TCL_BRACKET_TERM, &infoPtr->expr);
		if (result != TCL_OK) {
			return result;
		}
		infoPtr->expr++;
		if (((Interp *)interp)->noEval) {
			valuePtr->type = TYPE_INT;
			valuePtr->intValue = 0;
			Tcl_ResetResult(interp);
			return TCL_OK;
		}
		result = ExprParseString(interp, interp->result, valuePtr);
		if (result != TCL_OK) {
			return result;
		}
		Tcl_ResetResult(interp);
		return TCL_OK;
	case '"':
		infoPtr->token = VALUE;
		result = TclParseQuotes(interp, infoPtr->expr, '"', 0, &infoPtr->expr, &valuePtr->pv);
		if (result != TCL_OK) {
			return result;
		}
		return ExprParseString(interp, valuePtr->pv.buffer, valuePtr);
	case '{':
		infoPtr->token = VALUE;
		result = TclParseBraces(interp, infoPtr->expr, &infoPtr->expr, &valuePtr->pv);
		if (result != TCL_OK) {
			return result;
		}
		return ExprParseString(interp, valuePtr->pv.buffer, valuePtr);
	case '(':
		infoPtr->token = OPEN_PAREN;
		return TCL_OK;
	case ')':
		infoPtr->token = CLOSE_PAREN;
		return TCL_OK;
	case '*':
		infoPtr->token = MULT;
		return TCL_OK;
	case '/':
		infoPtr->token = DIVIDE;
		return TCL_OK;
	case '%':
		infoPtr->token = MOD;
		return TCL_OK;
	case '+':
		infoPtr->token = PLUS;
		return TCL_OK;
	case '-':
		infoPtr->token = MINUS;
		return TCL_OK;
	case '?':
		infoPtr->token = QUESTY;
		return TCL_OK;
	case ':':
		infoPtr->token = COLON;
		return TCL_OK;
	case '<':
		switch (p[1]) {
		case '<':
			infoPtr->expr = p + 2;
			infoPtr->token = LEFT_SHIFT;
			break;
		case '=':
			infoPtr->expr = p + 2;
			infoPtr->token = LEQ;
			break;
		default:
			infoPtr->token = LESS;
			break;
		}
		return TCL_OK;
	case '>':
		switch (p[1]) {
		case '>':
			infoPtr->expr = p + 2;
			infoPtr->token = RIGHT_SHIFT;
			break;
		case '=':
			infoPtr->expr = p + 2;
			infoPtr->token = GEQ;
			break;
		default:
			infoPtr->token = GREATER;
			break;
		}
		return TCL_OK;
	case '=':
		if (p[1] == '=') {
			infoPtr->expr = p + 2;
			infoPtr->token = EQUAL;
		}
		else {
			infoPtr->token = UNKNOWN;
		}
		return TCL_OK;
	case '!':
		if (p[1] == '=') {
			infoPtr->expr = p + 2;
			infoPtr->token = NEQ;
		}
		else {
			infoPtr->token = NOT;
		}
		return TCL_OK;
	case '&':
		if (p[1] == '&') {
			infoPtr->expr = p + 2;
			infoPtr->token = AND;
		}
		else {
			infoPtr->token = BIT_AND;
		}
		return TCL_OK;
	case '^':
		infoPtr->token = BIT_XOR;
		return TCL_OK;
	case '|':
		if (p[1] == '|') {
			infoPtr->expr = p + 2;
			infoPtr->token = OR;
		}
		else {
			infoPtr->token = BIT_OR;
		}
		return TCL_OK;
	case '~':
		infoPtr->token = BIT_NOT;
		return TCL_OK;
	case 0:
		infoPtr->token = END;
		infoPtr->expr = p;
		return TCL_OK;
	default:
		infoPtr->expr = p + 1;
		infoPtr->token = UNKNOWN;
		return TCL_OK;
	}
}

/*
*----------------------------------------------------------------------
*
* ExprGetValue --
*	Parse a "value" from the remainder of the expression in infoPtr.
*
* Results:
*	Normally TCL_OK is returned.  The value of the expression is returned in *valuePtr.  If an error occurred, then interp->result
*	contains an error message and TCL_ERROR is returned. InfoPtr->token will be left pointing to the token AFTER the
*	expression, and infoPtr->expr will point to the character just after the terminating token.
*
* Side effects:
*	None.
*
*----------------------------------------------------------------------
*/
static __device__ int ExprGetValue(Tcl_Interp *interp, register ExprInfo *infoPtr, int prec, Value *valuePtr) {
	Interp *iPtr = (Interp *)interp;
	int operator_; // Current operator (either unary or binary).
	int badType; // Type of offending argument;  used for error messages.
	// There are two phases to this procedure.  First, pick off an initial value.  Then, parse (binary operator, value) pairs until done.
	Value value2; // Second operand for current operator.
	value2.pv.buffer = value2.pv.next = value2.staticSpace;
	value2.pv.end = value2.pv.buffer + STATIC_STRING_SPACE - 1;
	value2.pv.expandProc = TclExpandParseValue;
	value2.pv.clientData = (ClientData)NULL;
	int result = ExprLex(interp, infoPtr, valuePtr);
	if (result != TCL_OK) {
		goto done;
	}
	bool gotOp; gotOp = false; // Non-zero means already lexed the operator (while picking up value for unary operator).  Don't lex again.
	if (infoPtr->token == OPEN_PAREN) {
		// Parenthesized sub-expression.
		result = ExprGetValue(interp, infoPtr, -1, valuePtr);
		if (result != TCL_OK) {
			goto done;
		}
		if (infoPtr->token != CLOSE_PAREN) {
			Tcl_ResetResult(interp);
			Tcl_AppendResult(interp, "unmatched parentheses in expression \"", infoPtr->originalExpr, "\"", (char *)NULL);
			result = TCL_ERROR;
			goto done;
		}
	}
	else {
		if (infoPtr->token == MINUS) {
			infoPtr->token = UNARY_MINUS;
		}
		if (infoPtr->token >= UNARY_MINUS) {
			// Process unary operators.
			operator_ = infoPtr->token;
			result = ExprGetValue(interp, infoPtr, _precTable[infoPtr->token], valuePtr);
			if (result != TCL_OK) {
				goto done;
			}
			switch (operator_) {
			case UNARY_MINUS:
				if (valuePtr->type == TYPE_INT) {
					valuePtr->intValue = -valuePtr->intValue;
				}
				else if (valuePtr->type == TYPE_DOUBLE) {
					valuePtr->doubleValue = -valuePtr->doubleValue;
				}
				else {
					badType = valuePtr->type;
					goto illegalType;
				}
				break;
			case NOT:
				if (valuePtr->type == TYPE_INT) {
					valuePtr->intValue = !valuePtr->intValue;
				}
				else if (valuePtr->type == TYPE_DOUBLE) {
					// Theoretically, should be able to use "!valuePtr->intValue", but apparently some compilers can't handle it.
					if (valuePtr->doubleValue == 0.0) {
						valuePtr->intValue = 1;
					}
					else {
						valuePtr->intValue = 0;
					}
					valuePtr->type = TYPE_INT;
				}
				else {
					badType = valuePtr->type;
					goto illegalType;
				}
				break;
			case BIT_NOT:
				if (valuePtr->type == TYPE_INT) {
					valuePtr->intValue = ~valuePtr->intValue;
				}
				else {
					badType = valuePtr->type;
					goto illegalType;
				}
				break;
			}
			gotOp = true;
		}
		else if (infoPtr->token != VALUE) {
			goto syntaxError;
		}
	}

	// Got the first operand.  Now fetch (operator, operand) pairs.
	if (!gotOp) {
		result = ExprLex(interp, infoPtr, &value2);
		if (result != TCL_OK) {
			goto done;
		}
	}
	while (true) {
		operator_ = infoPtr->token;
		value2.pv.next = value2.pv.buffer;
		if (operator_ < MULT || operator_ >= UNARY_MINUS) {
			if (operator_ == END || operator_ == CLOSE_PAREN) {
				result = TCL_OK;
				goto done;
			}
			else {
				goto syntaxError;
			}
		}
		if (_precTable[operator_] <= prec) {
			result = TCL_OK;
			goto done;
		}

		// If we're doing an AND or OR and the first operand already determines the result, don't execute anything in the
		// second operand:  just parse.  Same style for ?: pairs.
		if (operator_ == AND || operator_ == OR || operator_ == QUESTY) {
			if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue != 0);
				valuePtr->type = TYPE_INT;
			}
			else if (valuePtr->type == TYPE_STRING) {
				badType = TYPE_STRING;
				goto illegalType;
			}
			if ((operator_ == AND && !valuePtr->intValue) || (operator_ == OR && valuePtr->intValue)) {
				iPtr->noEval++;
				result = ExprGetValue(interp, infoPtr, _precTable[operator_], &value2);
				iPtr->noEval--;
			}
			else if (operator_ == QUESTY) {
				if (valuePtr->intValue != 0) {
					valuePtr->pv.next = valuePtr->pv.buffer;
					result = ExprGetValue(interp, infoPtr, _precTable[operator_], valuePtr);
					if (result != TCL_OK) {
						goto done;
					}
					if (infoPtr->token != COLON) {
						goto syntaxError;
					}
					value2.pv.next = value2.pv.buffer;
					iPtr->noEval++;
					result = ExprGetValue(interp, infoPtr, _precTable[operator_], &value2);
					iPtr->noEval--;
				}
				else {
					iPtr->noEval++;
					result = ExprGetValue(interp, infoPtr, _precTable[operator_], &value2);
					iPtr->noEval--;
					if (result != TCL_OK) {
						goto done;
					}
					if (infoPtr->token != COLON) {
						goto syntaxError;
					}
					valuePtr->pv.next = valuePtr->pv.buffer;
					result = ExprGetValue(interp, infoPtr, _precTable[operator_], valuePtr);
				}
			}
			else {
				result = ExprGetValue(interp, infoPtr, _precTable[operator_], &value2);
			}
		}
		else {
			result = ExprGetValue(interp, infoPtr, _precTable[operator_], &value2);
		}
		if (result != TCL_OK) {
			goto done;
		}
		if (infoPtr->token < MULT && infoPtr->token != VALUE && infoPtr->token != END && infoPtr->token != CLOSE_PAREN) {
			goto syntaxError;
		}

		// At this point we've got two values and an operator.  Check to make sure that the particular data types are appropriate
		// for the particular operator, and perform type conversion if necessary.
		switch (operator_) {
			// For the operators below, no strings are allowed and ints get converted to floats if necessary.
		case MULT: case DIVIDE: case PLUS: case MINUS:
			if ((valuePtr->type == TYPE_STRING)
				|| (value2.type == TYPE_STRING)) {
				badType = TYPE_STRING;
				goto illegalType;
			}
			if (valuePtr->type == TYPE_DOUBLE) {
				if (value2.type == TYPE_INT) {
					value2.doubleValue = value2.intValue;
					value2.type = TYPE_DOUBLE;
				}
			}
			else if (value2.type == TYPE_DOUBLE) {
				if (valuePtr->type == TYPE_INT) {
					valuePtr->doubleValue = valuePtr->intValue;
					valuePtr->type = TYPE_DOUBLE;
				}
			}
			break;
			// For the operators below, only integers are allowed.
		case MOD: case LEFT_SHIFT: case RIGHT_SHIFT: case BIT_AND: case BIT_XOR: case BIT_OR:
			if (valuePtr->type != TYPE_INT) {
				badType = valuePtr->type;
				goto illegalType;
			}
			else if (value2.type != TYPE_INT) {
				badType = value2.type;
				goto illegalType;
			}
			break;
			// For the operators below, any type is allowed but the two operands must have the same type.  Convert integers to floats and either to strings, if necessary.
		case LESS: case GREATER: case LEQ: case GEQ: case EQUAL: case NEQ:
			if (valuePtr->type == TYPE_STRING) {
				if (value2.type != TYPE_STRING) {
					ExprMakeString(&value2);
				}
			}
			else if (value2.type == TYPE_STRING) {
				if (valuePtr->type != TYPE_STRING) {
					ExprMakeString(valuePtr);
				}
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				if (value2.type == TYPE_INT) {
					value2.doubleValue = value2.intValue;
					value2.type = TYPE_DOUBLE;
				}
			}
			else if (value2.type == TYPE_DOUBLE) {
				if (valuePtr->type == TYPE_INT) {
					valuePtr->doubleValue = valuePtr->intValue;
					valuePtr->type = TYPE_DOUBLE;
				}
			}
			break;
			// For the operators below, no strings are allowed, but no int->double conversions are performed.
		case AND: case OR:
			if (valuePtr->type == TYPE_STRING) {
				badType = valuePtr->type;
				goto illegalType;
			}
			if (value2.type == TYPE_STRING) {
				badType = value2.type;
				goto illegalType;
			}
			break;
			// For the operators below, type and conversions are irrelevant:  they're handled elsewhere.
		case QUESTY: case COLON:
			break;
			// Any other operator is an error.
		default:
			interp->result = (char *)"unknown operator in expression";
			result = TCL_ERROR;
			goto done;
		}

		// If necessary, convert one of the operands to the type of the other.  If the operands are incompatible with
		// the operator (e.g. "+" on strings) then return an error.
		switch (operator_) {
		case MULT:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue *= value2.intValue;
			}
			else {
				valuePtr->doubleValue *= value2.doubleValue;
			}
			break;
		case DIVIDE:
			if (valuePtr->type == TYPE_INT) {
				if (value2.intValue == 0) {
				divideByZero:
					interp->result = (char *)"divide by zero";
					result = TCL_ERROR;
					goto done;
				}
				valuePtr->intValue /= value2.intValue;
			}
			else {
				if (value2.doubleValue == 0.0) {
					goto divideByZero;
				}
				valuePtr->doubleValue /= value2.doubleValue;
			}
			break;
		case MOD:
			if (value2.intValue == 0) {
				goto divideByZero;
			}
			valuePtr->intValue %= value2.intValue;
			break;
		case PLUS:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue += value2.intValue;
			}
			else {
				valuePtr->doubleValue += value2.doubleValue;
			}
			break;
		case MINUS:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue -= value2.intValue;
			}
			else {
				valuePtr->doubleValue -= value2.doubleValue;
			}
			break;
		case LEFT_SHIFT:
			valuePtr->intValue <<= value2.intValue;
			break;
		case RIGHT_SHIFT:
			// The following code is a bit tricky:  it ensures that right shifts propagate the sign bit even on machines where ">>" won't do it by default.
			if (valuePtr->intValue < 0) {
				valuePtr->intValue = ~((~valuePtr->intValue) >> value2.intValue);
			}
			else {
				valuePtr->intValue >>= value2.intValue;
			}
			break;
		case LESS:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue = (valuePtr->intValue < value2.intValue);
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue < value2.doubleValue);
			}
			else {
				valuePtr->intValue = (strcmp(valuePtr->pv.buffer, value2.pv.buffer) < 0);
			}
			valuePtr->type = TYPE_INT;
			break;
		case GREATER:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue = (valuePtr->intValue > value2.intValue);
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue > value2.doubleValue);
			}
			else {
				valuePtr->intValue = (strcmp(valuePtr->pv.buffer, value2.pv.buffer) > 0);
			}
			valuePtr->type = TYPE_INT;
			break;
		case LEQ:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue = (valuePtr->intValue <= value2.intValue);
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue <= value2.doubleValue);
			}
			else {
				valuePtr->intValue = (strcmp(valuePtr->pv.buffer, value2.pv.buffer) <= 0);
			}
			valuePtr->type = TYPE_INT;
			break;
		case GEQ:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue = (valuePtr->intValue >= value2.intValue);
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue >= value2.doubleValue);
			}
			else {
				valuePtr->intValue = (strcmp(valuePtr->pv.buffer, value2.pv.buffer) >= 0);
			}
			valuePtr->type = TYPE_INT;
			break;
		case EQUAL:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue = (valuePtr->intValue == value2.intValue);
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue == value2.doubleValue);
			}
			else {
				valuePtr->intValue = (strcmp(valuePtr->pv.buffer, value2.pv.buffer) == 0);
			}
			valuePtr->type = TYPE_INT;
			break;
		case NEQ:
			if (valuePtr->type == TYPE_INT) {
				valuePtr->intValue = (valuePtr->intValue != value2.intValue);
			}
			else if (valuePtr->type == TYPE_DOUBLE) {
				valuePtr->intValue = (valuePtr->doubleValue != value2.doubleValue);
			}
			else {
				valuePtr->intValue = (strcmp(valuePtr->pv.buffer, value2.pv.buffer) != 0);
			}
			valuePtr->type = TYPE_INT;
			break;
		case BIT_AND:
			valuePtr->intValue &= value2.intValue;
			break;
		case BIT_XOR:
			valuePtr->intValue ^= value2.intValue;
			break;
		case BIT_OR:
			valuePtr->intValue |= value2.intValue;
			break;
			// For AND and OR, we know that the first value has already been converted to an integer.  Thus we need only consider
			// the possibility of int vs. double for the second value.
		case AND:
			if (value2.type == TYPE_DOUBLE) {
				value2.intValue = (value2.doubleValue != 0);
				value2.type = TYPE_INT;
			}
			valuePtr->intValue = (valuePtr->intValue && value2.intValue);
			break;
		case OR:
			if (value2.type == TYPE_DOUBLE) {
				value2.intValue = (value2.doubleValue != 0);
				value2.type = TYPE_INT;
			}
			valuePtr->intValue = (valuePtr->intValue || value2.intValue);
			break;
		case COLON:
			interp->result = (char *)"can't have : operator without ? first";
			result = TCL_ERROR;
			goto done;
		}
	}

done:
	if (value2.pv.buffer != value2.staticSpace) {
		_freeFast(value2.pv.buffer);
	}
	return result;

syntaxError:
	Tcl_ResetResult(interp);
	Tcl_AppendResult(interp, "syntax error in expression \"", infoPtr->originalExpr, "\"", (char *)NULL);
	result = TCL_ERROR;
	goto done;

illegalType:
	Tcl_AppendResult(interp, "can't use ", (badType == TYPE_DOUBLE ? "floating-point value" : "non-numeric string"), " as operand of \"", _operatorStrings[operator_], "\"", (char *)NULL);
	result = TCL_ERROR;
	goto done;
}

/*
*--------------------------------------------------------------
*
* ExprMakeString --
*	Convert a value from int or double representation to a string.
*
* Results:
*	The information at *valuePtr gets converted to string format, if it wasn't that way already.
*
* Side effects:
*	None.
*
*--------------------------------------------------------------
*/
static __device__ void ExprMakeString(register Value *valuePtr) {
	int shortfall = 150 - (int)(valuePtr->pv.end - valuePtr->pv.buffer);
	if (shortfall > 0) {
		(*valuePtr->pv.expandProc)(&valuePtr->pv, shortfall);
	}
	if (valuePtr->type == TYPE_INT) {
		sprintf(valuePtr->pv.buffer, "%ld", valuePtr->intValue);
	}
	else if (valuePtr->type == TYPE_DOUBLE) {
		sprintf(valuePtr->pv.buffer, "%g", valuePtr->doubleValue);
	}
	valuePtr->type = TYPE_STRING;
}

/*
*--------------------------------------------------------------
*
* ExprTopLevel --
*	This procedure provides top-level functionality shared by procedures like Tcl_ExprInt, Tcl_ExprDouble, etc.
*
* Results:
*	The result is a standard Tcl return value.  If an error occurs then an error message is left in interp->result.
*	The value of the expression is returned in *valuePtr, in whatever form it ends up in (could be string or integer
*	or double).  Caller may need to convert result.  Caller is also responsible for freeing string memory in *valuePtr,
*	if any was allocated.
*
* Side effects:
*	None.
*
*--------------------------------------------------------------
*/
static __device__ int ExprTopLevel(Tcl_Interp *interp, char *string, Value *valuePtr) {
	ExprInfo info;
	info.originalExpr = string;
	info.expr = string;
	valuePtr->pv.buffer = valuePtr->pv.next = valuePtr->staticSpace;
	valuePtr->pv.end = valuePtr->pv.buffer + STATIC_STRING_SPACE - 1;
	valuePtr->pv.expandProc = TclExpandParseValue;
	valuePtr->pv.clientData = (ClientData)NULL;

	int result = ExprGetValue(interp, &info, -1, valuePtr);
	if (result != TCL_OK) {
		return result;
	}
	if (info.token != END) {
		Tcl_AppendResult(interp, "syntax error in expression \"", string, "\"", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

/*
*--------------------------------------------------------------
*
* Tcl_ExprLong, Tcl_ExprDouble, Tcl_ExprBoolean --
*	Procedures to evaluate an expression and return its value in a particular form.
*
* Results:
*	Each of the procedures below returns a standard Tcl result. If an error occurs then an error message is left in
*	interp->result.  Otherwise the value of the expression, in the appropriate form, is stored at *resultPtr.  If
*	the expression had a result that was incompatible with the desired form then an error is returned.
*
* Side effects:
*	None.
*
*--------------------------------------------------------------
*/
__device__ int Tcl_ExprLong(Tcl_Interp *interp, char *string, long *ptr) {
	Value value;
	int result = ExprTopLevel(interp, string, &value);
	if (result == TCL_OK) {
		if (value.type == TYPE_INT) {
			*ptr = value.intValue;
		}
		else if (value.type == TYPE_DOUBLE) {
			*ptr = (long)value.doubleValue;
		}
		else {
			interp->result = (char *)"expression didn't have numeric value";
			result = TCL_ERROR;
		}
	}
	if (value.pv.buffer != value.staticSpace) {
		_freeFast(value.pv.buffer);
	}
	return result;
}

__device__ int Tcl_ExprDouble(Tcl_Interp *interp, char *string, double *ptr) {
	Value value;
	int result = ExprTopLevel(interp, string, &value);
	if (result == TCL_OK) {
		if (value.type == TYPE_INT) {
			*ptr = value.intValue;
		}
		else if (value.type == TYPE_DOUBLE) {
			*ptr = value.doubleValue;
		}
		else {
			interp->result = (char *)"expression didn't have numeric value";
			result = TCL_ERROR;
		}
	}
	if (value.pv.buffer != value.staticSpace) {
		_freeFast(value.pv.buffer);
	}
	return result;
}

__device__ int Tcl_ExprBoolean(Tcl_Interp *interp, char *string, int *ptr) {
	Value value;
	int result = ExprTopLevel(interp, string, &value);
	if (result == TCL_OK) {
		if (value.type == TYPE_INT) {
			*ptr = (value.intValue != 0);
		}
		else if (value.type == TYPE_DOUBLE) {
			*ptr = (value.doubleValue != 0.0);
		}
		else {
			interp->result = (char *)"expression didn't have numeric value";
			result = TCL_ERROR;
		}
	}
	if (value.pv.buffer != value.staticSpace) {
		_freeFast(value.pv.buffer);
	}
	return result;
}

/*
*--------------------------------------------------------------
*
* Tcl_ExprString --
*	Evaluate an expression and return its value in string form.
*
* Results:
*	A standard Tcl result.  If the result is TCL_OK, then the interpreter's result is set to the string value of the
*	expression.  If the result is TCL_OK, then interp->result contains an error message.
*
* Side effects:
*	None.
*
*--------------------------------------------------------------
*/
__device__ int Tcl_ExprString(Tcl_Interp *interp, char *string) {
	Value value;
	int result = ExprTopLevel(interp, string, &value);
	if (result == TCL_OK) {
		if (value.type == TYPE_INT) {
			sprintf(interp->result, "%ld", value.intValue);
		}
		else if (value.type == TYPE_DOUBLE) {
			sprintf(interp->result, "%g", value.doubleValue);
		}
		else {
			if (value.pv.buffer != value.staticSpace) {
				interp->result = value.pv.buffer;
				interp->freeProc = (Tcl_FreeProc *)free;
				value.pv.buffer = value.staticSpace;
			}
			else {
				Tcl_SetResult(interp, value.pv.buffer, TCL_VOLATILE);
			}
		}
	}
	if (value.pv.buffer != value.staticSpace) {
		_freeFast(value.pv.buffer);
	}
	return result;
}
