#include <stdiocu.h>
#include <stdlibcu.h>
#include <stringcu.h>
#include <ctypecu.h>
#include <regexcu.h>

// No utf-8 support. 1 byte = 1 char
#define utf8_strlen(S, B) ((B) < 0 ? strlen(S) : (B))
#define utf8_tounicode(S, CP) (*(CP) = (unsigned char)*(S), 1)
#define utf8_getchars(CP, C) (*(CP) = (C), 1)
#define utf8_upper(C) toupper(C)
#define utf8_title(C) toupper(C)
#define utf8_lower(C) tolower(C)
#define utf8_index(C, I) (I)
#define utf8_charlen(C) 1
#define utf8_prev_len(S, L) 1

#define REG_MAX_PAREN 100

// Structure for regexp "program".  This is essentially a linear encoding of a nondeterministic finite-state machine (aka syntax charts or
// "railroad normal form" in parsing technology).  Each node is an opcode plus a "next" pointer, possibly plus an operand.  "Next" pointers of
// all nodes except BRANCH implement concatenation; a "next" pointer with a BRANCH on both ends of it is connecting two alternatives.  (Here we
// have one of the subtle syntax dependencies:  an individual BRANCH (as opposed to a collection of them) is never concatenated with anything
// because of operator precedence.)  The operand of some types of node is a literal string; for others, it is a node leading into a sub-FSM.  In
// particular, the operand of a BRANCH node is the first node of the branch.
// (NB this is *not* a tree structure:  the tail of the branch connects to the thing following the set of BRANCHes.)  The opcodes are:

#define	OP_END		0		// no	End of program
#define	OP_BOL		1		// no	Match "" at beginning of line
#define	OP_EOL		2		// no	Match "" at end of line
#define	OP_ANY		3		// no	Match any one character
#define	OP_ANYOF	4		// str	Match any character in this string
#define	OP_ANYBUT	5		// str	Match any character not in this string
#define	OP_BRANCH	6		// node	Match this alternative, or the next
#define	OP_BACK		7		// no	Match "", "next" ptr points backward
#define	OP_EXACTLY	8		// str	Match this string
#define	OP_NOTHING	9		// no	Match empty string
#define	OP_REP		10		// max,min	Match this (simple) thing [min,max] times
#define	OP_REPMIN	11		// max,min	Match this (simple) thing [min,max] times, minimal match
#define	OP_REPX		12		// max,min	Match this (complex) thing [min,max] times
#define	OP_REPXMIN	13		// max,min	Match this (complex) thing [min,max] times, minimal match
#define	OP_WORDA	15		// no	Match "" at wordchar, where prev is nonword
#define	OP_WORDZ	16		// no	Match "" at nonwordchar, where prev is word
#define	OP_OPENNC 	1000	// no	Non-capturing parentheses - must be OPEN-1
#define	OP_OPEN   	1001	// no	Mark this point in input as start of #n
//	OP_OPEN+1 is number 1, etc. must not be any other opts between OPEN and CLOSE
#define	OP_CLOSENC	2000 	// no	Non-capturing parentheses - must be CLOSE-1
#define	OP_CLOSE	2001 	// no	Analogous to OPEN
#define	OP_CLOSE_END (OP_CLOSE+REG_MAX_PAREN)

// The first word of the regexp internal "program" is actually this magic number; the start node begins in the second word.
#define	REG_MAGIC	0xFADED00D

// Opcode notes:
//
// BRANCH	The set of branches constituting a single choice are hooked together with their "next" pointers, since precedence prevents
//		anything being concatenated to any individual branch.  The "next" pointer of the last BRANCH in a choice points to the
//		thing following the whole choice.  This is also where the final "next" pointer of each individual branch points; each
//		branch starts with the operand node of a BRANCH node.
//
// BACK		Normal "next" pointers all implicitly point forward; BACK exists to make loop structures possible.
//
// REP,REPX	Repeated matches ('?', '*', '+' and {min,max}) are implemented as either simple repeats (REP) or complex repeats (REPX).
//              These opcodes include a "min" and "max" count after the opcode.
//		This is followed by a fourth "current count" word that is only used by REPX, as it implements a recursive match.
//		REPMIN and REPXMIN are identical except they implement minimal repeats.
//
// OPEN,CLOSE	...are numbered at compile time.

// A node is one word of opcode followed by one word of "next" pointer. The "next" pointer value is a positive offset from the opcode of the node
// containing it. An operand, if any, simply follows the node.  (Note that much of the code generation knows about this implicit relationship.)
#define	OP(preg, p)	(preg->program[p])
#define	NEXT(preg, p) (preg->program[p + 1])
#define	OPERAND(p) ((p) + 2)

// Utility definitions.
#define	FAIL(R,M) { (R)->err = (M); return (M); }
#define	ISMULT(c) ((c) == '*' || (c) == '+' || (c) == '?' || (c) == '{')
#define	META "^$.[()|?{+*"

// Flags to be passed up and down.
#define	HASWIDTH 1	// Known never to match null string
#define	SIMPLE 2	// Simple enough to be STAR/PLUS operand
#define	SPSTART 4	// Starts with * or +
#define	WORST 0		// Worst case
#define MAX_REP_COUNT 1000000

// Forward declarations for regcomp()'s friends.
static __device__ int reg(regex_t *preg, int paren, int *flagp);
static __device__ int regpiece(regex_t *preg, int *flagp);
static __device__ int regbranch(regex_t *preg, int *flagp);
static __device__ int regatom(regex_t *preg, int *flagp);
static __device__ int regnode(regex_t *preg, int op);
static __device__ int regnext(regex_t *preg, int p);
static __device__ void regc(regex_t *preg, int b);
static __device__ int reginsert(regex_t *preg, int op, int size, int opnd);
static __device__ void regtail(regex_t *preg, int p, int val);
static __device__ void regoptail(regex_t *preg, int p, int val);
static __device__ int regopsize(regex_t *preg, int p);
static __device__ int reg_range_find(const int *string, int c);
static __device__ const char *str_find(const char *string, int c, int nocase);
static __device__ int prefix_cmp(const int *prog, int proglen, const char *string, int nocase);

#ifdef _DEBUGx
static __device__ int regnarrate = 0;
static __device__ void regdump(regex_t *preg);
static __device__ const char *regprop(int op);
#endif

// Returns the length of the null-terminated integer sequence.
static __forceinline__ __device__ int _strlenint(const int *seq) {
	int n = 0;
	while (*seq++) { n++; }
	return n;
}

// regcomp - compile a regular expression into internal code
// We can't allocate space until we know how big the compiled form will be, but we can't compile it (and thus know how big it is) until we've got a
// place to put the code.  So we cheat:  we compile it twice, once with code generation turned off and size counting turned on, and once "for real".
// This also means that we don't allocate space until we are sure that the thing really will compile successfully, and we never have to move the
// code and thus invalidate pointers into it.  (Note that it has to be in one piece because free() must be able to free it all.)
//
// Beware that the optimization-preparation code in here knows about some of the structure of the compiled regexp.
__device__ int regcomp_(regex_t *preg, const char *exp, int cflags) {
#ifdef _DEBUGx
	printf("Compiling: '%s'\n", exp);
#endif
	memset(preg, 0, sizeof(*preg));
	if (!exp)
		FAIL(preg, REG_ERR_NULL_ARGUMENT);
	// First pass: determine size, legality.
	preg->cflags = cflags;
	preg->regparse = exp;
	// Allocate space
	preg->proglen = (strlen(exp) + 1) * 5;
	preg->program = (int *)malloc(preg->proglen * sizeof(int));
	if (!preg->program)
		FAIL(preg, REG_ERR_NOMEM);
	// Note that since we store a magic value as the first item in the program, program offsets will never be 0
	regc(preg, REG_MAGIC);
	int flags;
	if (!reg(preg, 0, &flags))
		return preg->err;
	// Small enough for pointer-storage convention?
	if (preg->re_nsub >= REG_MAX_PAREN)
		FAIL(preg, REG_ERR_TOO_BIG);
	// Dig out information for optimizations
	preg->regstart = 0;	// Worst-case defaults
	preg->reganch = 0;
	preg->regmust = 0;
	preg->regmlen = 0;
	int scan = 1; // First BRANCH
	if (OP(preg, regnext(preg, scan)) == OP_END) { // Only one top-level choice
		scan = OPERAND(scan);
		// Starting-point info
		if (OP(preg, scan) == OP_EXACTLY)
			preg->regstart = preg->program[OPERAND(scan)];
		else if (OP(preg, scan) == OP_BOL)
			preg->reganch++;
		// If there's something expensive in the r.e., find the longest literal string that must appear and make it the
		// regmust.  Resolve ties in favor of later strings, since the regstart check works with the beginning of the r.e.
		// and avoiding duplication strengthens checking.  Not a strong reason, but sufficient in the absence of others.
		if (flags & SPSTART) {
			int longest = 0;
			int len = 0;
			for (; scan != 0; scan = regnext(preg, scan)) {
				if (OP(preg, scan) == OP_EXACTLY) {
					int plen = _strlenint(preg->program + OPERAND(scan));
					if (plen >= len) {
						longest = OPERAND(scan);
						len = plen;
					}
				}
			}
			preg->regmust = longest;
			preg->regmlen = len;
		}
	}
#ifdef _DEBUGx
	regdump(preg);
#endif
	return 0;
}

// reg - regular expression, i.e. main body or parenthesized thing
// Caller must absorb opening parenthesis.
// Combining parenthesis handling with the base level of regular expression is a trifle forced, but the need to tie the tails of the branches to what
// follows makes it hard to avoid.
static __device__ int reg(regex_t *preg, int paren, int *flagp) {
	*flagp = HASWIDTH;	// Tentatively

	// Make an OPEN node, if parenthesized
	int ret;
	int parno = 0;
	if (paren) {
		if (preg->regparse[0] == '?' && preg->regparse[1] == ':') { preg->regparse += 2; parno = -1; } // non-capturing paren
		else parno = ++preg->re_nsub;
		ret = regnode(preg, OP_OPEN + parno);
	}
	else
		ret = 0;

	// Pick up the branches, linking them togethe
	int flags;
	int br = regbranch(preg, &flags);
	if (!br)
		return 0;
	if (ret)
		regtail(preg, ret, br);	// OPEN -> first
	else
		ret = br;
	if (!(flags & HASWIDTH))
		*flagp &= ~HASWIDTH;
	*flagp |= flags & SPSTART;
	while (*preg->regparse == '|') {
		preg->regparse++;
		br = regbranch(preg, &flags);
		if (br == 0)
			return 0;
		regtail(preg, ret, br);	// BRANCH -> BRANCH
		if (!(flags & HASWIDTH))
			*flagp &= ~HASWIDTH;
		*flagp |= flags & SPSTART;
	}

	// Make a closing node, and hook it on the end
	int ender = regnode(preg, paren ? OP_CLOSE + parno : OP_END);
	regtail(preg, ret, ender);

	// Hook the tails of the branches to the closing node
	for (br = ret; br; br = regnext(preg, br))
		regoptail(preg, br, ender);

	// Check for proper termination
	if (paren && *preg->regparse++ != ')') { preg->err = REG_ERR_UNMATCHED_PAREN; return 0; }
	else if (!paren && *preg->regparse != '\0') {
		if (*preg->regparse == ')') { preg->err = REG_ERR_UNMATCHED_PAREN; return 0; }
		else { preg->err = REG_ERR_JUNK_ON_END; return 0; }
	}
	return ret;
}

// regbranch - one alternative of an | operator
// Implements the concatenation operator.
static __device__ int regbranch(regex_t *preg, int *flagp) {
	*flagp = WORST; // Tentatively
	int ret = regnode(preg, OP_BRANCH);
	int chain = 0;
	while (*preg->regparse != '\0' && *preg->regparse != ')' && *preg->regparse != '|') {
		int flags;
		int latest = regpiece(preg, &flags);
		if (latest == 0)
			return 0;
		*flagp |= flags & HASWIDTH;
		if (chain == 0) // First piece
			*flagp |= flags & SPSTART;
		else
			regtail(preg, chain, latest);
		chain = latest;
	}
	if (chain == 0)	// Loop ran zero times
		regnode(preg, OP_NOTHING);
	return ret;
}

// regpiece - something followed by possible [*+?]
// Note that the branching code sequences used for ? and the general cases of * and + are somewhat optimized:  they use the same NOTHING node as
// both the endmarker for their branch list and the body of the last branch. It might seem that this node could be dispensed with entirely, but the
// endmarker role is not redundant.
static __device__ int regpiece(regex_t *preg, int *flagp) {
	int flags;
	int ret = regatom(preg, &flags);
	if (ret == 0)
		return 0;

	char op = *preg->regparse;
	if (!ISMULT(op)) { *flagp = flags; return ret; }
	if (!(flags & HASWIDTH) && op != '?') { preg->err = REG_ERR_OPERAND_COULD_BE_EMPTY; return 0; }

	// Handle braces (counted repetition) by expansion
	int min, max;
	if (op == '{') {
		char *end;
		min = strtoul(preg->regparse + 1, &end, 10);
		if (end == preg->regparse + 1) { preg->err = REG_ERR_BAD_COUNT; return 0; }
		if (*end == '}')
			max = min;
		else {
			preg->regparse = end;
			max = strtoul(preg->regparse + 1, &end, 10);
			if (*end != '}') { preg->err = REG_ERR_UNMATCHED_BRACES; return 0; }
		}
		if (end == preg->regparse + 1)
			max = MAX_REP_COUNT;
		else if (max < min || max >= 100) { preg->err = REG_ERR_BAD_COUNT; return 0; }
		if (min >= 100) { preg->err = REG_ERR_BAD_COUNT; return 0; }

		preg->regparse = strchr(preg->regparse, '}');
	}
	else {
		min = op == '+';
		max = op == '?' ? 1 : MAX_REP_COUNT;
	}

	int next;
	if (preg->regparse[1] == '?') {
		preg->regparse++;
		next = reginsert(preg, flags & SIMPLE ? OP_REPMIN : OP_REPXMIN, 5, ret);
	}
	else
		next = reginsert(preg, flags & SIMPLE ? OP_REP : OP_REPX, 5, ret);
	preg->program[ret + 2] = max;
	preg->program[ret + 3] = min;
	preg->program[ret + 4] = 0;

	*flagp = min ? WORST | HASWIDTH : WORST | SPSTART;

	if (!(flags & SIMPLE)) {
		int back = regnode(preg, OP_BACK);
		regtail(preg, back, ret);
		regtail(preg, next, back);
	}

	preg->regparse++;
	if (ISMULT(*preg->regparse)) { preg->err = REG_ERR_NESTED_COUNT; return 0; }
	return ret;
}

// Add all characters in the inclusive range between lower and upper.
// Handles a swapped range (upper < lower).
static __device__ void reg_addrange(regex_t *preg, int lower, int upper) {
	if (lower > upper)
		reg_addrange(preg, upper, lower);
	// Add a range as length, start
	regc(preg, upper - lower + 1);
	regc(preg, lower);
}

// Add a null-terminated literal string as a set of ranges.
static __device__ void reg_addrange_str(regex_t *preg, const char *str) {
	while (*str) {
		reg_addrange(preg, *str, *str);
		str++;
	}
}

// Extracts the next unicode char from utf8.
// If 'upper' is set, converts the char to uppercase.
static __device__ int reg_utf8_tounicode_case(const char *s, int *uc, int upper) {
	int l = utf8_tounicode(s, uc);
	if (upper)
		*uc = utf8_upper(*uc);
	return l;
}

// Converts a hex digit to decimal.
// Returns -1 for an invalid hex digit.
static __device__ int hexdigitval(int c) {
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

// Parses up to 'n' hex digits at 's' and stores the result in *uc.
// Returns the number of hex digits parsed.
// If there are no hex digits, returns 0 and stores nothing.
static __device__ int parse_hex(const char *s, int n, int *uc) {
	int val = 0;
	int k;
	for (k = 0; k < n; k++) {
		int c = hexdigitval(*s++);
		if (c == -1)
			break;
		val = (val << 4) | c;
	}
	if (k)
		*uc = val;
	return k;
}

// Call for chars after a backlash to decode the escape sequence.
// Stores the result in *ch.
// Returns the number of bytes consumed.
static __device__ int reg_decode_escape(const char *s, int *ch) {
	int n;
	const char *s0 = s;
	*ch = *s++;
	switch (*ch) {
	case 'b': *ch = '\b'; break;
	case 'e': *ch = 27; break;
	case 'f': *ch = '\f'; break;
	case 'n': *ch = '\n'; break;
	case 'r': *ch = '\r'; break;
	case 't': *ch = '\t'; break;
	case 'v': *ch = '\v'; break;
	case 'u':
		if (*s == '{') { // Expect \u{NNNN}
			n = parse_hex(s + 1, 6, ch);
			if (n > 0 && s[n + 1] == '}' && *ch >= 0 && *ch <= 0x1fffff)
				s += n + 2;
			else
				*ch = 'u'; // Invalid, so just treat as an escaped 'u'
		}
		else if ((n = parse_hex(s, 4, ch)) > 0)
			s += n;
		break;
	case 'U':
		if ((n = parse_hex(s, 8, ch)) > 0)
			s += n;
		break;
	case 'x':
		if ((n = parse_hex(s, 2, ch)) > 0)
			s += n;
		break;
	case '\0':
		s--;
		*ch = '\\';
		break;
	}
	return (int)(s - s0);
}

// regatom - the lowest level
// Optimization:  gobbles an entire sequence of ordinary characters so that it can turn them into a single node, which is smaller to store and
// faster to run.  Backslashed characters are exceptions, each becoming a separate node; the code is simpler that way and it's not worth fixing.
static __device__ int regatom(regex_t *preg, int *flagp) {
	int ret;
	int flags;
	int nocase = (preg->cflags & REG_ICASE);
	int ch;
	int n = reg_utf8_tounicode_case(preg->regparse, &ch, nocase);
	*flagp = WORST; // Tentatively
	preg->regparse += n;
	switch (ch) { // FIXME: these chars only have meaning at beg/end of pat?
	case '^': ret = regnode(preg, OP_BOL); break;
	case '$': ret = regnode(preg, OP_EOL); break;
	case '.': ret = regnode(preg, OP_ANY); *flagp |= HASWIDTH | SIMPLE; break;
	case '[': {
		const char *pattern = preg->regparse;
		if (*pattern == '^') { ret = regnode(preg, OP_ANYBUT); pattern++; } // Complement of range
		else ret = regnode(preg, OP_ANYOF);

		// Special case. If the first char is ']' or '-', it is part of the set
		if (*pattern == ']' || *pattern == '-') { reg_addrange(preg, *pattern, *pattern); pattern++; }

		while (*pattern && *pattern != ']') {
			// Is this a range? a-z
			int start;
			int end;
			pattern += reg_utf8_tounicode_case(pattern, &start, nocase);
			if (start == '\\') {
				pattern += reg_decode_escape(pattern, &start);
				if (start == 0) { preg->err = REG_ERR_NULL_CHAR; return 0; }
			}
			if (pattern[0] == '-' && pattern[1] && pattern[1] != ']') {
				// skip '-'
				pattern += utf8_tounicode(pattern, &end);
				pattern += reg_utf8_tounicode_case(pattern, &end, nocase);
				if (end == '\\')
				{
					pattern += reg_decode_escape(pattern, &end);
					if (end == 0) { preg->err = REG_ERR_NULL_CHAR; return 0; }
				}
				reg_addrange(preg, start, end);
				continue;
			}
			if (start == '[') {
				if (!strncmp(pattern, ":alpha:]", 8)) {
					if (!(preg->cflags & REG_ICASE))
						reg_addrange(preg, 'a', 'z');
					reg_addrange(preg, 'A', 'Z');
					pattern += 8;
					continue;
				}
				if (!strncmp(pattern, ":alnum:]", 8)) {
					if ((preg->cflags & REG_ICASE) == 0)
						reg_addrange(preg, 'a', 'z');
					reg_addrange(preg, 'A', 'Z');
					reg_addrange(preg, '0', '9');
					pattern += 8;
					continue;
				}
				if (!strncmp(pattern, ":space:]", 8)) {
					reg_addrange_str(preg, " \t\r\n\f\v");
					pattern += 8;
					continue;
				}
			}
			// Not a range, so just add the char
			reg_addrange(preg, start, start);
		}
		regc(preg, '\0');

		if (*pattern)
			pattern++;
		preg->regparse = pattern;

		*flagp |= HASWIDTH | SIMPLE;
	} break;
	case '(':
		ret = reg(preg, 1, &flags);
		if (ret == 0)
			return 0;
		*flagp |= flags & (HASWIDTH | SPSTART);
		break;
	case '\0':
	case '|':
	case ')':
		preg->err = REG_ERR_INTERNAL;
		return 0; // Supposed to be caught earlier
	case '?':
	case '+':
	case '*':
	case '{':
		preg->err = REG_ERR_COUNT_FOLLOWS_NOTHING;
		return 0;
	case '\\':
		switch (*preg->regparse++) {
		case '\0': preg->err = REG_ERR_TRAILING_BACKSLASH; return 0;
		case '<':
		case 'm': ret = regnode(preg, OP_WORDA); break;
		case '>':
		case 'M': ret = regnode(preg, OP_WORDZ); break;
		case 'd':
			ret = regnode(preg, OP_ANYOF);
			reg_addrange(preg, '0', '9');
			regc(preg, '\0');
			*flagp |= HASWIDTH | SIMPLE;
			break;
		case 'w':
			ret = regnode(preg, OP_ANYOF);
			if (!(preg->cflags & REG_ICASE))
				reg_addrange(preg, 'a', 'z');
			reg_addrange(preg, 'A', 'Z');
			reg_addrange(preg, '0', '9');
			reg_addrange(preg, '_', '_');
			regc(preg, '\0');
			*flagp |= HASWIDTH | SIMPLE;
			break;
		case 's':
			ret = regnode(preg, OP_ANYOF);
			reg_addrange_str(preg, " \t\r\n\f\v");
			regc(preg, '\0');
			*flagp |= HASWIDTH | SIMPLE;
			break;
			// FIXME: Someday handle \1, \2, ...
		default:
			// Handle general quoted chars in exact-match routine. Back up to include the backslash
			preg->regparse--;
			goto de_fault;
		}
		break;
	de_fault:
	default: {
		// Encode a string of characters to be matched exactly.
		int added = 0;
		// Back up to pick up the first char of interest
		preg->regparse -= n;
		ret = regnode(preg, OP_EXACTLY);

		// Note that a META operator such as ? or * consumes the preceding char.
		// Thus we must be careful to look ahead by 2 and add the last char as it's own EXACTLY if necessary
		// Until end of string or a META char is reached
		while (*preg->regparse && strchr(META, *preg->regparse) == NULL) {
			n = reg_utf8_tounicode_case(preg->regparse, &ch, (preg->cflags & REG_ICASE));
			if (ch == '\\' && preg->regparse[n]) {
				// Non-trailing backslash. Is this a special escape, or a regular escape?
				if (strchr("<>mMwds", preg->regparse[n]))
					break; // A special escape. All done with EXACTLY
				// Decode it. Note that we add the length for the escape sequence to the length for the backlash so we can skip
				// the entire sequence, or not as required.
				n += reg_decode_escape(preg->regparse + n, &ch);
				if (ch == 0) { preg->err = REG_ERR_NULL_CHAR; return 0; }
			}

			// Now we have one char 'ch' of length 'n'. Check to see if the following char is a MULT
			if (ISMULT(preg->regparse[n])) {
				// Yes. But do we already have some EXACTLY chars?
				if (added)
					break; // Yes, so return what we have and pick up the current char next time around
				// No, so add this single char and finish
				regc(preg, ch);
				added++;
				preg->regparse += n;
				break;
			}

			// No, so just add this char normally
			regc(preg, ch);
			added++;
			preg->regparse += n;
		}
		regc(preg, '\0');

		*flagp |= HASWIDTH;
		if (added == 1)
			*flagp |= SIMPLE;
		break; }
	}
	return ret;
}

static __device__ void reg_grow(regex_t *preg, int n) {
	if (preg->p + n >= preg->proglen) {
		preg->proglen = (preg->p + n) * 2;
		preg->program = (int *)realloc(preg->program, preg->proglen * sizeof(int));
	}
}

// regnode - emit a node
// Location
static __device__ int regnode(regex_t *preg, int op) {
	reg_grow(preg, 2);
	// The OP followed by a next pointer
	preg->program[preg->p++] = op;
	preg->program[preg->p++] = 0;
	// Return the start of the node
	return preg->p - 2;
}

// regc - emit (if appropriate) a byte of code
static __device__ void regc(regex_t *preg, int b) {
	reg_grow(preg, 1);
	preg->program[preg->p++] = b;
}

// reginsert - insert an operator in front of already-emitted operand
// Means relocating the operand. Returns the new location of the original operand.
static __device__ int reginsert(regex_t *preg, int op, int size, int opnd) {
	reg_grow(preg, size);
	// Move everything from opnd up
	memmove(preg->program + opnd + size, preg->program + opnd, sizeof(int) * (preg->p - opnd));
	// Zero out the new space
	memset(preg->program + opnd, 0, sizeof(int) * size);
	preg->program[opnd] = op;
	preg->p += size;
	return opnd + size;
}

// regtail - set the next-pointer at the end of a node chain
static __device__ void regtail(regex_t *preg, int p, int val) {
	// Find last node
	int scan = p;
	for (;;) {
		int temp = regnext(preg, scan);
		if (temp == 0)
			break;
		scan = temp;
	}
	int offset = OP(preg, scan) == OP_BACK ? scan - val : val - scan;
	preg->program[scan + 1] = offset;
}

// regoptail - regtail on operand of first argument; nop if operandless
static __device__ void regoptail(regex_t *preg, int p, int val) {
	// "Operandless" and "op != BRANCH" are synonymous in practice
	if (p != 0 && OP(preg, p) == OP_BRANCH)
		regtail(preg, OPERAND(p), val);
}

// regexec and friends
// Forwards.
static __device__ int regtry(regex_t *preg, const char *string);
static __device__ int regmatch(regex_t *preg, int prog);
static __device__ int regrepeat(regex_t *preg, int p, int max);

// regexec - match a regexp against a string
__device__ int regexec_(regex_t *preg, const char *string, size_t nmatch, regmatch_t pmatch[], int eflags) {
	// Be paranoid...
	if (preg == NULL || preg->program == NULL || string == NULL)
		return REG_ERR_NULL_ARGUMENT;
	// Check validity of program
	if (*preg->program != REG_MAGIC)
		return REG_ERR_CORRUPTED;

#ifdef _DEBUGx
	printf("regexec: %s\n", string);
	regdump(preg);
#endif
	preg->eflags = eflags;
	preg->pmatch = pmatch;
	preg->nmatch = (int)nmatch;
	preg->start = string; // All offsets are computed from here

	// Must clear out the embedded repeat counts of REPX and REPXMIN opcodes
	for (int scan = OPERAND(1); scan != 0; scan += regopsize(preg, scan)) {
		int op = OP(preg, scan);
		if (op == OP_END)
			break;
		if (op == OP_REPX || op == OP_REPXMIN)
			preg->program[scan + 4] = 0;
	}

	// If there is a "must appear" string, look for it
	const char *s;
	if (preg->regmust != 0) {
		s = string;
		while ((s = str_find(s, preg->program[preg->regmust], preg->cflags & REG_ICASE)) != NULL) {
			if (prefix_cmp(preg->program + preg->regmust, preg->regmlen, s, preg->cflags & REG_ICASE) >= 0)
				break;
			s++;
		}
		if (!s)
			return REG_NOMATCH; // Not present
	}

	// Mark beginning of line for ^
	preg->regbol = string;

	// Simplest case:  anchored match need be tried only once (maybe per line)
	if (preg->reganch) {
		if (eflags & REG_NOTBOL)
			goto nextline; // This is an anchored search, but not an BOL, so possibly skip to the next line
		while (1) {
			if (regtry(preg, string))
				return REG_NOERROR;
			if (*string) {
			nextline:
				if (preg->cflags & REG_NEWLINE) {
					// Try the next anchor?
					string = strchr(string, '\n');
					if (string) {
						preg->regbol = ++string;
						continue;
					}
				}
			}
			return REG_NOMATCH;
		}
	}

	// Messy cases:  unanchored match
	s = string;
	if (preg->regstart != '\0') // We know what char it must start with
		while ((s = str_find(s, preg->regstart, preg->cflags & REG_ICASE)) != NULL) {
			if (regtry(preg, s))
				return REG_NOERROR;
			s++;
		}
	else // We don't -- general case
		while (1) {
			if (regtry(preg, s))
				return REG_NOERROR;
			int c; UNUSED_SYMBOL(c);
			if (*s == '\0')
				break;
			else
				s += utf8_tounicode(s, &c);
		}
	return REG_NOMATCH; // Failure
}

// regtry - try match at specific point
// 0 failure, 1 success
static __device__ int regtry(regex_t *preg, const char *string) {
	preg->reginput = string;
	for (int i = 0; i < preg->nmatch; i++) {
		preg->pmatch[i].rm_so = -1;
		preg->pmatch[i].rm_eo = -1;
	}
	if (regmatch(preg, 1)) {
		preg->pmatch[0].rm_so = (int)(string - preg->start);
		preg->pmatch[0].rm_eo = (int)(preg->reginput - preg->start);
		return 1;
	}
	return 0;
}

// Returns bytes matched if 'pattern' is a prefix of 'string'.
// If 'nocase' is non-zero, does a case-insensitive match.
// Returns -1 on not found.
static __device__ int prefix_cmp(const int *prog, int proglen, const char *string, int nocase) {
	const char *s = string;
	while (proglen && *s) {
		int ch;
		int n = reg_utf8_tounicode_case(s, &ch, nocase);
		if (ch != *prog)
			return -1;
		prog++;
		s += n;
		proglen--;
	}
	return !proglen ? (int)(s - string) : -1;
}

// Searchs for 'c' in the range 'range'.
// Returns 1 if found, or 0 if not.
static __device__ int reg_range_find(const int *range, int c) {
	while (*range) {
		// printf("Checking %d in range [%d,%d]\n", c, range[1], (range[0] + range[1] - 1));
		if (c >= range[1] && c <= (range[0] + range[1] - 1))
			return 1;
		range += 2;
	}
	return 0;
}

// Search for the character 'c' in the utf-8 string 'string'.
// If 'nocase' is set, the 'string' is assumed to be uppercase and 'c' is converted to uppercase before matching.
// Returns the byte position in the string where the 'c' was found, or NULL if not found.
static __device__ const char *str_find(const char *string, int c, int nocase) {
	if (nocase) // The "string" should already be converted to uppercase
		c = utf8_upper(c);
	while (*string) {
		int ch;
		int n = reg_utf8_tounicode_case(string, &ch, nocase);
		if (c == ch)
			return string;
		string += n;
	}
	return nullptr;
}

// Returns true if 'ch' is an end-of-line char.
// In REG_NEWLINE mode, \n is considered EOL in addition to \0
static __device__ int reg_iseol(regex_t *preg, int ch) {
	return preg->cflags & REG_NEWLINE ? ch == '\0' || ch == '\n' : ch == '\0';
}

static __device__ int regmatchsimplerepeat(regex_t *preg, int scan, int matchmin) {
	int max = preg->program[scan + 2];
	int min = preg->program[scan + 3];
	int next = regnext(preg, scan);

	// Lookahead to avoid useless match attempts when we know what character comes next.
	int nextch = '\0';
	if (OP(preg, next) == OP_EXACTLY)
		nextch = preg->program[OPERAND(next)];
	const char *save = preg->reginput;
	int no = regrepeat(preg, scan + 5, max);
	if (no < min)
		return 0;
	if (matchmin) {
		// from min up to no
		max = no;
		no = min;
	}
	// else from no down to min
	while (1) {
		if (matchmin) {
			if (no > max)
				break;
		}
		else {
			if (no < min)
				break;
		}
		preg->reginput = save + utf8_index(save, no);
		int c;
		reg_utf8_tounicode_case(preg->reginput, &c, (preg->cflags & REG_ICASE));
		// If it could work, try it
		if (reg_iseol(preg, nextch) || c == nextch)
			if (regmatch(preg, next))
				return 1;
		if (matchmin) // Couldn't or didn't, add one more
			no++;
		else // Couldn't or didn't -- back up
			no--;
	}
	return 0;
}

static __device__ int regmatchrepeat(regex_t *preg, int scan, int matchmin) {
	int *scanpt = preg->program + scan;
	int max = scanpt[2];
	int min = scanpt[3];
	// Have we reached min?
	if (scanpt[4] < min) {
		// No, so get another one
		scanpt[4]++;
		if (regmatch(preg, scan + 5))
			return 1;
		scanpt[4]--;
		return 0;
	}
	if (scanpt[4] > max)
		return 0;
	if (matchmin) {
		// minimal, so try other branch first
		if (regmatch(preg, regnext(preg, scan)))
			return 1;
		// No, so try one more
		scanpt[4]++;
		if (regmatch(preg, scan + 5))
			return 1;
		scanpt[4]--;
		return 0;
	}
	if (scanpt[4] < max) {
		// maximal, so try this branch again
		scanpt[4]++;
		if (regmatch(preg, scan + 5))
			return 1;
		scanpt[4]--;
	}
	// At this point we are at max with no match. Try the other branch
	return regmatch(preg, regnext(preg, scan));
}

// regmatch - main matching routine
// Conceptually the strategy is simple:  check to see whether the current node matches, call self recursively to see whether the rest matches,
// and then act accordingly.  In practice we make some effort to avoid recursion, in particular by going through "ordinary" nodes (that don't
// need to know whether the rest of the match failed) by a loop instead of by recursion.
// 0 failure, 1 success
static __device__ int regmatch(regex_t *preg, int prog) {
	const char *save;
	int scan = prog; // Current node
#ifdef _DEBUGx
	if (scan != 0 && regnarrate)
		printf("%s(\n", regprop(scan));
#endif
	while (scan != 0) {
#ifdef _DEBUGx
		if (regnarrate)
			printf("%3d: %s...\n", scan, regprop(OP(preg, scan)));	// Where, what
#endif
		int next = regnext(preg, scan); // Next node
		int c;
		int n = reg_utf8_tounicode_case(preg->reginput, &c, (preg->cflags & REG_ICASE));
		switch (OP(preg, scan)) {
		case OP_BOL:
			if (preg->reginput != preg->regbol)
				return 0;
			break;
		case OP_EOL:
			if (!reg_iseol(preg, c))
				return 0;
			break;
		case OP_WORDA:
			// Must be looking at a letter, digit, or _
			if ((!isalnum(c)) && c != '_')
				return(0);
			// Prev must be BOL or nonword
			if (preg->reginput > preg->regbol && (isalnum(preg->reginput[-1]) || preg->reginput[-1] == '_'))
				return 0;
			break;
		case OP_WORDZ:
			// Can't match at BOL
			if (preg->reginput > preg->regbol) {
				// Current must be EOL or nonword
				if (reg_iseol(preg, c) || !isalnum(c) || c != '_') {
					c = preg->reginput[-1];
					// Previous must be word
					if (isalnum(c) || c == '_')
						break;
				}
			}
			// No
			return 0;
		case OP_ANY:
			if (reg_iseol(preg, c))
				return 0;
			preg->reginput += n;
			break;
		case OP_EXACTLY: {
			int opnd = OPERAND(scan);
			int len = _strlenint(preg->program + opnd);
			int slen = prefix_cmp(preg->program + opnd, len, preg->reginput, preg->cflags & REG_ICASE);
			if (slen < 0)
				return 0;
			preg->reginput += slen;
		} break;
		case OP_ANYOF:
			if (reg_iseol(preg, c) || reg_range_find(preg->program + OPERAND(scan), c) == 0)
				return 0;
			preg->reginput += n;
			break;
		case OP_ANYBUT:
			if (reg_iseol(preg, c) || reg_range_find(preg->program + OPERAND(scan), c) != 0)
				return 0;
			preg->reginput += n;
			break;
		case OP_NOTHING:
			break;
		case OP_BACK:
			break;
		case OP_BRANCH:
			if (OP(preg, next) != OP_BRANCH) // No choice
				next = OPERAND(scan); // Avoid recursion
			else {
				do {
					save = preg->reginput;
					if (regmatch(preg, OPERAND(scan)))
						return 1;
					preg->reginput = save;
					scan = regnext(preg, scan);
				} while (scan != 0 && OP(preg, scan) == OP_BRANCH);
				return 0;
			}
			break;
		case OP_REP:
		case OP_REPMIN:
			return regmatchsimplerepeat(preg, scan, OP(preg, scan) == OP_REPMIN);
		case OP_REPX:
		case OP_REPXMIN:
			return regmatchrepeat(preg, scan, OP(preg, scan) == OP_REPXMIN);
		case OP_END:
			return 1; // Success!
		case OP_OPENNC:
		case OP_CLOSENC:
			return regmatch(preg, next);
		default:
			if (OP(preg, scan) >= OP_OPEN + 1 && OP(preg, scan) < OP_CLOSE_END) {
				save = preg->reginput;
				if (regmatch(preg, next)) {
					if (OP(preg, scan) < OP_CLOSE) {
						int no = OP(preg, scan) - OP_OPEN;
						if (no < preg->nmatch && preg->pmatch[no].rm_so == -1)
							preg->pmatch[no].rm_so = (int)(save - preg->start);
					}
					else {
						int no = OP(preg, scan) - OP_CLOSE;
						if (no < preg->nmatch && preg->pmatch[no].rm_eo == -1)
							preg->pmatch[no].rm_eo = (int)(save - preg->start);
					}
					return 1;
				}
				return 0;
			}
			return REG_ERR_INTERNAL;
		}
		scan = next;
	}

	// We get here only if there's trouble -- normally "case END" is the terminating point.
	return REG_ERR_INTERNAL;
}

// regrepeat - repeatedly match something simple, report how many
static __device__ int regrepeat(regex_t *preg, int p, int max) {
	int count = 0;
	int ch;
	int n;
	const char *scan = preg->reginput;
	int opnd = OPERAND(p);
	switch (OP(preg, p)) {
	case OP_ANY:
		// No need to handle utf8 specially here
		while (!reg_iseol(preg, *scan) && count < max) {
			count++;
			scan++;
		}
		break;
	case OP_EXACTLY:
		while (count < max) {
			n = reg_utf8_tounicode_case(scan, &ch, preg->cflags & REG_ICASE);
			if (preg->program[opnd] != ch)
				break;
			count++;
			scan += n;
		}
		break;
	case OP_ANYOF:
		while (count < max) {
			n = reg_utf8_tounicode_case(scan, &ch, preg->cflags & REG_ICASE);
			if (reg_iseol(preg, ch) || reg_range_find(preg->program + opnd, ch) == 0)
				break;
			count++;
			scan += n;
		}
		break;
	case OP_ANYBUT:
		while (count < max) {
			n = reg_utf8_tounicode_case(scan, &ch, preg->cflags & REG_ICASE);
			if (reg_iseol(preg, ch) || reg_range_find(preg->program + opnd, ch) != 0)
				break;
			count++;
			scan += n;
		}
		break;
	default: // Oh dear.  Called inappropriately
		preg->err = REG_ERR_INTERNAL;
		count = 0; // Best compromise
		break;
	}
	preg->reginput = scan;
	return count;
}

// regnext - dig the "next" pointer out of a node
static __device__ int regnext(regex_t *preg, int p) {
	int offset = NEXT(preg, p);
	if (offset == 0) return 0;
	return OP(preg, p) == OP_BACK ? p - offset : p + offset;
}

// regopsize - returns the size of opcode + operands at 'p' in words
static __device__ int regopsize(regex_t *preg, int p) {
	// Almost all opcodes are 2 words, but some are more
	switch (OP(preg, p)) {
	case OP_REP:
	case OP_REPMIN:
	case OP_REPX:
	case OP_REPXMIN:
		return 5;
	case OP_ANYOF:
	case OP_ANYBUT:
	case OP_EXACTLY:
		int s = p + 2;
		while (preg->program[s++]) {}
		return (int)(s - p);
	}
	return 2;
}

#ifdef _DEBUGx
// regdump - dump a regexp onto stdout in vaguely comprehensible form
#define MAX_UTF8_LEN 4
static __device__ void regdump(regex_t *preg) {
	for (int i = 1; i < preg->p; i++) {
		printf("%02x ", (unsigned char)preg->program[i]);
		if (i % 16 == 0)
			printf("\n");
	}
	printf("\n");

	int op = OP_EXACTLY; // Arbitrary non-END op
	char buf[MAX_UTF8_LEN + 1];
	int s = 1;
	while (op != OP_END && s < preg->p) { // While that wasn't END last time...
		op = OP(preg, s);
		printf("%3d: %s", s, regprop(op)); // Where, what
		int next = regnext(preg, s);
		if (!next) // Next ptr
			printf("(0)");
		else
			printf("(%d)", next);
		s += 2;
		if (op == OP_REP || op == OP_REPMIN || op == OP_REPX || op == OP_REPXMIN) {
			int max = preg->program[s];
			int min = preg->program[s + 1];
			if (max == 65535)
				printf("{%d,*}", min);
			else
				printf("{%d,%d}", min, max);
			printf(" %d", preg->program[s + 2]);
			s += 3;
		}
		else if (op == OP_ANYOF || op == OP_ANYBUT) {
			// set of ranges
			while (preg->program[s]) {
				int len = preg->program[s++];
				int first = preg->program[s++];
				buf[utf8_getchars(buf, first)] = 0;
				printf("%s", buf);
				if (len > 1) {
					buf[utf8_getchars(buf, first + len - 1)] = 0;
					printf("-%s", buf);
				}
			}
			s++;
		}
		else if (op == OP_EXACTLY) {
			// Literal string, where present
			while (preg->program[s]) {
				buf[utf8_getchars(buf, preg->program[s])] = 0;
				printf("%s", buf);
				s++;
			}
			s++;
		}
		putchar('\n');
	}

	if (op == OP_END) {
		// Header fields of interest
		if (preg->regstart) {
			buf[utf8_getchars(buf, preg->regstart)] = 0;
			printf("start '%s' ", buf);
		}
		if (preg->reganch)
			printf("anchored ");
		if (preg->regmust != 0) {
			printf("must have:");
			for (int i = 0; i < preg->regmlen; i++)
				putchar(preg->program[preg->regmust + i]);
			putchar('\n');
		}
	}
	printf("\n");
}

// regprop - printable representation of opcode
static __device__ char _regprop_buf[50];
static __device__ const char *regprop(int op) {
	switch (op) {
	case OP_BOL: return "BOL";
	case OP_EOL: return "EOL";
	case OP_ANY: return "ANY";
	case OP_ANYOF: return "ANYOF";
	case OP_ANYBUT: return "ANYBUT";
	case OP_BRANCH: return "BRANCH";
	case OP_EXACTLY: return "EXACTLY";
	case OP_NOTHING: return "NOTHING";
	case OP_BACK: return "BACK";
	case OP_END: return "END";
	case OP_REP: return "REP";
	case OP_REPMIN: return "REPMIN";
	case OP_REPX: return "REPX";
	case OP_REPXMIN: return "REPXMIN";
	case OP_WORDA: return "WORDA";
	case OP_WORDZ: return "WORDZ";
	case OP_OPENNC: return "OPEN";
	case OP_CLOSENC: return "CLOSE";
	default:
		if (op >= OP_OPEN && op < OP_CLOSE) snprintf(_regprop_buf, sizeof(_regprop_buf), "OPEN%d", op - OP_OPEN);
		else if (op >= OP_CLOSE && op < OP_CLOSE_END) snprintf(_regprop_buf, sizeof(_regprop_buf), "CLOSE%d", op - OP_CLOSE);
		else snprintf(_regprop_buf, sizeof(_regprop_buf), "?%d?\n", op);
		return _regprop_buf;
	}
}
#endif
static __constant__ const char *error_strings[] = {
	"success",
	"no match",
	"bad pattern",
	"null argument",
	"unknown error",
	"too big",
	"out of memory",
	"too many ()",
	"parentheses () not balanced",
	"braces {} not balanced",
	"invalid repetition count(s)",
	"extra characters",
	"*+ of empty atom",
	"nested count",
	"internal error",
	"count follows nothing",
	"trailing backslash",
	"corrupted program",
	"contains null char",
};
__device__ size_t regerror_(int errcode, const regex_t *preg, char *errbuf, size_t errbuf_size) {
	const char *err;
	if (errcode < 0 || errcode >= REG_ERR_NUM) err = "Bad error code";
	else err = error_strings[errcode];
	return snprintf(errbuf, errbuf_size, "%s", err);
}

__device__ void regfree_(regex_t *preg) {
	free(preg->program);
}

#undef OP
#undef NEXT
#undef OPERAND