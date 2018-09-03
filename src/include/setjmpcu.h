/*
setjmp.h - 7.13 Nonlocal jumps
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
#ifndef _SETJMPCU_H
#define _SETJMPCU_H
#include <crtdefscu.h>

#include <setjmp.h>
#if defined(__CUDA_ARCH__)
__BEGIN_DECLS;

//struct __jmp_buf_tag {
//};
//typedef struct __jmp_buf_tag jmp_buf[1];

__BEGIN_NAMESPACE_STD;
/* Store the calling environment in ENV, also saving the signal mask. Return 0.  */
extern __device__ int setjmp_(jmp_buf env);
#undef setjmp
#define setjmp setjmp_
__END_NAMESPACE_STD;

///* Store the calling environment in ENV, also saving the signal mask if SAVEMASK is nonzero.  Return 0. This is the internal name for `sigsetjmp'.  */
//nosupport: extern __device__ int __sigsetjmp_(struct __jmp_buf_tag env[1], int savemask);
//#define __sigsetjmp __sigsetjmp_

///* Store the calling environment in ENV, not saving the signal mask. Return 0.  */
//nosupport: extern __device__ int _setjmp_(struct __jmp_buf_tag env[1]);
//#define _setjmp _setjmp_
///* Do not save the signal mask.  This is equivalent to the `_setjmp' BSD function.  */
//#define setjmp(env) _setjmp_(env)

__BEGIN_NAMESPACE_STD;
/* Jump to the environment saved in ENV, making the `setjmp' call there return VAL, or 1 if VAL is 0.  */
extern __device__ void longjmp_(jmp_buf env, int val);
#define longjmp longjmp_
__END_NAMESPACE_STD;

__END_DECLS;
#endif  /* __CUDA_ARCH__ */

#endif  /* _SETJMPCU_H */
