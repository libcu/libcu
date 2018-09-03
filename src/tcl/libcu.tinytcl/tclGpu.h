#ifndef __TCLGPU_H__
#define __TCLGPU_H__

#include <direntcu.h>
#include <unistdcu.h>
#include <sys/statcu.h>

/*
 * Define access mode constants if they aren't already defined.
 */

#ifndef F_OK
#    define F_OK 00
#endif
#ifndef X_OK
#    define X_OK 01
#endif
#ifndef W_OK
#    define W_OK 02
#endif
#ifndef R_OK
#    define R_OK 04
#endif

#endif /* __TCLGPU_H__ */
