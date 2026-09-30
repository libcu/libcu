#include "fsystem.h"
#include <sys/statcu.h>
#include <fcntlcu.h>
#include <sentinel-fcntlmsg.h>
#include <sentinel-unistdmsg.h>
#include <stdio.h> // panic's printf

__BEGIN_DECLS;

__device__ int vfcntl_(int fd, int cmd, va_list va) {
	if (ISHOSTHANDLE(fd)) { fcntl_fcntl msg(fd, cmd, va_arg(va, int), false); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	panic("Not Implemented");
	// (int fd, unsigned int cmd, unsigned long arg, struct file *filp)
	//	long err = -EINVAL;
	//	switch (cmd) {
	//	case F_DUPFD: err = f_dupfd(arg, filp, 0); break;
	//	case F_GETFD: err = get_close_on_exec(fd) ? FD_CLOEXEC : 0; break;
	//	case F_SETFD: err = 0; set_close_on_exec(fd, arg & FD_CLOEXEC); break;
	//	case F_GETFL: err = filp->f_flags; break;
	//	case F_SETFL: err = setfl(fd, filp, arg); break;
	//	case F_GETOWN: err = f_getown(filp); force_successful_syscall_return(); break;
	//	case F_SETOWN: f_setown(filp, arg, 1); err = 0; break;
	//	default:
	//		break;
	//	}
	//	return err;
	return 0;
#endif
}
__device__ int fcntl_(int fd, int cmd, ...) { va_list va; va_start(va, cmd); int r = vfcntl_(fd, cmd, va); va_end(va); return r; }
#ifdef __USE_LARGEFILE64
__device__ int vfcntl64_(int fd, int cmd, va_list va) {
	if (ISHOSTHANDLE(fd)) { fcntl_fcntl msg(fd, cmd, va_arg(va, int), true); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	panic("Not Implemented");
	return 0;
#endif
}
__device__ int fcntl64_(int fd, int cmd, ...) { va_list va; va_start(va, cmd); int r = vfcntl64_(fd, cmd, va); va_end(va); return r; }
#endif

__device__ int vopen_(const char *file, int oflag, va_list va) {
	if (ISHOSTPATH(file)) { fcntl_open msg(file, oflag, va_arg(va, int), false); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	int fd; fsystemOpen(file, oflag, &fd); return fd;
#endif
}
__device__ int open_(const char *file, int oflag, ...) { va_list va; va_start(va, oflag); int r = vopen_(file, oflag, va); va_end(va); return r; }
#ifdef __USE_LARGEFILE64
__device__ int vopen64_(const char *file, int oflag, va_list va) {
	if (ISHOSTPATH(file)) { fcntl_open msg(file, oflag, va_arg(va, int), true); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	int fd; fsystemOpen(file, oflag, &fd); return fd;
#endif
}
__device__ int open64_(const char *file, int oflag, ...) { va_list va; va_start(va, oflag); int r = vopen64_(file, oflag, va); va_end(va); return r; }
#endif

/* Close the file descriptor FD.  */
__device__ int close_(int fd) {
	if (ISHOSTHANDLE(fd)) { unistd_close msg(fd); return msg.rc; }
#ifdef LIBCU_LEAN_FSYSTEM
	return panic_no_fsystem();
#else
	fsystemClose(fd);
	return 0;
#endif
}

__END_DECLS;
