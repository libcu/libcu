#include <errnocu.h>

__BEGIN_DECLS;

__device__ int errno_;
extern __device__ int *_errno_(void) { return &errno_; }
extern __device__ int _set_errno_(int value) { return (errno_ = value); }
extern __device__ int _get_errno_(int *value) { if (value) *value = errno_; return errno_; }

__END_DECLS;

// PORTABILITY
#pragma region PORTABILITY 
__BEGIN_DECLS;

#if __OS_WIN
#include <Windows.h>
extern int __Errno() {
	switch (GetLastError()) {
	case ERROR_FILE_NOT_FOUND: return ENOENT;
	case ERROR_PATH_NOT_FOUND: return ENOENT;
	case ERROR_TOO_MANY_OPEN_FILES: return EMFILE;
	case ERROR_ACCESS_DENIED: return EACCES;
	case ERROR_INVALID_HANDLE: return EBADF;
	case ERROR_BAD_ENVIRONMENT: return E2BIG;
	case ERROR_BAD_FORMAT: return ENOEXEC;
	case ERROR_INVALID_ACCESS: return EACCES;
	case ERROR_INVALID_DRIVE: return ENOENT;
	case ERROR_CURRENT_DIRECTORY: return EACCES;
	case ERROR_NOT_SAME_DEVICE: return EXDEV;
	case ERROR_NO_MORE_FILES: return ENOENT;
	case ERROR_WRITE_PROTECT: return EROFS;
	case ERROR_BAD_UNIT: return ENXIO;
	case ERROR_NOT_READY: return EBUSY;
	case ERROR_BAD_COMMAND: return EIO;
	case ERROR_CRC: return EIO;
	case ERROR_BAD_LENGTH: return EIO;
	case ERROR_SEEK: return EIO;
	case ERROR_WRITE_FAULT: return EIO;
	case ERROR_READ_FAULT: return EIO;
	case ERROR_GEN_FAILURE: return EIO;
	case ERROR_SHARING_VIOLATION: return EACCES;
	case ERROR_LOCK_VIOLATION: return EACCES;
	case ERROR_SHARING_BUFFER_EXCEEDED: return ENFILE;
	case ERROR_HANDLE_DISK_FULL: return ENOSPC;
	case ERROR_NOT_SUPPORTED: return ENODEV;
	case ERROR_REM_NOT_LIST: return EBUSY;
	case ERROR_DUP_NAME: return EEXIST;
	case ERROR_BAD_NETPATH: return ENOENT;
	case ERROR_NETWORK_BUSY: return EBUSY;
	case ERROR_DEV_NOT_EXIST: return ENODEV;
	case ERROR_TOO_MANY_CMDS: return EAGAIN;
	case ERROR_ADAP_HDW_ERR: return EIO;
	case ERROR_BAD_NET_RESP: return EIO;
	case ERROR_UNEXP_NET_ERR: return EIO;
	case ERROR_NETNAME_DELETED: return ENOENT;
	case ERROR_NETWORK_ACCESS_DENIED: return EACCES;
	case ERROR_BAD_DEV_TYPE: return ENODEV;
	case ERROR_BAD_NET_NAME: return ENOENT;
	case ERROR_TOO_MANY_NAMES: return ENFILE;
	case ERROR_TOO_MANY_SESS: return EIO;
	case ERROR_SHARING_PAUSED: return EAGAIN;
	case ERROR_REDIR_PAUSED: return EAGAIN;
	case ERROR_FILE_EXISTS: return EEXIST;
	case ERROR_CANNOT_MAKE: return ENOSPC;
	case ERROR_OUT_OF_STRUCTURES: return ENFILE;
	case ERROR_ALREADY_ASSIGNED: return EEXIST;
	case ERROR_INVALID_PASSWORD: return EPERM;
	case ERROR_NET_WRITE_FAULT: return EIO;
	case ERROR_NO_PROC_SLOTS: return EAGAIN;
	case ERROR_DISK_CHANGE: return EXDEV;
	case ERROR_BROKEN_PIPE: return EPIPE;
	case ERROR_OPEN_FAILED: return ENOENT;
	case ERROR_DISK_FULL: return ENOSPC;
	case ERROR_NO_MORE_SEARCH_HANDLES: return EMFILE;
	case ERROR_INVALID_TARGET_HANDLE: return EBADF;
	case ERROR_INVALID_NAME: return ENOENT;
	case ERROR_PROC_NOT_FOUND: return ESRCH;
	case ERROR_WAIT_NO_CHILDREN: return ECHILD;
	case ERROR_CHILD_NOT_COMPLETE: return ECHILD;
	case ERROR_DIRECT_ACCESS_HANDLE: return EBADF;
	case ERROR_SEEK_ON_DEVICE: return ESPIPE;
	case ERROR_BUSY_DRIVE: return EAGAIN;
	case ERROR_DIR_NOT_EMPTY: return EEXIST;
	case ERROR_NOT_LOCKED: return EACCES;
	case ERROR_BAD_PATHNAME: return ENOENT;
	case ERROR_LOCK_FAILED: return EACCES;
	case ERROR_ALREADY_EXISTS: return EEXIST;
	case ERROR_FILENAME_EXCED_RANGE: return ENAMETOOLONG;
	case ERROR_BAD_PIPE: return EPIPE;
	case ERROR_PIPE_BUSY: return EAGAIN;
	case ERROR_PIPE_NOT_CONNECTED: return EPIPE;
	case ERROR_DIRECTORY: return ENOTDIR;
	}
	return EINVAL;
}
#endif
extern const char *__Strerror() { return strerror(__Errno()); }

__END_DECLS;
#pragma endregion
