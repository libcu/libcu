#if __CUDACC__
#define TCL_PLATFORM_OS "unknown"
#define TCL_PLATFORM_PLATFORM "gpu"
#define TCL_PLATFORM_PATH_SEPARATOR ":"
//#define HAVE_LONG_LONG
#define HAVE_DIRENT_H
#define HAVE_UNISTD_H
#elif defined(_MSC_VER)
#define TCL_PLATFORM_OS "windows"
#define TCL_PLATFORM_PLATFORM "windows"
#define TCL_PLATFORM_PATH_SEPARATOR ";"
#define HAVE_MKDIR_ONE_ARG
#define HAVE_SYSTEM
#elif defined(__MINGW32__)
#define TCL_PLATFORM_OS "mingw"
#define TCL_PLATFORM_PLATFORM "windows"
#define TCL_PLATFORM_PATH_SEPARATOR ";"
#define HAVE_MKDIR_ONE_ARG
#define HAVE_SYSTEM
#define HAVE_SYS_TIME_H
#define HAVE_DIRENT_H
#define HAVE_UNISTD_H
#else
#define TCL_PLATFORM_OS "unknown"
#define TCL_PLATFORM_PLATFORM "unix"
#define TCL_PLATFORM_PATH_SEPARATOR ":"
#define HAVE_VFORK
#define HAVE_WAITPID
#define HAVE_ISATTY
#define HAVE_MKSTEMP
#define HAVE_LINK
#define HAVE_SYS_TIME_H
#define HAVE_DIRENT_H
#define HAVE_UNISTD_H
#endif