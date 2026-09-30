#ifndef LIBCU_LEAN_AND_MEAN
#include <stringcu.h>
#include <pwdcu.h>

__BEGIN_DECLS;

#if __OS_WIN
static __device__ passwd __pwds[] = { { (char *)"std", 1, 1 }, { nullptr } };
static __device__ passwd *__pwdIdx = nullptr;
#endif

/* search user database for a user ID */
__device__ struct passwd *getpwuid_(uid_t uid) {
#if __OS_WIN
	register passwd *p = __pwds;
	while (p->pw_name && p->pw_uid != uid) *p++;
	return (p->pw_name ? p : nullptr);
#elif __OS_UNIX
	return nullptr;
#endif
}

/* search user database for a name */
__device__ struct passwd *getpwnam_(const char *name) {
#if __OS_WIN
	if (!name) return nullptr;
	register passwd *p = __pwds;
	while (p->pw_name && strcmp(p->pw_name, name)) *p++;
	return (p->pw_name ? p : nullptr);
#elif __OS_UNIX
	return nullptr;
#endif
}

/* get user database entry [NOTSAFE] */
__device__ struct passwd *getpwent_() {
#if __OS_WIN
	if (!__pwdIdx) __pwdIdx = __pwds;
	else if (__pwdIdx->pw_name) __pwdIdx++;
	return (__pwdIdx->pw_name ? __pwdIdx : nullptr);
#elif __OS_UNIX
	return nullptr;
#endif
}

/* close the user database */
/* setpwent - reset user database to first entry */
__device__ void endpwent_() {
#if __OS_WIN
	__pwdIdx = nullptr;
#elif __OS_UNIX
#endif
}

__END_DECLS;
#endif