#include <stringcu.h>
#include <grpcu.h>

__BEGIN_DECLS;

#if __OS_WIN
static __device__ group __grps[] = { { "std", 1, nullptr }, { nullptr } };
static __device__ group *__grpIdx = nullptr;
#endif

/* get group database entry for a group ID */
__device__ struct group *getgrgid_(gid_t gid) {
#if __OS_WIN
	register group *p = __grps;
	while (p->gr_name && p->gr_gid != gid) p++;
	return (p->gr_name ? p : nullptr);
#elif __OS_UNIX
	return nullptr;
#endif
}

/* search group database for a name */
__device__ struct group *getgrnam_(const char *name) {
#if __OS_WIN
	if (!name) return nullptr;
	register group *p = __grps;
	while (p->gr_name && strcmp(p->gr_name, name)) *p++;
	return (p->gr_name ? p : nullptr);
#elif __OS_UNIX
	return nullptr;
#endif
}

/* get the group database entry */
__device__ struct group *getgrent_() {
#if __OS_WIN
	if (!__grpIdx) __grpIdx = __grps;
	else if (__grpIdx->gr_name) __grpIdx++;
	return (__grpIdx->gr_name ? __grpIdx : nullptr);
#elif __OS_UNIX
	return nullptr;
#endif
}

/* close the group database */
/* setgrent - reset group database to first entry */
__device__ void endgrent_() {
#if __OS_WIN
	__grpIdx = nullptr;
#elif __OS_UNIX
#endif
}

__END_DECLS;
