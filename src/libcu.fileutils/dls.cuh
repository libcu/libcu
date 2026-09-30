#include <ext/pipeline.h>
#include <sys/types.h>
#include <sys/statcu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <direntcu.h>
#include <pwdcu.h>
#include <grpcu.h>
#include <timecu.h>
#include <unistdcu.h>
#include "fileutils.h"

#undef S_ISLNK
#define	LISTSIZE 256
#define COLS 80

// Flags for the LS command.
#define	LSF_LONG	0x01
#define	LSF_DIR		0x02
#define	LSF_INODE	0x04
#define	LSF_MULT	0x08
#define LSF_ALL		0x10		// List files starting with `.'
#define LSF_CLASS	0x20		// Classify files (append symbol)

__device__ char **_list;
__device__ int _listSize;
__device__ int _listUsed;
__device__ int _cols = 0;
__device__ int _col = 0;
__device__ char _fmt[10] = "%s";

// Return the standard ls-like mode string from a file mode. This is static and so is overwritten on each call.
static __device__ char _modeString_buf[12];
__device__ char *modeString(int mode) {
	strcpy(_modeString_buf, "----------");

	// Fill in the file type.
	if (S_ISDIR(mode)) _modeString_buf[0] = 'd';
	if (S_ISCHR(mode)) _modeString_buf[0] = 'c';
	if (S_ISBLK(mode)) _modeString_buf[0] = 'b';
	if (S_ISFIFO(mode)) _modeString_buf[0] = 'p';
#ifdef S_ISLNK
	if (S_ISLNK(mode)) _modeString_buf[0] = 'l';
#endif
#ifdef	S_ISSOCK
	if (S_ISSOCK(mode)) _modeString_buf[0] = 's';
#endif

	// Now fill in the normal file permissions.
	if (mode & S_IRUSR) _modeString_buf[1] = 'r';
	if (mode & S_IWUSR) _modeString_buf[2] = 'w';
	if (mode & S_IXUSR) _modeString_buf[3] = 'x';
	if (mode & S_IRGRP) _modeString_buf[4] = 'r';
	if (mode & S_IWGRP) _modeString_buf[5] = 'w';
	if (mode & S_IXGRP) _modeString_buf[6] = 'x';
	if (mode & S_IROTH) _modeString_buf[7] = 'r';
	if (mode & S_IWOTH) _modeString_buf[8] = 'w';
	if (mode & S_IXOTH) _modeString_buf[9] = 'x';

	// Finally fill in magic stuff like suid and sticky text.
	//if (mode & S_ISUID) _modeString_buf[3] = ((mode & S_IXUSR) ? 's' : 'S');
	//if (mode & S_ISGID) _modeString_buf[6] = ((mode & S_IXGRP) ? 's' : 'S');
	//if (mode & S_ISVTX) _modeString_buf[9] = ((mode & S_IXOTH) ? 't' : 'T');

	return _modeString_buf;
}

// Get the time to be used for a file. This is down to the minute for new files, but only the date for old files.
// The string is returned from a static buffer, and so is overwritten for each call.
static __device__ char _timeString_buf[26];
__device__ char *timeString(time_t t) {
#ifndef LIBCU_LEAN_AND_MEAN
	time_t now = time(nullptr);
	char *str = ctime(&t);
	strcpy(_timeString_buf, &str[4]);
	_timeString_buf[12] = '\0';
	if (t > now || t < now - 365 * 24 * 60 * 60L) {
		strcpy(&_timeString_buf[7], &str[20]);
		_timeString_buf[11] = '\0';
	}
	return _timeString_buf;
#else
	return (char *)"time";
#endif
}

// Do an LS of a particular file name according to the flags.
static __device__ void lsFile(pipelineRedir *redir, char *fullName, char *name, struct stat *statbuf, int flags) {
	char *cp;
#ifndef LIBCU_LEAN_AND_MEAN
	struct passwd *pwd;
	struct group *grp;
#endif
	char buf[PATHLEN];
	static char userName[12];
	static int userId;
	static bool userIdKnown;
	static char groupName[12];
	static int groupId;
	static bool groupIdKnown;
	char *class_;

	cp = buf;
	*cp = '\0';

	if (flags & LSF_INODE) {
		sprintf(cp, "%5ld ", statbuf->st_ino);
		cp += strlen(cp);
	}

	if (flags & LSF_LONG) {
		strcpy(cp, modeString(statbuf->st_mode));
		cp += strlen(cp);

		sprintf(cp, "%3d ", statbuf->st_nlink);
		cp += strlen(cp);

		if (!userIdKnown || (statbuf->st_uid != userId)) {
#ifndef LIBCU_LEAN_AND_MEAN
			pwd = (struct passwd *)getpwuid(statbuf->st_uid);
			if (pwd)
				strcpy(userName, pwd->pw_name);
			else
				sprintf(userName, "%d", statbuf->st_uid);
#else
			strcpy(userName, "pwd");
#endif
			userId = statbuf->st_uid;
			userIdKnown = true;
		}

		sprintf(cp, "%-8s ", userName);
		cp += strlen(cp);

		if (!groupIdKnown || statbuf->st_gid != groupId) {
#ifndef LIBCU_LEAN_AND_MEAN
			grp = (struct group *)getgrgid(statbuf->st_gid);
			if (grp)
				strcpy(groupName, grp->gr_name);
			else
				sprintf(groupName, "%d", statbuf->st_gid);
#else
			strcpy(userName, "grp");
#endif
			groupId = statbuf->st_gid;
			groupIdKnown = true;
		}

		sprintf(cp, "%-8s ", groupName);
		cp += strlen(cp);

		if (S_ISBLK(statbuf->st_mode) || S_ISCHR(statbuf->st_mode))
			sprintf(cp, "%3d, %3d ", (int)statbuf->st_rdev >> 8,
			(int)statbuf->st_rdev & 0xff);
		else
			sprintf(cp, "%8ld ", statbuf->st_size);
		cp += strlen(cp);

		sprintf(cp, " %-12s ", timeString(statbuf->st_mtime));
	}
	fputs(buf, redir->out);

	class_ = name + strlen(name);
	*class_ = 0;
	if (flags & LSF_CLASS) {
#ifdef S_ISLNK
		if (S_ISLNK(statbuf->st_mode)) *class_ = '@';
		else
#endif
			if (S_ISDIR(statbuf->st_mode)) *class_ = '/';
			else if (S_IEXEC & statbuf->st_mode) *class_ = '*';
			else if (S_ISFIFO(statbuf->st_mode)) *class_ = '|';
			else if (S_ISSOCK(statbuf->st_mode)) *class_ = '=';
	}
	printf(_fmt, name);
#ifdef S_ISLNK
	if ((flags & LSF_LONG) && S_ISLNK(statbuf->st_mode)) {
		if (fullName) len = readlink(fullName, buf, PATHLEN - 1);
		else len = readlink(name, buf, PATHLEN - 1);
		if (len >= 0) {
			buf[len] = '\0';
			printf(" -> %s", buf);
		}
	}
#endif
	if (flags & LSF_LONG || ++_col == _cols) {
		fputc('\n', redir->out);
		_col = 0;
	}
}

// Build a path name from the specified directory name and file name. If the directory name is NULL, then the original filename is returned.
// The built path is in a static area, and is overwritten for each call.
//static __device__ char _buildName_buf[PATHLEN];
//__device__ char *buildName(char *dirName, char *fileName) {
//	if (!dirName || (*dirName == '\0'))
//		return fileName;
//	char *cp = strrchr(fileName, '/');
//	if (cp)
//		fileName = cp + 1;
//	strcpy(_buildName_buf, dirName);
//	strcat(_buildName_buf, "/");
//	strcat(_buildName_buf, fileName);
//	return _buildName_buf;
//}

// Sort routine for list of filenames.
__device__ int nameSort(const void *pp1, const void *pp2) {
	char **p1 = (char **)pp1;
	char **p2 = (char **)pp2;
	return strcmp(*p1, *p2);
}

__device__ int d_dls_rc;
__global__ void g_dls(pipelineRedir redir, char *name, int flags, bool endSlash) {
	if (!name) {
		// alloc list
		if (_listSize == 0) {
			_list = (char **)malloc(LISTSIZE * sizeof(char *));
			if (!_list) {
				printf("No memory for ls buffer\n");
				exit(1);
			}
			_listSize = LISTSIZE;
		}
		_listUsed = 0;
		d_dls_rc = -1;
		return;
	}

	struct stat statbuf;
	if (LSTAT(name, &statbuf) < 0) {
		fperror(redir.out, name);
		d_dls_rc = -1;
		return;
	}

	printf("c: %s %x %x\n", name, flags, statbuf.st_mode);
	if ((flags & LSF_DIR) || !S_ISDIR(statbuf.st_mode)) {
		lsFile(&redir, NULL, name, &statbuf, flags);
		if (~flags & LSF_LONG)
			fputc('\n', redir.out);
		d_dls_rc = -1;
		return;
	}

	printf("d: \n");
	// Do all the files in a directory.
	DIR *dirp = opendir(name);
	if (dirp == NULL) {
		fperror(redir.out, name);
		d_dls_rc = -1;
		return;
	}
	if (flags & LSF_MULT)
		fprintf(redir.out, "\n%s:\n", name);
	struct dirent *dp;
	char fullName[PATHLEN];
	while (dp = readdir(dirp)) {
		fullName[0] = '\0';
		if (*name != '.' || name[1] != '\0') {
			strcpy(fullName, name);
			if (!endSlash)
				strcat(fullName, "/");
		}
		// add to list
		strcat(fullName, dp->d_name);
		if (_listUsed >= _listSize) {
			char **newList = (char **)realloc(_list, (sizeof(char **) * (_listSize + LISTSIZE)));
			if (!newList) {
				printf("No memory for ls buffer\n");
				break;
			}
			_list = newList;
			_listSize += LISTSIZE;
		}
		strcat(fullName, " ");
		_list[_listUsed] = strdup(fullName);
		if (!_list[_listUsed]) {
			printf("No memory for filenames\n");
			break;
		}
		_list[_listUsed][strlen(fullName) - 1] = 0;
		_listUsed++;
	}
	closedir(dirp);

	// Sort the files.
	qsort((char *)_list, _listUsed, sizeof(char *), nameSort);

	int i; char *cp;

	// Get list entry size for multi-column output.
	if (~flags & LSF_LONG) {
		int len, maxlen;
		for (maxlen = i = 0; i < _listUsed; i++) {
			if (cp = strrchr(_list[i], '/')) cp++;
			else cp = _list[i];
			if ((len = strlen(cp)) > maxlen)
				maxlen = len;
		}
		maxlen += 2;
		_cols = (COLS - 1) / maxlen;
		sprintf(_fmt, "%%-%d.%ds", maxlen, maxlen);
	}

	// Now finally list the filenames.
	int num;
	for (num = i = 0; i < _listUsed; i++) {
		name = _list[i];
		if (LSTAT(name, &statbuf) < 0) {
			fperror(redir.out, name);
			free(name);
			continue;
		}
		cp = strrchr(name, '/');
		if (cp) cp++;
		else cp = name;
		if (flags & LSF_ALL || *cp != '.') {
			lsFile(&redir, name, cp, &statbuf, flags);
			num++;
		}
		free(name);
	}
	if ((~flags & LSF_LONG) && (num % _cols))
		fputc('\n', redir.out);
	_listUsed = 0;
	d_dls_rc = 0;
}
int dls(pipelineRedir redir, char *str, int flags, bool endSlash) {
	pipelineOpen(redir);
	char *d_str;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	g_dls<<<1, 1>>>(redir, d_str, flags, endSlash);
	if (d_str) cudaFree(d_str);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dls_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
