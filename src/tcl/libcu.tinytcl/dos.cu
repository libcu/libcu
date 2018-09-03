// dos.c New DOS functions.
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include "Tcl.h"
#include "TclExtdInt.h"
#include <bios.h>
#include <time.h>
#include <dos.h>
#include <dir.h>
#include <alloc.h>
#include <conio.h>

#ifdef COMPILE_BIOS_MEMSIZE
// bios_memsize - return the size of memory according to the BIOS
int cmdbios_memsize(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	long memsize = _bios_memsize() * 1024;
	sprintf(interp->result, "%ld", memsize);
	return TCL_OK;
}
#endif

// bios_equiplist - return the equipment list according to the BIOS
//
// This is a very primitive command, probably dating back to the original PC.  You can't find out much, and it's all packed into the word that's
// returned.  If you really need this data, this code should be expanded to unpack it for you.
#ifdef COMPILE_BIOS_EQUIPLIST
int cmdbios_equiplist(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	sprintf(interp->result, "%u", _bios_equiplist());
	return TCL_OK;
}
#endif

// kbhit - returns 1 if a key has been hit, else 0
int cmdkbhit(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	sprintf(interp->result, "%d", kbhit());
	return TCL_OK;
}

// getkey - returns a key as an integer keycode.  waits until a key has been pressed.  (So use kbhit to see if one is there first, if you don't want to wait.)
int cmdgetkey(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	sprintf(interp->result, "%d", getch());
	return TCL_OK;
}

// sound frequency - start the sound playing a square wave at the specified frequency in hertz.  If 0, stops the sound from playing.
int cmdsound(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " frequency\"", (char *)NULL);
		return TCL_ERROR;
	}
	int frequency;
	if (Tcl_GetInt(interp, args[1], &frequency) != TCL_OK) {
		return TCL_ERROR;
	}
	if (frequency == 0) {
		nosound();
	} else {
		sound(frequency);
	}
	return TCL_OK;
}

// getdate - returns the current date as a list containing month, day, year
int cmdgetdate(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	struct dosdate_t date;
	_dos_getdate(&date);
	sprintf(interp->result,  "%d %d %d", date.month, date.day, date.year);
	return TCL_OK;
}

// setdate month day year -- set the current date to the specified month, day and year
int cmdsetdate(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " month day year", (char *)NULL);
		return TCL_ERROR;
	}
	int month;
	if (Tcl_GetInt(interp, args[1], &month) != TCL_OK) {
		return TCL_ERROR;
	}
	int day;
	if (Tcl_GetInt(interp, args[2], &day) != TCL_OK) {
		return TCL_ERROR;
	}
	int year;
	if (Tcl_GetInt(interp, args[3], &year) != TCL_OK) {
		return TCL_ERROR;
	}
	struct dosdate_t date;
	date.year = year;
	date.day = day;
	date.month = month;
	date.dayofweek = 0;
	if (_dos_setdate (&date) != 0) {
		Tcl_AppendResult(interp, "invalid date", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

// gettime - returns the current time as a list containing hours, minutes, and seconds
int cmdgettime(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	struct dostime_t time;
	_dos_gettime(&time);
	sprintf(interp->result, "%d %d %d", time.hour, time.minute, time.second);
	return TCL_OK;
}

// settime hour minute second -- set the current time to the specified hour, minute and second
int cmdsettime(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 4) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " hour minute second", (char *)NULL);
		return TCL_ERROR;
	}
	int hour;
	if (Tcl_GetInt(interp, args[1], &hour) != TCL_OK) {
		return TCL_ERROR;
	}
	int minute;
	if (Tcl_GetInt(interp, args[2], &minute) != TCL_OK) {
		return TCL_ERROR;
	}
	int second;
	if (Tcl_GetInt(interp, args[3], &second) != TCL_OK) {
		return TCL_ERROR;
	}
	struct dostime_t time;
	time.hour = hour;
	time.minute = minute;
	time.second = second;
	time.hsecond = 0;
	if (_dos_settime(&time) != 0) {
		Tcl_AppendResult(interp, "invalid time", (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

// convert_drive_id letter - converts a drive ID from a letter, like 'a' or 'A' to the integer dos uses, where 0 = A, 1 = B, etc.
//
// returns -1 if the ID is invalid, and sets an error message into the interpreter result buffer.
int convert_drive_id(Tcl_Interp *interp, char *driveString)
{
	int drive;
	char driveChar = *driveString;
	if (driveChar == '\0' || driveString[1] != '\0') goto bad_drive;
	if (isupper(driveChar)) {
		drive = driveChar - 'A' + 1;
		return drive;
	}
	if (_islower(driveChar)) {
		drive = driveChar - 'a' + 1;
		return drive;
	}
bad_drive:
	Tcl_AppendResult(interp, "invalid drive id", (char *)NULL);
	return -1;
}

// diskfree - returns a list containing the free kbytes and the total kbytes on the specified drive letter.
//
// Currently this works under a DOS window but not on the handheld.
#ifdef COMPILE_DISKFREE
int cmddiskfree(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " drive\"", (char *)NULL);
		return TCL_ERROR;
	}
	int drive;
	if ((drive = convert_drive_id (interp, args[1])) < 0) {
		return TCL_ERROR;
	}
	struct diskfree_t diskfree;
	if (_dos_getdiskfree (drive, &diskfree) != 0) {
		Tcl_AppendResult(interp, "Couldn't get disk space: ", Tcl_OSError(interp), (char *)NULL);
		return TCL_ERROR;
	}
	long freeClusters = diskfree.avail_clusters;
	long totalClusters = diskfree.total_clusters;
	int kbytesPerCluster = (diskfree.bytes_per_sector * diskfree.sectors_per_cluster) / 1024;
	sprintf(interp->result, "%ld %ld", (long)(freeClusters * kbytesPerCluster), (long)(totalClusters * kbytesPerCluster));
	return TCL_OK;
}
#endif

// cmdgetfat - returns a list containing the FAT ID byte, sectors per cluster, number of clusters, and bytes per sector.
#ifdef COMPILE_GETFAT
int cmdgetfat(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " drive\"", (char *)NULL);
		return TCL_ERROR;
	}
	int drive;
	if ((drive = convert_drive_id(interp, args[1])) < 0) {
		return TCL_ERROR;
	}
	struct fatinfo dtable;
	getfat(drive, &dtable);
	sprintf(interp->result, "%u %u %u %u", dtable.fi_fatid, dtable.fi_sclus, dtable.fi_nclus, dtable.fi_bysec);
	return TCL_OK;
}
#endif

// cmdgetdfree - returns a list containing the available clusters, total clusters, bytes per sector, and sectors per cluster.
int cmdgetdfree(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " drive\"", (char *)NULL);
		return TCL_ERROR;
	}
	int drive;
	if ((drive = convert_drive_id (interp, args[1])) < 0) {
		return TCL_ERROR;
	}
	struct dfree dtable;
	getdfree(drive, &dtable);
	sprintf(interp->result, "%u %u %u %u", dtable.df_avail, dtable.df_total, dtable.df_bsec, dtable.df_sclus);
	return TCL_OK;
}

// drive letter - set the current drive to the specified letter
int cmddrive(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc > 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " ?driveid?\"", (char *)NULL);
		return TCL_ERROR;
	}

	unsigned drive;
	if (argc == 1) {
		_dos_getdrive(&drive);
		sprintf(interp->result, "%c", drive + 'A' - 1);
		return TCL_OK;
	}
	if ((drive = convert_drive_id(interp, args[1])) < 0) {
		return TCL_ERROR;
	}
	unsigned ndrives;
	unsigned newdrive;
	_dos_setdrive(drive, &ndrives);
	_dos_getdrive(&newdrive);
	if (drive != newdrive) {
		Tcl_AppendResult(interp, "invalid drive id", (char *)NULL);
	}
	return TCL_OK;
}

// memfree - returns the amount of RAM left on the system
int cmdmemfree(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	sprintf(interp->result, "%lu", farcoreleft());
	return TCL_OK;
}

// stackfree - returns the amount of stack left on the system
int cmdstackfree(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	sprintf(interp->result, "%u", stackavail());
	return TCL_OK;
}

#ifdef COMPILE_BIOS_SERIALCOM
// bios_serialcom - does serial I/O stuff through the BIOS
int cmdbios_serialcom(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc < 3 || argc > 4) {
argcount:
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " command port (data)\"", (char *)NULL);
		return TCL_ERROR;
	}
	int command;
	int port;
	int data = 0;
	bool needData = false;
	if (STREQU(args[1], "status")) {
		command = _COM_STATUS;
		if (argc != 3) goto argcount;
	} else if (STREQU(args[1], "receive")) {
		command = _COM_RECEIVE;
		if (argc != 3) goto argcount;
	} else if (STREQU(args[1], "send")) {
		command = _COM_SEND;
		if (argc != 4) goto argcount;
		if (Tcl_GetInt(interp, args[2], &port) != TCL_OK) {
			return TCL_ERROR;
		}
		char *s;
		for (s = args[3]; *s != '\0'; s++) {
			_bios_serialcom(command, port, *s);
		}
		return TCL_OK;
	} else if (STREQU (args[1], "init")) {
		command = _COM_INIT;
		if (argc == 3) {
			data = (_COM_9600|_COM_NOPARITY|_COM_CHR8|_COM_STOP1);
		} else {
			if (argc != 4) goto argcount;
			needData = true;
		}
	} else {
		Tcl_AppendResult(interp, "bad arg: ", args[0], " command must be one of \"init\", \"send\", \"receive\" or \"status\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (Tcl_GetInt(interp, args[2], &port) != TCL_OK) {
		return TCL_ERROR;
	}
	if (needData && Tcl_GetInt(interp, args[3], &data) != TCL_OK) {
		return TCL_ERROR;
	}
	sprintf(interp->result, "%u", _bios_serialcom(command, port, data));
	return TCL_OK;
}
#endif

// rawclock - returns the raw clock value in ticks
int cmdrawclock(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	sprintf(interp->result, "%lu", clock());
	return TCL_OK;
}

// getverify - returns the operating system verify flag.  If 0, writes are not being verified.  If 1, they are.
#ifdef COMPILE_GETVERIFY
int cmdgetverify(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	sprintf(interp->result, "%d", getverify());
	return TCL_OK;
}
#endif

// wait ms - wait the specified number of milliseconds
int cmdwait(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " milliseconds\"", (char *)NULL);
		return TCL_ERROR;
	}
	int milliseconds;
	if (Tcl_GetInt(interp, args[1], &milliseconds) != TCL_OK) {
		return TCL_ERROR;
	}
	delay((unsigned)milliseconds);
	return TCL_OK;
}

// gotoxy - address the cursor to the specified x and y location
int cmdgotoxy(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 3) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " x y\"", (char *)NULL);
		return TCL_ERROR;
	}
	int x;
	if (Tcl_GetInt(interp, args[1], &x) != TCL_OK) {
		return TCL_ERROR;
	}
	int y;
	if (Tcl_GetInt(interp, args[2], &y) != TCL_OK) {
		return TCL_ERROR;
	}
	gotoxy(x, y);
	return TCL_OK;
}

// clrscr - clear the screen
int cmdclrscr(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	clrscr();
	return TCL_OK;
}

// heapcheck - check the heap for corruption
int cmdheapcheck(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 1) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], "\"", (char *)NULL);
		return TCL_ERROR;
	}
	switch (heapcheck())
	{
	case _HEAPCORRUPT:
		panic("Memory heap corrupted.");
		break;
	case _HEAPEMPTY:
		panic("No memory heap.");
		break;
	case _HEAPOK:
		return TCL_OK;
	default:
		_printf("Unknown error in memory heap.");
		break;
	}
}

// mkdir dir - create a directory
int cmdmkdir(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " dirname\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (mkdir(args[1]) < 0) {
		Tcl_AppendResult(interp, "Couldn't make directory: ", args[1], ": ", Tcl_OSError(interp), (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

// unlink file - delete a file
int cmdunlink(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc != 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " filename\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (unlink(args[1]) < 0) {
		Tcl_AppendResult(interp, "Couldn't unlink file: ", args[1], ": ", Tcl_OSError(interp), (char *)NULL);
		return TCL_ERROR;
	}
	return TCL_OK;
}

// execvp -- terminate the current process and execute a new one
int cmdexecvp(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc < 2) {
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " command ?args?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (execvp (args[1], &args[1]) < 0) {
		Tcl_AppendResult(interp, "Couldn't execvp: ", args[1], ": ", Tcl_OSError(interp), (char *)NULL);
		return TCL_ERROR;
	}
	// Actually we should not ever get here.  If the execvp succeeds, we terminate.
	return TCL_OK;
}

/*
video normal
video dim
video bright

video write data

video goto x y

video bgcolor n
video color n
video color n blink

normvideo
lowvideo
highvideo
textbackground
textcolor

textbackground(0-7)
0	black
1	blue
2	green
3	cyan
4	red
5	magenta
6	brown
7	lightgray

textcolor(0-15, +128 = blink)
same as above, plus
8	darkgray
9	lightblue
10	lightgreen
11	lightcyan
12	lightred
13	lightmagenta
14	yellow
15	white

wherex - get horizontal cursor position
wherey - get vertical cursor position
*/
/* video - do stuff to the video */
int cmdvideo(ClientData clientData, Tcl_Interp *interp, int argc, const char *args[])
{
	if (argc < 2) {
argcount:
		Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " subcommand ?options?\"", (char *)NULL);
		return TCL_ERROR;
	}
	if (STREQU(args[1], "write")) {
		if (argc != 3) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " ", args[1], " data\"", (char *)NULL);
			return TCL_ERROR;
		}
		cputs(args[2]);
		return TCL_OK;
	}
	int x;
	int y;
	if (STREQU(args[1], "goto")) {
		if (argc != 4) {
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " ", args[1], " x y\"", (char *)NULL);
			return TCL_ERROR;
		}
		if (Tcl_GetInt(interp, args[2], &x) != TCL_OK) {
			return TCL_ERROR;
		}
		if (Tcl_GetInt(interp, args[3], &y) != TCL_OK) {
			return TCL_ERROR;
		}
		gotoxy(x, y);
		return TCL_OK;
	}
	if (STREQU(args[1], "normal")) {
		if (argc != 2) goto bad2arg;
		normvideo();
		return TCL_OK;
	}
	if (STREQU (args[1], "dim")) {
		if (argc != 2) goto bad2arg;
		lowvideo();
		return TCL_OK;
	}
	int color;
	int blink;
	if (STREQU (args[1], "bright")) {
		if (argc != 2) goto bad2arg;
		highvideo();
		return TCL_OK;
	}
	else if (STREQU(args[1], "color")) {
		if (argc == 3) {
			blink = 0;
		} else {
			if (argc != 4) goto argcount;
			blink = 128;
		}
		if (Tcl_GetInt(interp, args[2], &color) != TCL_OK) {
			return TCL_ERROR;
		}
		if (color < 0 || color > 15) {
			Tcl_AppendResult(interp, "color must be between 0 & 15", (char *)NULL);
			return TCL_ERROR;
		}
		textcolor(color + blink);
		return TCL_OK;
	}
	if (STREQU(args[1], "bgcolor")) {
		if (argc != 3) goto argcount;
		if (Tcl_GetInt(interp, args[2], &color) != TCL_OK) {
			return TCL_ERROR;
		}
		if (color < 0 || color > 7) {
			Tcl_AppendResult(interp, "bgcolor must be between 0 & 7", (char *)NULL);
			return TCL_ERROR;
		}
		textbackground(color + blink);
		return TCL_OK;
	}
	if (STREQU(args[1], "clear")) {
		if (argc != 2) {
bad2arg:
			Tcl_AppendResult(interp, "wrong # args: should be \"", args[0], " ", args[1], "\"", (char *)NULL);
			return TCL_ERROR;
		}
		clrscr();
		return TCL_OK;
	}
	extern int _directvideo;
	if (STREQU(args[1], "direct")) {
		int directvideo;
		if (argc == 2) {
			sprintf(interp->result, "%d", _directvideo);
			return TCL_OK;
		}
		if (argc > 3) goto argcount;
		if (Tcl_GetInt(interp, args[2], &directvideo) != TCL_OK) {
			return TCL_ERROR;
		}
		_directvideo = directvideo;
		return TCL_OK;
	}
	Tcl_AppendResult(interp, "bad arg: ", args[0], " subcommand must be one of \"data\", \"goto\", \"clear\", \"normal\", \"dim\", \"bright\", \"color\" or \"bgcolor\"", (char *)NULL);
	return TCL_ERROR;
}

// Tcl_InitDos - add all of the DOS functions defined in this file to the specified interpreter.
int Tcl_InitDos(Tcl_Interp *interp)
{
#ifdef COMPILE_BIOS_MEMSIZE
	Tcl_CreateCommand(interp, "bios_memsize", cmdbios_memsize, (ClientData) 0, (Tcl_CmdDeleteProc *)NULL);
#endif
#ifdef COMPILE_BIOS_EQUIPLIST
	Tcl_CreateCommand(interp, "bios_equiplist", cmdbios_equiplist, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
#ifdef COMPILE_BIOS_SERIALCOM
	Tcl_CreateCommand(interp, "com", cmdbios_serialcom, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
	Tcl_CreateCommand(interp, "kbhit", cmdkbhit, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "getkey", cmdgetkey, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "sound", cmdsound, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "rawclock", cmdrawclock, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "getdate", cmdgetdate, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "setdate", cmdsetdate, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "gettime", cmdgettime, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "settime", cmdsettime, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#ifdef COMPILE_DISKFREE
	Tcl_CreateCommand(interp, "diskfree", cmddiskfree, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
#ifdef COMPILE_GETFAT
	Tcl_CreateCommand(interp, "getfat", cmdgetfat, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
	Tcl_CreateCommand(interp, "getdfree", cmdgetdfree, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "drive", cmddrive, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "memfree", cmdmemfree, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "stackfree", cmdstackfree, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "wait", cmdwait, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "gotoxy", cmdgotoxy, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "cls", cmdclrscr, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#ifdef COMPILE_GETVERIFY
	Tcl_CreateCommand(interp, "getverify", cmdgetverify, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
#endif
	Tcl_CreateCommand(interp, "heapcheck", cmdheapcheck, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "mkdir", cmdmkdir, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "unlink", cmdunlink, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "execvp", cmdexecvp, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	Tcl_CreateCommand(interp, "video", cmdvideo, (ClientData)0, (Tcl_CmdDeleteProc *)NULL);
	return TCL_OK;
}

