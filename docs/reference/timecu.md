## #include <timecu.h>

Also includes:
```
#include <time.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ clock_t clock();``` | Time used by the program so far (user time + system time). The result / CLOCKS_PER_SECOND is program time in seconds.
```__device__ time_t time(time_t *timer);``` | Return the current time and put it in *TIMER if TIMER is not NULL.
```__device__ double difftime(time_t time1, time_t time0);``` | Return the difference between TIME1 and TIME0.
```__device__ time_t mktime(struct tm *tp);``` | Return the 'time_t' representation of TP and normalize TP.
```__device__ size_t strftime(char *__restrict s, size_t maxsize, const char *__restrict format, const struct tm *__restrict tp);``` | Format TP into S according to FORMAT. no more than MAXSIZE characters and return the number of characters written, or 0 if it would exceed MAXSIZE.
```__device__ struct tm *gmtime(const time_t *timer);``` | Return the 'struct tm' representation of *TIMER in Universal Coordinated Time (aka Greenwich Mean Time).
```#define localtime``` | Return the `struct tm' representation of *TIMER in the local timezone.
```__device__ char *asctime(const struct tm *tp);``` | Return a string of the form "Day Mon dd hh:mm:ss yyyy\n" that is the representation of TP in this format.
```__device__ char *ctime(const time_t *timer);``` | Equivalent to 'asctime (localtime (timer))'.
