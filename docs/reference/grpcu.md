## #include <grpcu.h>

Also includes:
```
#include <grp.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ struct group *getgrgid(gid_t gid);``` | get group database entry for a group ID
```__device__ struct group *getgrnam(const char *name);``` | search group database for a name
```__device__ struct group *getgrent();``` | get the group database entry
```__device__ void endgrent();``` | close the group database
```#define setgrent``` | setgrent - reset group database to first entry
