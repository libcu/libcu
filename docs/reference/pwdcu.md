## #include <pwdcu.h>

Also includes:
```
#include <pwd.h>
```

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ struct passwd *getpwuid(uid_t uid);``` | search user database for a user ID
```__device__ struct passwd *getpwnam(const char *name);``` | search user database for a name
```__device__ struct passwd *getpwent();``` | get user database entry
```__device__ void endpwent();``` | close the user database
```#define setpwent``` | setpwent - reset user database to first entry
