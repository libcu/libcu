#include <stdiocu.h>
#include <sentinel.h>
#include <sentinel-hostmsg.h>
#include <assert.h>

//static __forceinline__ int getprocessid_() { host_getprocessid msg; return msg.RC; }
//getprocessid_();

static __global__ void g_host_test1() {
	printf("host_test1\n");

	
}
cudaError_t host_test1() { g_host_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
