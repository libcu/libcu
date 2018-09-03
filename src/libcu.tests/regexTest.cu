#include <stdiocu.h>
#include <regexcu.h>
#include <assert.h>

static __device__ void exact() {
	regex_t re;
	assert(!regcomp(&re, "sam", 0));

	regmatch_t pm;
	char str[128] = "onces sam lived with samle to win samile hehe sam hoho sam\0";
	int a = regexec(&re, &str[0], 1, &pm, REG_EXTENDED);
	assert(a == REG_NOERROR);

	int idx = 0; int offset = 0; int offsets[5];
	while (a == REG_NOERROR) {
		printf("%s match at %d\n", offset ? "next" : "first", offset + pm.rm_so);
		offsets[idx++] = offset + pm.rm_so;
		offset += pm.rm_eo;
		a = regexec(&re, &str[0] + offset, 1, &pm, 0);
	}
	assert(idx == 5);
	assert(offsets[0] == 6 && offsets[1] == 21 && offsets[2] == 34 && offsets[3] == 46 && offsets[4] == 55);
}

static __global__ void g_regex_test1() {
	printf("regex_test1\n");

	//// REGCOMP, REGEXEC, REGERROR, REGFREE ////
	//extern __device__ int regcomp_(regex_t *preg, const char *regex, int cflags);
	//extern __device__ int regexec_(regex_t *preg, const char *string, size_t nmatch, regmatch_t pmatch[], int eflags);
	//extern __device__ size_t regerror_(int errcode, const regex_t *preg, char *errbuf, size_t errbuf_size);
	//extern __device__ void regfree_(regex_t *preg);
	exact();
}
cudaError_t regex_test1() { g_regex_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
