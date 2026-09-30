#include <ext/pipeline.h>
#include <sys/statcu.h>
#include <stdiocu.h>
#include <stringcu.h>

__device__ int d_dcmp_rc;
__global__ void g_dcmp(pipelineRedir redir, char *str, char *str2) {
	struct stat statbuf1;
	if (stat(str, &statbuf1) < 0) {
		perror(str);
		d_dcmp_rc = 2;
		return;
	}
	struct stat statbuf2;
	if (stat(str2, &statbuf2) < 0) {
		perror(str2);
		d_dcmp_rc = 2;
		return;
	}
	if (statbuf1.st_dev == statbuf2.st_dev && statbuf1.st_ino == statbuf2.st_ino) {
		printf("Files are links to each other\n");
		d_dcmp_rc = 0;
		return;
	}
	if (statbuf1.st_size != statbuf2.st_size) {
		printf("Files are different sizes\n");
		d_dcmp_rc = 1;
		return;
	}
	FILE *f1 = fopen(str, "r");
	if (!f1) {
		perror(str);
		d_dcmp_rc = 2;
		return;
	}
	FILE *f2 = fopen(str2, "r");
	if (!f2) {
		perror(str2);
		fclose(f1);
		d_dcmp_rc = 2;
		return;
	}
	//
	long pos = 0;
	char buf1[512];
	char buf2[512];
	char *bp1;
	char *bp2;
	while (true) {
		size_t cc1 = fread(buf1, 1, sizeof(buf1), f1);
		if (cc1 == (size_t)-1) {
			perror(str);
			goto eof;
		}
		size_t cc2 = fread(buf2, 1, sizeof(buf2), f2);
		if (cc2 == (size_t)-1) {
			perror(str2);
			goto differ;
		}
		if (cc1 == 0 && cc2 == 0) {
			printf("Files are identical\n");
			goto same;
		}
		if (cc1 < cc2) {
			printf("First file is shorter than second\n");
			goto differ;
		}
		if (cc1 > cc2) {
			printf("Second file is shorter than first\n");
			goto differ;
		}
		if (!memcmp(buf1, buf2, cc1)) {
			pos += (long)cc1;
			continue;
		}
		//
		bp1 = buf1;
		bp2 = buf2;
		while (*bp1++ == *bp2++)
			pos++;
		printf("Files differ at byte position %ld\n", pos);
		goto differ;
	}
eof:
	fclose(f1);
	fclose(f2);
	d_dcmp_rc = 2;
	return;
same:
	fclose(f1);
	fclose(f2);
	d_dcmp_rc = 0;
	return;
differ:
	fclose(f1);
	fclose(f2);
	d_dcmp_rc = 1;
	return;
}
int dcmp(pipelineRedir redir, char *str, char *str2) {
	pipelineOpen(redir);
	char *d_str;
	char *d_str2;
	if (str) {
		size_t strLength = strlen(str) + 1;
		cudaMalloc(&d_str, strLength);
		cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	}
	else d_str = 0;
	if (str2) {
		size_t str2Length = strlen(str2) + 1;
		cudaMalloc(&d_str2, str2Length);
		cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	}
	else d_str2 = 0;
	g_dcmp<<<1, 1>>>(redir, d_str, d_str2);
	if (d_str) cudaFree(d_str);
	if (d_str2) cudaFree(d_str2);
	pipelineClose(redir);
	int rc; cudaMemcpyFromSymbol(&rc, d_dcmp_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
