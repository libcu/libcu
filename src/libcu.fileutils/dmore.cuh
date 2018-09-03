#include <sys/statcu.h>
#include <stdiocu.h>
#include <unistdcu.h>
#include <fcntlcu.h>

__device__ int d_dmore_rc;
__global__ void g_dmore(char *name, int fd)
{
	if (!name) {
		close(fd);
		d_dmore_rc = -1;
		return;
	}
	else if (fd == -1) {
		fd = open(name, O_RDONLY);
		if (fd == -1) {
			perror(name);
			d_dmore_rc = -1;
			return;
		}
		printf("<< %s >>\b", name);
	}
	d_dmore_rc = fd;
	unsigned char ch;
	while (fd > -1 && (read(fd, &ch, 1))) {
		int line = 1;
		int col = 0;
		switch (ch) {
		case '\r': col = 0; break;
		case '\n': line++; col = 0; break;
		case '\t': col = ((col + 1) | 0x07) + 1; break;
		case '\b': if (col > 0) col--; break;
		default: col++;
		}
		putchar(ch);
		if (col >= 80) {
			col -= 80;
			line++;
		}
		if (line < 24)
			continue;
		if (col > 0)
			putchar('\n');
		printf("--More--");
		fflush(stdout);
		return;
	}
	if (fd)
		close(fd);
	d_dmore_rc = -1;
}
int dmore(char *str, int fd)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dmore<<<1,1>>>(d_str, fd);
	cudaFree(d_str);
	int rc; cudaMemcpyFromSymbol(&rc, d_dmore_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
