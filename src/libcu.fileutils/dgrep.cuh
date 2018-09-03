#include <stringcu.h>
#include <stdiocu.h>
#include <ctypecu.h>

// See if the specified word is found in the specified string.
__device__ bool search(char *string, char *word, bool ignoreCase)
{
	int len = strlen(word);
	if (!ignoreCase) {
		while (true) {
			string = strchr(string, word[0]);
			if (!string)
				return false;
			if (!memcmp(string, word, len))
				return true;
			string++;
		}
	}
	// Here if we need to check case independence. Do the search by lower casing both strings.
	int lowfirst = *word;
	if (isupper(lowfirst))
		lowfirst = tolower(lowfirst);
	while (true) {
		while (*string && *string != lowfirst && (!isupper(*string) || tolower(*string) != lowfirst))
			string++;
		if (*string == '\0')
			return false;
		char *cp1 = string;
		char *cp2 = word;
		int	ch1, ch2;
		do {
			if (*cp2 == '\0')
				return true;
			ch1 = *cp1++;
			if (isupper(ch1))
				ch1 = tolower(ch1);
			ch2 = *cp2++;
			if (isupper(ch2))
				ch2 = tolower(ch2);
		} while (ch1 == ch2);
		string++;
	}
}

__device__ int d_dgrep_rc;
__global__ void g_dgrep(char *name, char *word, bool ignoreCase, bool tellName, bool tellLine)
{
	FILE *f = fopen(name, "r");
	if (!f) {
		perror(name);
		d_dgrep_rc = 0;
		return;
	}
	long line = 0;
	char buf[8192];
	while (fgets(buf, sizeof(buf), f)) {
		char *cp = &buf[strlen(buf) - 1];
		if (*cp != '\n')
			printf("%s: Line too long\n", name);
		if (search(buf, word, ignoreCase)) {
			if (tellName) printf("%s: ", name);
			if (tellLine) printf("%d: ", line);
			fputs(buf, stdout);
		}
	}
	if (ferror(f))
		perror(name);
	fclose(f);
	d_dgrep_rc = 1;
}

int dgrep(char *str, char *str2, bool ignoreCase, bool tellName, bool tellLine)
{
	size_t strLength = strlen(str) + 1;
	size_t str2Length = strlen(str2) + 1;
	char *d_str;
	char *d_str2;
	cudaMalloc(&d_str, strLength);
	cudaMalloc(&d_str2, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	g_dgrep<<<1,1>>>(d_str, d_str2, ignoreCase, tellName, tellLine);
	cudaFree(d_str);
	cudaFree(d_str2);
	int rc; cudaMemcpyFromSymbol(&rc, d_dgrep_rc, sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}
