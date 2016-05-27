#include "SparseVec.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define BUF_SZ 1000
int check_sorted(SV sv)
{
	int i, current_idx;
	if(sv.nnz == 0)
		return 1;
	current_idx = -1;
	for(i = 0; i < sv.nnz; i++)
	{
		if(sv.data[i].idx <= current_idx)
			return 0;
		else
			current_idx = sv.data[i].idx;
	}
	return 1;
}

int check_range(SV sv, int d)
{
	if(sv.nnz == 0)
		return 1;
	if(sv.data[0].idx < 0)
		return 0;
	if(sv.data[sv.nnz - 1].idx >= d)
		return 0;
	return 1;
}

//Initilization
void init_SparseVec(SV* psv)
{
	//psv -> length = 0;
	psv -> nnz = 0;
	psv -> data = NULL;
}

void free_SparseVec(SV* psv)
{
	free(psv -> data);
}

//Read something like "5:78.3\0"
NZE read_NZE(char* str)
{
	int i, t;
	char buffer[512];
	NZE nze;
	i = 0;
	t = 0;
	while(str[i] != ':')
	{
		buffer[t] = str[i];
		i++;
		t++;
	}
	buffer[t] = '\0';
	nze.idx = atoi(buffer);
	t = 0;
	i++;

	while(str[i] != '\0' && str[i] != '\n')
	{
		buffer[t] = str[i];
		i++;
		t++;
	}
	buffer[t] = '\0';
	nze.val = atof(buffer);
	return nze;
}

void print_NZE(NZE nze)
{
	printf("%d:%f", nze.idx, nze.val);
}

void print_SparseVec(SV sv)
{
	int i;
	for(i = 0; i < sv.nnz; i++)
	{
		print_NZE(sv.data[i]);
		if(i != sv.nnz - 1)
			putchar(' ');
		else
			putchar('\n');
	}
}

//read something like "3:12.1 5:54.1 6:78.4\0"
SV read_SparseVec(char* str)
{
	//printf("%s\n", str);
	SV sv;
	init_SparseVec(&sv);

	if(str[0] == '\0')
		return sv;
	int i = 0;
	//Determine nnz first.
	sv.nnz = 0;
	while(str[i] != '\0' && str[i] != '\n')
	{
		if(str[i] == ' ')
			sv.nnz++;
		i++;
	}
	sv.nnz++;

	//printf("%d\n", sv.nnz);
	sv.data = (NZE*) malloc(sv.nnz * sizeof(NZE));

	int t = 0, s = 0;
	i = 0;

	char buffer[512];

	//Start parsing
	while(str[i] != '\0' && str[i] != '\n')
	{
		//putchar(str[i]);
		if(str[i] != ' ')
		{
			buffer[t] = str[i];
			t++;
			i++;
		}
		else
		{
			buffer[t] = '\0';
			sv.data[s] = read_NZE(buffer);
			//printf("%s\n", buffer);
			s++;
			t = 0;
			i++;
		}
	}
	//Process the last one
	buffer[t] = '\0';
	sv.data[s] = read_NZE(buffer);

	return sv;
}

void concat_SV(SV* sv1, SV* sv2, int l1, int l2, SV* dest)
{
	int i;
	NZE* temp;
	(dest -> nnz) = (sv1 -> nnz) + (sv2 -> nnz);
	temp = realloc(dest -> data, (dest -> nnz) * sizeof(NZE));
	if(temp == NULL)
	{
		fprintf(stderr, "Error in realloc.\n");
		exit(1);
	}
	(dest -> data) = temp;
	int t = 0;
	for(i = 0; i < sv1 -> nnz; i++)
	{
		(dest -> data)[t] = (sv1 -> data)[i];
		t++;
	}

	for(i = 0; i < sv2 -> nnz; i++)
	{
		(dest -> data)[t] = (sv2 -> data)[i];
		(dest -> data)[t].idx += l1;
		t++;
	}
}

void copy_SV(SV* dest, SV* src)
{
	int i;
	dest ->  nnz = src -> nnz;
	NZE* temp = realloc(dest -> data, (dest -> nnz) * sizeof(NZE));
	if(temp == NULL)
	{
		fprintf(stderr, "Error in realloc.\n");
		exit(1);
	}
	(dest -> data) = temp;
	for(i = 0; i < dest -> nnz; i++)
		(dest -> data)[i] = (src -> data)[i];
}

void Mat_mult_SV(double** M, SV* sv, double* result, int m, int n)
{
	int i, j;
	memset(result, 0, m * sizeof(double));
	for(i = 0; i < m; i++)
		for(j = 0; j < sv -> nnz; j++)
			result[i] += M[i][(sv -> data)[j].idx] * (sv -> data)[j].val;
}

double Vec_inner_SV(double* vec, SV* sv)
{
	int i;
	double to_return = 0.0;
	for(i = 0; i < sv -> nnz; i++)
		to_return += ((sv -> data)[i]).val * vec[((sv -> data)[i]).idx];
	return to_return;
}

void Update_with_SV(double* vec, SV* sv, double c)
{
	int i;
	for(i = 0; i < sv -> nnz; i++)
		vec[((sv -> data)[i]).idx] += c * ((sv -> data)[i]).val;
}
