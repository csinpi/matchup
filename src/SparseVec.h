#ifndef SPARSEVEC
#define SPARSEVEC
typedef struct NonezeroElement
{
    int idx; 
	double val;
}
NZE;

typedef struct SparseVec
{
    //int length; //Length of the entire non-sparse vector 
	int nnz; //number of nonezero element
	NZE* data;
}
SV;

int check_sorted(SV sv);
int check_range(SV sv, int d);
void init_SparseVec(SV* psv);
void free_SparseVec(SV* psv);
NZE read_NZE(char* str);
void print_NZE(NZE nze);
void print_SparseVec(SV sv);
SV read_SparseVec(char* str);

void concat_SV(SV* sv1, SV* sv2, int l1, int l2, SV* dest);

void copy_SV(SV* dest, SV* src);

void Mat_mult_SV(double** M, SV* sv, double* result, int m, int n);

double Vec_inner_SV(double* vec, SV* sv);

void Update_with_SV(double* vec, SV* sv, double c);



#endif
