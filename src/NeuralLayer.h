#ifndef NEURALLAYER
#define NEURALLAYER
#define NOACT 0  
#define SIGMOID 1  
#define TANH 2  
#include "SparseVec.h"

typedef struct FullyConnectedLayer
{
	int input_d;
	int output_d;
	int activation_fun;
	double** M; 
	double** M_grad; 
	void* upper; //pointer to upper layer
	void* lower; //pointer to lower layer
	double* input_reg;
	SV input_sp;
	double* output; //These two vectors stores both intermediate values and gradient values
}
FCL;

FCL create_FCL(int input_d, int output_d, int activation_fun);
void free_FCL(FCL* p);
void forward_activate_reg(FCL* p);
void forward_activate_sparse(FCL* p);
void backward_gradient_reg(FCL* p, double eta, int is_last_layer);
void backward_gradient_sparse(FCL* p, double eta, int is_last_layer);

typedef struct BladeChestLayer
{
	int d;
	int type;
	double prob_w; // The probability of winner wins
	double* blade_w;
	double* blade_l;
	double* chest_w;
	double* chest_l;

	double* blade_g;
	double* chest_g; //Only for split model

	double* grad_blade_w; // Contains temporary gradients
	double* grad_blade_l;
	double* grad_chest_w;
	double* grad_chest_l;

	double* grad_blade_g;
	double* grad_chest_g; //Only for split model

	double* blade_w_for_split;
	double* blade_l_for_split;
	double* chest_w_for_split;
	double* chest_l_for_split;

	double* grad_blade_w_for_split;
	double* grad_blade_l_for_split;
	double* grad_chest_w_for_split;
	double* grad_chest_l_for_split;
}
BCL;

BCL create_BCL(int d, int type, double* blade_w, double* blade_l, double* chest_w, double* chest_l, double* blade_g, double* chest_g);
void free_BCL(BCL* p);
void forward_BCL(BCL* p);
void backward_BCL(BCL* p);
//void copy_BCL(BCL* dest, BCL* src);



typedef struct FullyConnectedLayer_mult
{
	int input_d;
	int output_d;
	int k; //number of input/output vectors
	int activation_fun;
	double** M; 
	double** M_grad; 
	SV* input_sp; //Multiple sparse vectors as input. Two for blade-chest model 
	double** output; //Multiple dense vectors as output
	double** output_backup; //Basically copy output. Useful when doing backprop.

}
FCLM;

FCLM create_FCLM(int input_d, int output_d, int k, int activation_fun, double scale);
void free_FCLM(FCLM* p);
void forward_activate_mult(FCLM* p);
void backward_gradient_mult(FCLM* p, double eta, double lambda);
double squared_norm_FCLM(FCLM* p);

double sigmoid(double x);
void activate_on_mat(double** X, int activation_fun, int m, int n);
void reverse_activate_on_mat(double** X, int activation_fun, int m, int n);
double** randarray_pm(int m, int n, double c);
#endif
