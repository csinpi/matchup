#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "NeuralLayer.h"
#include "LogisticEmbed_common.h"
#include "SparseVec.h"
//Have to manually link to other layers
FCL create_FCL(int input_d, int output_d, int activation_fun)
{
	FCL fcl;
	fcl.input_d = input_d;
	fcl.output_d = output_d;
	fcl.activation_fun = activation_fun;
	fcl.M = randarray_pm(output_d, input_d, 1);
	//fcl.M = zerosarray(output_d, input_d);
	fcl.M_grad = zerosarray(output_d, input_d);
	fcl.input_reg = (double*)malloc(input_d * sizeof(double));
	fcl.output = (double*)malloc(output_d * sizeof(double));
	init_SparseVec(&(fcl.input_sp));
	return fcl;
}

void free_FCL(FCL* p)
{
	Array2Dfree(p -> M, p -> output_d, p -> input_d);
	Array2Dfree(p -> M_grad, p -> output_d, p -> input_d);
	free(p -> input_reg);
	free(p -> output);
	free_SparseVec(&(p -> input_sp));
}

void forward_activate_reg(FCL* p)
{
	int i;
	//Do matrix multiplication
	for(i = 0; i < p -> output_d; i++)
	{
		(p -> output)[i] = innerprod(p -> input_reg, (p -> M)[i], p -> input_d);
		if(p -> activation_fun == SIGMOID)
			(p -> output)[i] = sigmoid((p -> output)[i]);
		else if(p -> activation_fun == TANH) 
			(p -> output)[i] = tanh((p -> output)[i]);
	}
}

void forward_activate_sparse(FCL* p)
{
	int i, j;
	memset(p -> output, 0, (p -> output_d) * sizeof(double));
	for(i = 0; i < (p -> output_d); i++)
		for(j = 0; j < (p -> input_sp).nnz; j++)
			(p -> output)[i] += (p -> M)[i][((p -> input_sp).data)[j].idx] * ((p -> input_sp).data)[j].val;
	for(i = 0; i < p -> output_d; i++)
	{
		if(p -> activation_fun == SIGMOID)
			(p -> output)[i] = sigmoid((p -> output)[i]);
		else if(p -> activation_fun == TANH) 
			(p -> output)[i] = tanh((p -> output)[i]);
	}
}

void backward_gradient_reg(FCL* p, double eta, int is_last_layer)
{
	int i, j;
	//Reverse the activation function first
	for(i = 0; i < p -> output_d; i++)
	{
		if(p -> activation_fun == SIGMOID)
			(p -> output)[i] = (p -> output)[i] * (1 - (p -> output)[i]);
		else if(p -> activation_fun == TANH) 
			(p -> output)[i] = 1 - pow((p -> output)[i], 2.0);
	}

	//Compute the gradient for matrix M
	set_zerosarray(p -> M_grad, p -> output_d, p -> input_d);
	for(i = 0; i < p -> output_d; i++)
		add_vec((p -> M_grad)[i], p -> input_reg,  p -> input_d, (p -> output)[i]);

	//Pass gradient back to input layer
	if(!is_last_layer)
	{
		memset(p -> input_reg, 0, (p -> input_d) * sizeof(double));
		for(i = 0; i < p -> output_d; i++)
			for(j = 0; j < p -> input_d; j++)
				(p -> input_reg)[j] += (p -> M)[i][j] * (p -> output)[i];
	}

	//Add gradient to M, with learning rate eta 
	add_mat(p -> M, p -> M_grad, p -> output_d, p -> input_d, eta);
}

void backward_gradient_sparse(FCL* p, double eta, int is_last_layer)
{
	int i, j, t;
	double v;
	//Reverse the activation function first
	for(i = 0; i < p -> output_d; i++)
	{
		if(p -> activation_fun == SIGMOID)
			(p -> output)[i] = (p -> output)[i] * (1 - (p -> output)[i]);
		else if(p -> activation_fun == TANH) 
			(p -> output)[i] = 1 - pow((p -> output)[i], 2.0);
	}

	//Compute the gradient for matrix M
	set_zerosarray(p -> M_grad, p -> output_d, p -> input_d);
	for(i = 0; i < p -> output_d; i++)
	{
		for(j = 0; j < (p -> input_sp).nnz; j++)
		{
			t = ((p -> input_sp).data)[j].idx;
			v = ((p -> input_sp).data)[j].val;
			(p -> M_grad)[i][t] += v * (p -> output)[i];
		}
	}

	//Pass gradient back to input layer
	if(!is_last_layer)
	{
		for(j = 0; j < (p -> input_sp).nnz; j++)
			(p -> input_sp).data[j].val = 0.0;
		for(i = 0; i < p -> output_d; i++)
		{
			for(j = 0; j < (p -> input_sp).nnz; j++)
			{
				t = ((p -> input_sp).data)[j].idx;
				(p -> input_reg)[t] += (p -> M)[i][t] * (p -> output)[i];
			}
		}
	}

	//Add gradient to M, with learning rate eta 
	add_mat(p -> M, p -> M_grad, p -> output_d, p -> input_d, eta);
}

//BCL create_BCL(int d, int type)
BCL create_BCL(int d, int type, double* blade_w, double* blade_l, double* chest_w, double* chest_l, double* blade_g, double* chest_g)
{
	BCL bcl;
	bcl.d = d;
	bcl.type = type;

	bcl.blade_w = blade_w;
	bcl.blade_l = blade_l;
	bcl.chest_w = chest_w;
	bcl.chest_l = chest_l;

	bcl.blade_g = blade_g; //Only for split model
	bcl.chest_g = chest_g;

	//bcl.blade_w = (double*)malloc(d * sizeof(double));
	//bcl.blade_l = (double*)malloc(d * sizeof(double));
	//bcl.chest_w = (double*)malloc(d * sizeof(double));
	//bcl.chest_l = (double*)malloc(d * sizeof(double));

	bcl.grad_blade_w = (double*)malloc(d * sizeof(double));
	bcl.grad_blade_l = (double*)malloc(d * sizeof(double));
	bcl.grad_chest_w = (double*)malloc(d * sizeof(double));
	bcl.grad_chest_l = (double*)malloc(d * sizeof(double));

	bcl.grad_blade_g = (double*)malloc(d * sizeof(double)); //Only for split model
	bcl.grad_chest_g = (double*)malloc(d * sizeof(double));

	bcl.blade_w_for_split = (double*)malloc(d * sizeof(double));
	bcl.blade_l_for_split = (double*)malloc(d * sizeof(double));
	bcl.chest_w_for_split = (double*)malloc(d * sizeof(double));
	bcl.chest_l_for_split = (double*)malloc(d * sizeof(double));

	bcl.grad_blade_w_for_split = (double*)malloc(d * sizeof(double));
	bcl.grad_blade_l_for_split = (double*)malloc(d * sizeof(double));
	bcl.grad_chest_w_for_split = (double*)malloc(d * sizeof(double));
	bcl.grad_chest_l_for_split = (double*)malloc(d * sizeof(double));

	return bcl;
}

void free_BCL(BCL* p)
{
	//free(p -> blade_w);
	//free(p -> blade_l);
	//free(p -> chest_w);
	//free(p -> chest_l);

	free(p -> grad_blade_w);
	free(p -> grad_blade_l);
	free(p -> grad_chest_w);
	free(p -> grad_chest_l);

	free(p -> grad_blade_g);
	free(p -> grad_chest_g);

	free(p -> blade_w_for_split);
	free(p -> blade_l_for_split);
	free(p -> chest_w_for_split);
	free(p -> chest_l_for_split);

	free(p -> grad_blade_w_for_split);
	free(p -> grad_blade_l_for_split);
	free(p -> grad_chest_w_for_split);
	free(p -> grad_chest_l_for_split);
}



void forward_BCL(BCL* p)
{
	if(p -> type <= 1)
		p -> prob_w =  sigmoid(innerprod(p -> blade_w, p -> chest_l, p -> d) - innerprod(p -> blade_l, p -> chest_w, p -> d));
	else
	{
		//vec_mult(double* vec1, double* vec2, double* dest, int length)
		vec_mult(p -> blade_w, p -> blade_g, p -> blade_w_for_split, p -> d);
		vec_mult(p -> blade_l, p -> blade_g, p -> blade_l_for_split, p -> d);
		vec_mult(p -> chest_w, p -> chest_g, p -> chest_w_for_split, p -> d);
		vec_mult(p -> chest_l, p -> chest_g, p -> chest_l_for_split, p -> d);
		p -> prob_w =  sigmoid(innerprod(p -> blade_w_for_split, p -> chest_l_for_split, p -> d) - innerprod(p -> blade_l_for_split, p -> chest_w_for_split, p -> d));
	}
}

void backward_BCL(BCL* p)
{
	//(p -> prob_w) = (p -> prob_w) * (1.0 - (p -> prob_w));
	(p -> prob_w) = 1.0 - (p -> prob_w);

	if(p -> type <= 1)
	{
		Veccopy(p -> chest_l, p -> grad_blade_w, p -> d);
		Veccopy(p -> chest_w, p -> grad_blade_l, p -> d);
		Veccopy(p -> blade_l, p -> grad_chest_w, p -> d);
		Veccopy(p -> blade_w, p -> grad_chest_l, p -> d);

		scale_vec(p -> grad_blade_w, p -> d, p -> prob_w);
		scale_vec(p -> grad_blade_l, p -> d, -(p -> prob_w));
		scale_vec(p -> grad_chest_w, p -> d, -(p -> prob_w));
		scale_vec(p -> grad_chest_l, p -> d, p -> prob_w);

		Veccopy(p -> grad_blade_w, p -> blade_w, p -> d);
		Veccopy(p -> grad_blade_l, p -> blade_l, p -> d);
		Veccopy(p -> grad_chest_w, p -> chest_w, p -> d);
		Veccopy(p -> grad_chest_l, p -> chest_l, p -> d);
	}
	else
	{
		//printf("In here.\n");
		Veccopy(p -> chest_l_for_split, p -> grad_blade_w_for_split, p -> d);
		Veccopy(p -> chest_w_for_split, p -> grad_blade_l_for_split, p -> d);
		Veccopy(p -> blade_l_for_split, p -> grad_chest_w_for_split, p -> d);
		Veccopy(p -> blade_w_for_split, p -> grad_chest_l_for_split, p -> d);

		scale_vec(p -> grad_blade_w_for_split, p -> d, p -> prob_w);
		scale_vec(p -> grad_blade_l_for_split, p -> d, -(p -> prob_w));
		scale_vec(p -> grad_chest_w_for_split, p -> d, -(p -> prob_w));
		scale_vec(p -> grad_chest_l_for_split, p -> d, p -> prob_w);

		Veccopy(p -> grad_blade_w_for_split, p -> blade_w_for_split, p -> d);
		Veccopy(p -> grad_blade_l_for_split, p -> blade_l_for_split, p -> d);
		Veccopy(p -> grad_chest_w_for_split, p -> chest_w_for_split, p -> d);
		Veccopy(p -> grad_chest_l_for_split, p -> chest_l_for_split, p -> d);

		vec_mult(p -> blade_w_for_split, p -> blade_g, p -> grad_blade_w, p -> d);
		vec_mult(p -> blade_l_for_split, p -> blade_g, p -> grad_blade_l, p -> d);
		vec_mult(p -> chest_w_for_split, p -> chest_g, p -> grad_chest_w, p -> d);
		vec_mult(p -> chest_l_for_split, p -> chest_g, p -> grad_chest_l, p -> d);

		double* temp = malloc(p -> d * sizeof(double));

		vec_mult(p -> blade_w_for_split, p -> blade_w, p -> grad_blade_g, p -> d);
		vec_mult(p -> blade_l_for_split, p -> blade_l, temp, p -> d);
		add_vec(p -> grad_blade_g, temp, p -> d, 1.0);
		vec_mult(p -> chest_w_for_split, p -> chest_w, p -> grad_chest_g, p -> d);
		vec_mult(p -> chest_l_for_split, p -> chest_l, temp, p -> d);
		add_vec(p -> grad_chest_g, temp, p -> d, 1.0);

		free(temp);

		Veccopy(p -> grad_blade_w, p -> blade_w, p -> d);
		Veccopy(p -> grad_blade_l, p -> blade_l, p -> d);
		Veccopy(p -> grad_chest_w, p -> chest_w, p -> d);
		Veccopy(p -> grad_chest_l, p -> chest_l, p -> d);
		Veccopy(p -> grad_blade_g, p -> blade_g, p -> d);
		Veccopy(p -> grad_chest_g, p -> chest_g, p -> d);
	}
}

//void copy_BCL(BCL* dest, BCL* src)
//{
//}

FCLM create_FCLM(int input_d, int output_d, int k, int activation_fun, double scale)
{
	FCLM fclm;
	fclm.input_d = input_d;
	fclm.output_d = output_d;
	fclm.k = k;
	fclm.activation_fun = activation_fun;
	//fclm.M = randarray_pm(output_d, input_d, 0.01);
	//fclm.M = randarray_pm(output_d, input_d, 0.1);
	fclm.M = randarray_pm(output_d, input_d, scale);
	fclm.M_grad = zerosarray(output_d, input_d);
	fclm.output = zerosarray(k, output_d);
	fclm.output_backup = zerosarray(k, output_d);
	fclm.input_sp = (SV*)malloc(k * sizeof(SV));
	int i;
	for(i = 0; i < k; i++)
		init_SparseVec(fclm.input_sp + i);
	return fclm;
}

void free_FCLM(FCLM* p)
{
	Array2Dfree(p -> M, p -> output_d, p -> input_d);
	Array2Dfree(p -> M_grad, p -> output_d, p -> input_d);
	Array2Dfree(p -> output, p -> k, p -> output_d);
	Array2Dfree(p -> output_backup, p -> k, p -> output_d);
	int i;
	for(i = 0; i < p -> k; i++)
		free_SparseVec((p -> input_sp) + i);
	free(p -> input_sp);
}

//Only for sparse vectors now, cause it's already enough for our model
void forward_activate_mult(FCLM* p)
{
	int i, j;
	//memset(p -> output, 0, (p -> output_d) * sizeof(double));
	//for(i = 0; i < (p -> output_d); i++)
	//	for(j = 0; j < (p -> input_sp).nnz; j++)
	//		(p -> output)[i] += (p -> M)[i][((p -> input_sp).data)[j].idx] * ((p -> input_sp).data)[j].val;
	for(i = 0; i < p -> k; i++)
		Mat_mult_SV(p -> M, (p -> input_sp + i), p -> output[i], p -> output_d, p -> input_d);

	activate_on_mat(p -> output, p -> activation_fun, p -> k, p -> output_d);

	Array2Dcopy(p -> output, p -> output_backup, p -> k, p -> output_d);
	//for(i = 0; i < p -> output_d; i++)
	//{
	//	if(p -> activation_fun == SIGMOID)
	//		(p -> output)[i] = sigmoid((p -> output)[i]);
	//	else if(p -> activation_fun == TANH) 
	//		(p -> output)[i] = tanh((p -> output)[i]);
	//}
}

//Also only have sparse version for now.
void backward_gradient_mult(FCLM* p, double eta, double lambda)
{
	int i, j, s, t;
	double v;
	
	//reverse_activate_on_mat(p -> output, p -> activation_fun, p -> k, p -> output_d);
	reverse_activate_on_mat(p -> output_backup, p -> activation_fun, p -> k, p -> output_d);
	for(s = 0; s < p -> k; s++)
		for(i = 0; i < p -> output_d; i++)
			(p -> output[s][i]) *= (p -> output_backup[s][i]);


	//Compute the gradient for matrix M
	set_zerosarray(p -> M_grad, p -> output_d, p -> input_d);
	for(s = 0; s < p -> k; s++)
	{
		for(i = 0; i < p -> output_d; i++)
		{
			for(j = 0; j < (p -> input_sp)[s].nnz; j++)
			{
				t = ((p -> input_sp)[s].data)[j].idx;
				v = ((p -> input_sp)[s].data)[j].val;
				(p -> M_grad)[i][t] += v * (p -> output)[s][i];
			}
		}
	}

	//Regularization update
	add_mat(p -> M_grad, p -> M, p -> output_d, p -> input_d, -lambda);

	//Add gradient to M, with learning rate eta 
	add_mat(p -> M, p -> M_grad, p -> output_d, p -> input_d, eta);
}

double squared_norm_FCLM(FCLM* p)
{
	return frob_norm(p -> M, p -> output_d, p -> input_d);
}


double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

void activate_on_mat(double** X, int activation_fun, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			if(activation_fun == SIGMOID)
				X[i][j] = sigmoid(X[i][j]);
			else if(activation_fun == TANH) 
				X[i][j] = tanh(X[i][j]);
		}
	}
}

void reverse_activate_on_mat(double** X, int activation_fun, int m, int n)
{
	int i;
	int j;
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			if(activation_fun == SIGMOID)
				X[i][j] = X[i][j] * (1 - X[i][j]);
			else if(activation_fun == TANH) 
				X[i][j] = 1 - pow(X[i][j], 2.0);
			else if(activation_fun == NOACT) 
				X[i][j] = 1.0;

		}
	}
}

double** randarray_pm(int m, int n, double c)
{
	//srand(time(NULL));
	double** X;
	int i;
	int j;
	X = (double**)malloc(m * sizeof(double*));
	for(i = 0; i < m; i++)
		X[i] = (double*)malloc(n * sizeof(double));
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			X[i][j] = ((double)rand()) / ((double) RAND_MAX);
			X[i][j] = (X[i][j] - 0.5) * 2 * c ;
		}
	}
	return X;
}
