#ifndef MATCHUP
#define MATCHUP
#include "GamePairHash.h"
#include "SparseVec.h"
#include "NeuralLayer.h"
//typedef struct Match
//{
//    int pa; //first player
//    int pb; //second player
//	int pw; //winning player
//}
//MATCH;
//
//typedef struct Matches
//{
//    int pa; //first player
//    int pb; //second player
//	int na; //Number of wins the first player gets
//	int nb; //Number of wins the second player gets
//}
//MATCHES;
//
//typedef struct GameRecords 
//{
//    int num_players;
//	int num_games;
//	MATCH* all_games;
//	int* test_mask;
//	char** all_players;
//	int with_mask;
//}
//GRECORDS;
//
//typedef struct GameEmbedding 
//{
//    int k;
//	int d;
//	int modeltype;
//	double* ranks;
//	double** tvecs;
//	double** hvecs;
//	int rankon;
//}
//GEMBEDDING;
//

typedef struct Match
{
	SV wvec;
	SV lvec;
	SV gvec;
	int tag; //indictate train, val or test.
}
MATCH;

typedef struct GameRecords 
{
    int num_p_features;
	int num_g_features;
	int num_games;
	int num_training_games;
	int num_validation_games;
	int num_testing_games;
	MATCH* all_games;
	//int* test_mask;

	//char** all_players;
	//int with_mask;
}
GRECORDS;


void free_match(MATCH* m);
void free_GRECORDS(GRECORDS* p);
MATCH read_match(char* str);
GRECORDS read_GRECORDS(char* filename, int use_rand, int seed, int verbose);
void print_match(MATCH m);
void print_GRECORDS(GRECORDS grs);
int check_GRECORDS(GRECORDS* p);
void simplify_GRECORDS(GRECORDS* p);

PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile);
double safe_log_logit(double a);
int random_in_range (unsigned int min_val, unsigned int max_val);
//double matchup_fun(GEMBEDDING gebd, int a, int b, int modeltype);
double vec_diff(const double* x, const double* y, int length);
double logistic_fun(double a);
double perturb_on_even(double prob_w);

typedef struct MatchupModel
{
	int type;
	int reg_type; //Regularization type
	double lambda; //Regularization parameter
	int num_p_features; //Number of player features
	int num_g_features; //Number of game features
	FCLM blade_layer;
	FCLM chest_layer;
	FCLM blade_layer_g;
	FCLM chest_layer_g;

	BCL bcl;
}
MUM;

typedef struct TestResults
{
	double ll_train;
	double ll_val;
	double ll_test;
	double acc_train;
	double acc_val;
	double acc_test;

	double obj;

	double realn_train;
	double realn_val;
	double realn_test;
}
TR;

MUM create_MatchupModel(int type, int d, int num_p_features, int num_g_features, int activation_fun, int reg_type, double lambda, double fclm_scale);
void free_MatchupModel(MUM* p);
void copy_MatchupModel(MUM* dest, MUM* src);
double feed_MatchupModel(MUM* p_mum, SV* p_wvec, SV* p_lvec, SV* p_gvec, int do_backward, double eta);
void one_pass(MUM* p_mum, GRECORDS grs, double eta, double only_check, TR* p_tr);
void print_test_result(TR tr, char* aux_str);
void print_training_paras(PARAS myparas);

//Baselines
void BT_rank_model(GRECORDS grs, double eta, double lambda, double eps, TR* p_tr);
void naive_history_model(GRECORDS grs, TR* p_tr);
void logistic_rank_model(GRECORDS grs, double eta, double lambda, double eps, TR* p_tr, int type);

#endif
