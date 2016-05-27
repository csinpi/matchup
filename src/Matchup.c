#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include "LogisticEmbed.h"
#include "HLogisticEmbed.h"
#include "Matchup.h"
#include "LogisticEmbed_common.h"
#include "PairHashTable.h"
#include "TransitionTable.h"
#include "EmbedIO.h"
#include "GamePairHash.h"
#include "SparseVec.h"
#include "NeuralLayer.h"
#define INT_BUF_SZ 200000
#define DOUBLE_BUF_SZ 2000
#define FOR_NONE -1
#define FOR_TRAINING 0
#define FOR_VALIDATION 1
#define FOR_TESTING 2
#define FOR_STRING_SIZE 20

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

char trainfile[200];
char testfile[200];
char embeddingfile[200];
char matchupmatfile[200];



void free_match(MATCH* m)
{
	free_SparseVec(&(m -> wvec));
	free_SparseVec(&(m -> lvec));
	free_SparseVec(&(m -> gvec));
}

void free_GRECORDS(GRECORDS* p)
{
	int i;
	for(i = 0; i < p -> num_games; i++)
		free_match((p -> all_games) + i);
}

MATCH read_match(char* str)
{
	MATCH m;

	//Figure out what this game is for
	int counter = 0;
	if(str[0] != 'F')
		m.tag = FOR_NONE;
	else
	{
		char for_string[FOR_STRING_SIZE];
		while(str[counter] != ' ')
			counter++;
		memcpy(for_string, str, counter * sizeof(char));
		for_string[counter] = '\0';
		if(strcmp("FOR_TRAINING", for_string) == 0)
			m.tag = FOR_TRAINING;
		else if(strcmp("FOR_VALIDATION", for_string) == 0)
			m.tag = FOR_VALIDATION;
		else
			m.tag = FOR_TESTING;
		counter++;//Making sure it's pointing at some meaningful char for the following read
		//printf("%s", str);
		//printf("%s, %d\n", for_string, m.tag);
	}
	
	int t = 0;
	char buffer[1024];

	while(str[counter] != '|' && str[counter] != '\n' && str[counter] != '\0')
		buffer[t++] = str[counter++];
	buffer[t] = '\0';
	m.gvec =  read_SparseVec(buffer);
	t = 0;
	counter++;

	while(str[counter] != '|' && str[counter] != '\n' && str[counter] != '\0')
		buffer[t++] = str[counter++];
	buffer[t] = '\0';
	m.wvec =  read_SparseVec(buffer);
	t = 0;
	counter++;

	while(str[counter] != '|' && str[counter] != '\n' && str[counter] != '\0')
		buffer[t++] = str[counter++];
	buffer[t] = '\0';
	m.lvec =  read_SparseVec(buffer);
	t = 0;

	return m;
}

//See if all the sparse vectors are ordered. Also see if the indices are within range.
int check_GRECORDS(GRECORDS* p)
{
	int i;
	for(i = 0; i < p -> num_games; i++)
	{
		if(!check_sorted((p -> all_games + i) -> wvec))
			return 0;
		if(!check_sorted((p -> all_games + i) -> lvec))
			return 0;
		if(!check_sorted((p -> all_games + i) -> gvec))
			return 0;
		if(!check_range((p -> all_games + i) -> wvec, p -> num_p_features))
			return 0;
		if(!check_range((p -> all_games + i) -> lvec, p -> num_p_features))
			return 0;
		if(!check_range((p -> all_games + i) -> gvec, p -> num_g_features))
			return 0;
	}
	return 1;
}


GRECORDS read_GRECORDS(char* filename, int use_rand, int seed, int verbose)
{
	char templine[INT_BUF_SZ];
	FILE* fp = fopen(filename, "r");
	GRECORDS grs;

	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_games = extract_tail_int(templine, "NumGames: ");

	grs.all_games = (MATCH*)malloc(grs.num_games * sizeof(MATCH));

	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_training_games = extract_tail_int(templine, "NumTrainingGames: ");

	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_validation_games = extract_tail_int(templine, "NumValidationGames: ");

	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_testing_games = extract_tail_int(templine, "NumTestingGames: ");

	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_g_features = extract_tail_int(templine, "NumGameFeatures: ");

	if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		grs.num_p_features = extract_tail_int(templine, "NumPlayerFeatures: ");

	int i;
	srand(seed);
	for(i = 0; i < grs.num_games; i++)
	{
		if(fgets(templine, INT_BUF_SZ, fp) != NULL)
		{
			grs.all_games[i] = read_match(templine);
			if(grs.all_games[i].tag == FOR_NONE || use_rand)
			{
				int temp = rand() % 10;
				if(temp <= 4)
				{
					grs.all_games[i].tag = FOR_TRAINING;
					//num_training_games++;
				}
				else if(temp <= 6)
					grs.all_games[i].tag = FOR_VALIDATION;
				else
					grs.all_games[i].tag = FOR_TESTING;
			}
		}
	}
	close(fp);
	return grs;
}

void print_match(MATCH m)
{
	if(m.tag == FOR_TRAINING)
		printf("for_training\n");
	else if(m.tag == FOR_VALIDATION)
		printf("for_validation\n");
	else if(m.tag == FOR_TESTING)
		printf("for_testing\n");
	else
		printf("for_none\n");

	printf("wvec ");
	print_SparseVec(m.wvec);
	printf("lvec ");
	print_SparseVec(m.lvec);
	printf("gvec ");
	print_SparseVec(m.gvec);
}

void print_GRECORDS(GRECORDS grs)
{
	printf("NumGames: %d\n", grs.num_games);
	printf("NumPlayerFeatures: %d\n", grs.num_p_features);
	printf("NumGameFeatures: %d\n", grs.num_g_features);
	int i;
	for(i = 0; i < grs.num_games; i++)
	{
		print_match(grs.all_games[i]);
		putchar('\n');
	}
}

//Remove all the additional features. Only keep the identity of the players
void simplify_GRECORDS(GRECORDS* p)
{
	
	int min_player = INT_MAX;
	int max_player = INT_MIN;

	int i, winner, loser;
	for(i = 0; i < p -> num_games; i++)
	{
		winner = (p -> all_games)[i].wvec.data[0].idx;
		loser = (p -> all_games)[i].lvec.data[0].idx;

		assert((p -> all_games)[i].wvec.data[0].val = 1.0);
		assert((p -> all_games)[i].lvec.data[0].val = 1.0);

		min_player = min(winner, min(loser, min_player));
		max_player = max(winner, max(loser, max_player));
	}

	//printf("min %d max %d\n", min_player, max_player);
	p -> num_p_features = max_player + 1;
	p -> num_g_features = 0;
	for(i = 0; i < p -> num_games; i++)
	{
		//Clear all game vectors
		(p -> all_games)[i].gvec.nnz = 0;
		free((p -> all_games)[i].gvec.data);
		(p -> all_games)[i].gvec.data = NULL;
		
		//Only keep the first nnz of player vectors
		(p -> all_games)[i].wvec.nnz = 1;
		(p -> all_games)[i].wvec.data = (NZE*)realloc((p -> all_games)[i].wvec.data, sizeof(NZE));
		(p -> all_games)[i].lvec.nnz = 1;
		(p -> all_games)[i].lvec.data = (NZE*)realloc((p -> all_games)[i].lvec.data, sizeof(NZE));
	}
}








//return -log(1 + exp(a))
double safe_log_logit(double a)
{
	if(a < 0.0)
		return -log(1.0 + exp(a));
	else
		return -a - log(1.0 + exp(-a));
}

double perturb_on_even(double prob_w)
{
	if(prob_w == 0.5)
	{
		float r = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
		r = r / fabs(r);
		prob_w += 1e-7 * r;
	}
	return prob_w;
}

int main(int argc, char* argv[])
{
	srand(0);
	//char str1[20] = "3:12.1";
	//NZE nze = read_NZE(str1);
	//print_NZE(nze);
	//putchar('\n');
	//char str2[1000] = "3:12.1 5:54.1 6:78.4";
	//SV sv = read_SparseVec(str2);
	//print_SparseVec(sv);
	//free_SparseVec(&sv);

	//char str3[1000] = "FOR_TESTING 3:12.1 5:54.1 6:78.4|5:4 6:1.8|7:1.345";
	//MATCH m = read_match(str3);

	//putchar('\n');
	//print_match(m);

	//GRECORDS grs = read_GRECORDS("../data/starcraft/starcraft.txt", 0);

	//print_GRECORDS(grs);
	//free_GRECORDS(&grs);

	PARAS myparas = parse_paras(argc, argv, trainfile, embeddingfile);
	GRECORDS grs = read_GRECORDS(trainfile, myparas.use_rand, myparas.seed, 0);

	if(!check_GRECORDS(&grs))
	{
		fprintf(stderr, "Inconsistent game records\n");
		exit(0);
	}

	if(myparas.featureless)
		simplify_GRECORDS(&grs);
	//print_GRECORDS(grs);

	if(myparas.baseline_only)
		goto baseline;

	int i;
	SV temp_sv;



	//exit(0);

	//for(i = 0; i < grs.num_games; i++)
	//{
	//	printf("%d\n", grs.all_games[i].tag);
	//	print_SparseVec(grs.all_games[i].gvec);
	//	print_SparseVec(grs.all_games[i].wvec);
	//	print_SparseVec(grs.all_games[i].lvec);
	//}


	//int num_hidden_node = 10;

	//FCL lower_layer = create_FCL(2 * grs.num_p_features, num_hidden_node, TANH);
	//FCL upper_layer = create_FCL(num_hidden_node, 1, SIGMOID);

	//FCL one_layer = create_FCL(2 * grs.num_p_features, 1, SIGMOID);

	//init_SparseVec(&temp_sv);
	//for(i = 0; i < grs.num_games; i++)
	//{
	//	concat_SV(&(grs.all_games[i].wvec), &(grs.all_games[i].lvec), grs.num_p_features, grs.num_p_features, &temp_sv);
	//	copy_SV(&(lower_layer.input_sp), &temp_sv);
	//	forward_activate_sparse(&lower_layer);
	//	memcpy(upper_layer.input_reg, lower_layer.output, num_hidden_node * sizeof(double));
	//	forward_activate_reg(&upper_layer);

	//	printf("%f\n", upper_layer.output[0]);
	//	upper_layer.output[0] = -0.001;
	//	backward_gradient_reg(&upper_layer, 1.0, 0);
	//	memcpy(lower_layer.output, upper_layer.input_reg, num_hidden_node * sizeof(double));
	//	backward_gradient_sparse(&lower_layer, 1.0, 1);
	//}
	//free_FCL(&lower_layer);
	//free_FCL(&upper_layer);
	//free_FCL(&one_layer);
	
	//int d = myparas.d;
	//double eta = myparas.eta;

	TR tr, tr_best, tr_last;

	//int act_type = NOACT;
	//int act_type = TANH;

	MUM mum = create_MatchupModel(myparas.modeltype, myparas.d, grs.num_p_features, grs.num_g_features, myparas.activation_function, 0, myparas.lambda, myparas.fclm_scale);
	MUM mum_last = create_MatchupModel(myparas.modeltype, myparas.d, grs.num_p_features, grs.num_g_features, myparas.activation_function, 0, myparas.lambda, myparas.fclm_scale);
	MUM mum_best = create_MatchupModel(myparas.modeltype, myparas.d, grs.num_p_features, grs.num_g_features, myparas.activation_function, 0, myparas.lambda, myparas.fclm_scale);


	int t = 0;

	one_pass(&mum, grs, myparas.eta, 1, &tr);
	tr_best = tr;

	int time_bomb = 0;
	int reboot_bomb = 0;
	int best_iter = 0;

	double* last_n_results = calloc(myparas.num_llhood_track, sizeof(double));
	int result_track_index = 0;

	for (i = 0; i < myparas.num_llhood_track; i++)
		last_n_results[i] = 1.0;

	print_training_paras(myparas);
	
	while(++t < myparas.max_iter)
	{
		copy_MatchupModel(&mum_last, &mum);
		tr_last = tr;
		printf("Iteration %d...\n", t);

		one_pass(&mum, grs, myparas.eta, 0, &tr);
		print_test_result(tr, "Ours");

		//Check convergence and abnomality
		//if(isnan(tr.obj))
		//	break;

		if(isnan(tr.obj) || tr.obj < tr_last.obj)
		{
			copy_MatchupModel(&mum, &mum_last);
			tr = tr_last;
			myparas.eta /= myparas.beta;
			printf("Rebooting... Reducing eta to %f\n\n", myparas.eta);
			reboot_bomb++;
			if(reboot_bomb > myparas.bomb_thresh)
				break;
			else
				continue;
		}
		else
			reboot_bomb = 0;

		if(tr.obj > tr_last.obj && fabs(tr.obj - tr_last.obj) / fabs(tr_last.obj) < 0.01)
		{
			myparas.eta *= myparas.alpha;
			printf("Increasing eta to %f\n\n", myparas.eta);
		}

		if(tr.obj > tr_best.obj) 
		{
			tr_best = tr;
			copy_MatchupModel(&mum_best, &mum_last);
			time_bomb = 0;
			best_iter = t;
		}
		else
			time_bomb++;

		if(time_bomb > myparas.bomb_thresh)
			break;


		last_n_results[result_track_index] = tr.obj;
		double min_llhood = DBL_MAX, max_llhood = -DBL_MAX, total_llhood = 0.0;
		for(i = 0; i < myparas.num_llhood_track; i++) {
			total_llhood += last_n_results[i];
			if (last_n_results[i] < min_llhood)
				min_llhood = last_n_results[i];
			if (last_n_results[i] > max_llhood)
				max_llhood = last_n_results[i];
		}
		printf("min_obj %f, max_obj %f, gap %f\n", min_llhood, max_llhood, fabs(min_llhood - max_llhood));
		if (myparas.num_llhood_track > 0
				&& fabs(min_llhood - max_llhood) < myparas.eps
				&& t > myparas.num_llhood_track) {
			break;
		}
		result_track_index++;
		result_track_index %= myparas.num_llhood_track;
		putchar('\n');
	}
	copy_MatchupModel(&mum, &mum_best);
	one_pass(&mum, grs, myparas.eta, 1, &tr);

	printf("\nTraining Done.\n");
	printf("Best training model is from iteration %d\n", best_iter);
	printf("Final eta: %f\n", myparas.eta);

	printf("\n-------- Final test results --------\n");

	print_test_result(tr, "Ours");




	//free_BCL(&bcl);
	//free_FCLM(&blade_layer);
	//free_FCLM(&chest_layer);
	free_MatchupModel(&mum);
	free_MatchupModel(&mum_last);
	free_MatchupModel(&mum_best);

	free(last_n_results);



baseline:
	printf("\n-------- Baselines --------\n");

	TR naive_baseline_tr;
	naive_history_model(grs, &naive_baseline_tr);
	print_test_result(naive_baseline_tr, "Naive");

	putchar('\n');

	TR BT_baseline_tr;
	BT_rank_model(grs, 0.1, myparas.lambda, myparas.eps, &BT_baseline_tr);
	print_test_result(BT_baseline_tr, "BT");


	putchar('\n');
	TR logistic_baseline_tr;
	logistic_rank_model(grs, 1e-7, myparas.lambda, myparas.eps, &logistic_baseline_tr, myparas.modeltype);
	print_test_result(logistic_baseline_tr, "Logistic");

	//free_SparseVec(&temp_sv);
	free_GRECORDS(&grs);
	//printf("hard part done.\n");
	return 0;
}

PARAS parse_paras(int argc, char* argv[], char* trainfile, char* embedfile)
{
	//Default options
	PARAS myparas;
	myparas.d = 10;
	myparas.eps = 1e-4;
	myparas.eta = 1e-3;
	myparas.rankon = 1;
	myparas.lambda = 0.0;
	myparas.seed = 0;
	myparas.train_ratio = 2;
	myparas.max_iter = 1000;
	myparas.training_multiplier = 10;
	myparas.bomb_thresh = 30;
	myparas.alpha = 1.1;
	myparas.beta = 2.0;
	myparas.training_mode = 0;
	myparas.num_llhood_track = 10;
	myparas.regularization_type = 0;
	myparas.modeltype = 0;
	matchupmatfile[0] = '\0';
	myparas.eta_reduction_thresh = log(0.5);
	myparas.activation_function = NOACT;
	myparas.baseline_only = 0;
	myparas.featureless = 0;
	myparas.fclm_scale = 0.01;
	myparas.use_rand = 0;

	int i;
	for(i = 1; (i < argc) && argv[i][0] == '-'; i++)
	{
		switch(argv[i][1])
		{
			case 'd': i++; myparas.d = atoi(argv[i]); break;
			case 'e': i++; myparas.eps = atof(argv[i]); break;
			case 'i': i++; myparas.eta = atof(argv[i]); break;
			case 'r': i++; myparas.rankon = atoi(argv[i]); break;
			case 'l': i++; myparas.lambda = atof(argv[i]); break;
			case 'S': i++; myparas.seed = atof(argv[i]); break;
			case 'R': i++; myparas.train_ratio = atoi(argv[i]); break;
			case 'm': i++; myparas.max_iter = atoi(argv[i]); break;
			case 'u': i++; myparas.training_multiplier = atoi(argv[i]); break;
			case 'b': i++; myparas.bomb_thresh = atoi(argv[i]); break;
			case 'A': i++; myparas.alpha = atof(argv[i]); break;
			case 'B': i++; myparas.beta = atof(argv[i]); break;
			case 'T': i++; myparas.training_mode = atoi(argv[i]); break;
            case 'w': i++; myparas.num_llhood_track = atoi(argv[i]); break;
            case 't': i++; myparas.regularization_type = atoi(argv[i]); break;
            case 'M': i++; myparas.modeltype = atoi(argv[i]); break;
			case 'E': i++; strcpy(matchupmatfile, argv[i]); break;
			case 'F': i++;
					  //printf("%s\n", argv[i]);
					  if(strcmp("NOACT", argv[i]) == 0)
						  myparas.activation_function = NOACT;
					  else if(strcmp("SIGMOID", argv[i]) == 0)
						  myparas.activation_function = SIGMOID;
					  else if(strcmp("TANH", argv[i]) == 0)
						  myparas.activation_function = TANH;
					  else
					  {
						  fprintf(stderr, "Wrong activation function\n");
						  exit(0);
					  }
					  break;
            case 'L': i++; myparas.baseline_only = atoi(argv[i]); break;
            case 'Y': i++; myparas.featureless = atoi(argv[i]); break;
			case 's': i++; myparas.fclm_scale = atof(argv[i]); break;
			case 'k': i++; myparas.use_rand = atoi(argv[i]); break;
			//case 'n': i++; myparas.do_normalization = atoi(argv[i]); break;
			//case 't': i++; myparas.method = atoi(argv[i]); break;
			//case 'r': i++; myparas.random_init = atoi(argv[i]); break;
			//case 'd': i++; myparas.d = atoi(argv[i]); break;
			//case 'i': i++; myparas.ita = atof(argv[i]); break;
			//case 'e': i++; myparas.eps = atof(argv[i]); break;
			//case 'l': i++; myparas.lambda = atof(argv[i]); break;
			//case 'f': i++; myparas.fast_collection= atoi(argv[i]); break;
			//case 's': i++; myparas.radius= atoi(argv[i]); break;
			//case 'a': i++; myparas.alpha = atof(argv[i]); break;
			//case 'b': i++; myparas.beta = atof(argv[i]); break;
			//case 'g':
			//		  i++;
			//		  if (argv[i][1] == '\0') {
			//			  myparas.regularization_type = atoi(argv[i]);
			//			  myparas.tag_regularizer = atoi(argv[i]);
			//			  printf("Both regularizers set to %d\n", myparas.regularization_type);
			//		  }
			//		  else {
			//			  char first_reg[2] = "\0\0";
			//			  char second_reg[2] = "\0\0";
			//			  first_reg[0] = argv[i][0];
			//			  second_reg[0] = argv[i][1];

            //              myparas.regularization_type = atoi(first_reg);
            //              myparas.tag_regularizer = atoi(second_reg);
            //              printf("Song regularizer set to %d\n", myparas.regularization_type);
            //              printf("Tag regularizer set to %d\n", myparas.tag_regularizer);
            //          }                
            //          break;
            //case 'h': i++; myparas.grid_heuristic = atoi(argv[i]); break;
            //case 'm': i++; myparas.landmark_heuristic = atoi(argv[i]); break;
            //case 'p': i++; myparas.bias_enabled = atoi(argv[i]); break;
            //case 'w': i++; myparas.num_llhood_track = atoi(argv[i]); break;
            //case 'c': i++; myparas.hessian = atoi(argv[i]); break;
            //case 'x': i++; strcpy(myparas.tagfile, argv[i]); break;
            //case 'q': i++; myparas.num_landmark = atoi(argv[i]); break;
            //case 'y': i++; myparas.lowerbound_ratio = atof(argv[i]); break;
            //case 'o': i++; myparas.reboot_enabled = atoi(argv[i]); break;
            //case '0': i++; myparas.landmark_burnin_iter = atoi(argv[i]); break;
            //case 'u': i++; myparas.nu_multiplier = atof(argv[i]); break;
            //case 'k': i++; myparas.use_hash_TTable = atoi(argv[i]); break;
            //case 'T': i++; myparas.triple_dependency = atoi(argv[i]); break;
            //case 'L': i++; myparas.angle_lambda = atof(argv[i]); break;
            //case 'N': i++; myparas.transition_range = atoi(argv[i]); break;
            //case 'D': i++; myparas.num_threads = atoi(argv[i]); break;
            //case '9': i++; myparas.rand_option = atoi(argv[i]); break;
            //case 'I': i++; strcpy(myparas.init_file, argv[i]); break;
            //case 'A': i++; myparas.ALL_candidate = atoi(argv[i]); break;
            //case 'M': i++; myparas.candidate_mode = atoi(argv[i]); break;
            //case 'S': i++; myparas.candidate_length_threshold = atoi(argv[i]); break;
            //case 'H': i++; myparas.hedonic_enabled = atoi(argv[i]); break;
            default: printf("Unrecognizable option -%c\n", argv[i][1]); exit(1);
		}

	}

	if((i + 1) < argc)
	{
		strcpy(trainfile, argv[i]);
		strcpy(embedfile, argv[i + 1]);
	}
	else
	{
		printf("Not enough parameters.\n");
		exit(1);
	}

	return myparas;
}




int random_in_range (unsigned int min_val, unsigned int max_val)
{
	int base_random = rand(); /* in [0, RAND_MAX] */
	if (RAND_MAX == base_random) return random_in_range(min_val, max_val);
	/* now guaranteed to be in [0, RAND_MAX) */
	int range       = max_val - min_val,
		remainder   = RAND_MAX % range,
		bucket      = RAND_MAX / range;
	/* There are range buckets, plus one smaller interval
	 *      within remainder of RAND_MAX */
	if (base_random < RAND_MAX - remainder) {
		return min_val + base_random/bucket;
	} else {
		return random_in_range (min_val, max_val);
	}
}

//double matchup_fun(GEMBEDDING gebd, int a, int b, int modeltype)
//{
//	double mf;
//	if(gebd.d == 0) 
//		mf = 0.0;
//	else if(modeltype == 0)
//		mf = vec_diff(gebd.hvecs[b], gebd.tvecs[a], gebd.d) - vec_diff(gebd.hvecs[a], gebd.tvecs[b], gebd.d);
//	else if(modeltype == 1)
//		mf = log(vec_diff(gebd.hvecs[b], gebd.tvecs[a], gebd.d)) - log(vec_diff(gebd.hvecs[a], gebd.tvecs[b], gebd.d));
//	else
//		mf = innerprod(gebd.hvecs[a], gebd.tvecs[b], gebd.d) - innerprod(gebd.tvecs[a], gebd.hvecs[b], gebd.d);
//	if(gebd.rankon)
//		mf = mf + gebd.ranks[a] - gebd.ranks[b];
//	return mf;
//}


double vec_diff(const double* x, const double* y, int length)
{
	int i;
	double temp = 0.0;
	for(i = 0; i < length; i++)
		temp += pow(x[i] - y[i], 2);
	return temp;
}

double logistic_fun(double a)
{
	return 1.0 / (1.0 + exp(-a));
}

double matchup_matrix_recover_error(double** true_matchup_mat, double** predict_matchup_mat, int k)
{
	int i, j;
	int num_error = 0;
	int num_count = 0;
	for(i = 0; i < k; i++)
	{
		for(j = 0; j < k; j++)
		{
			if(true_matchup_mat[i][j] != 5.0 )
			{
				num_count++;
				if(sign_fun(true_matchup_mat[i][j] - 5.0) != sign_fun(predict_matchup_mat[i][j] - 5.0))
					num_error++;
			}
		}
	}
	return (double)num_error / num_count;
}

int sign_fun(double a)
{
	if(a > 0.0)
		return 1;
	else if(a < 0.0)
		return -1;
	else
		return 0;
}

MUM create_MatchupModel(int type, int d, int num_p_features, int num_g_features, int activation_fun, int reg_type, double lambda, double fclm_scale)
{
	MUM mum;
	mum.type = type;
	mum.reg_type = reg_type;
	mum.lambda = lambda;
	mum.num_p_features = num_p_features;
	mum.num_g_features = num_g_features;
	if(type == 0)
	{
		mum.blade_layer = create_FCLM(num_p_features, d, 2, activation_fun, fclm_scale);
		mum.chest_layer = create_FCLM(num_p_features, d, 2, activation_fun, fclm_scale);
		mum.bcl = create_BCL(d, type, mum.blade_layer.output[0], mum.blade_layer.output[1], mum.chest_layer.output[0], mum.chest_layer.output[1], NULL, NULL);
	}
	else if(type == 1)
	{
		mum.blade_layer = create_FCLM(num_p_features + num_g_features, d, 2, activation_fun, fclm_scale);
		mum.chest_layer = create_FCLM(num_p_features + num_g_features, d, 2, activation_fun, fclm_scale);
		mum.bcl = create_BCL(d, type, mum.blade_layer.output[0], mum.blade_layer.output[1], mum.chest_layer.output[0], mum.chest_layer.output[1], NULL, NULL);
	}
	else if(type == 2)
	{
		mum.blade_layer = create_FCLM(num_p_features, d, 2, activation_fun, fclm_scale);
		mum.chest_layer = create_FCLM(num_p_features, d, 2, activation_fun, fclm_scale);
		mum.blade_layer_g = create_FCLM(num_g_features, d, 1, activation_fun, fclm_scale);
		mum.chest_layer_g = create_FCLM(num_g_features, d, 1, activation_fun, fclm_scale);
		mum.bcl = create_BCL(d, type, mum.blade_layer.output[0], mum.blade_layer.output[1], mum.chest_layer.output[0], mum.chest_layer.output[1], mum.blade_layer_g.output[0], mum.chest_layer_g.output[0]);
		//BCL create_BCL(int d, int type, double* blade_w, double* blade_l, double* chest_w, double* chest_l, double* blade_g, double* chest_g);
	}
	return mum;
}

void free_MatchupModel(MUM* p)
{
	free_BCL(&(p -> bcl));
	free_FCLM(&(p -> blade_layer));
	free_FCLM(&(p -> chest_layer));
	if(p -> type >= 2)
	{
		free_FCLM(&(p -> blade_layer_g));
		free_FCLM(&(p -> chest_layer_g));
	}
}

//Only need to deal with meaningful parameters
void copy_MatchupModel(MUM* dest, MUM* src)
{
	Array2Dcopy((src -> blade_layer).M, (dest -> blade_layer).M, (src -> blade_layer).output_d, (src -> blade_layer).input_d);
	Array2Dcopy((src -> chest_layer).M, (dest -> chest_layer).M, (src -> chest_layer).output_d, (src -> chest_layer).input_d);
	if(src -> type >= 2)
	{
		Array2Dcopy((src -> blade_layer_g).M, (dest -> blade_layer_g).M, (src -> blade_layer_g).output_d, (src -> blade_layer_g).input_d);
		Array2Dcopy((src -> chest_layer_g).M, (dest -> chest_layer_g).M, (src -> chest_layer_g).output_d, (src -> chest_layer_g).input_d);
	}
}

double feed_MatchupModel(MUM* p_mum, SV* p_wvec, SV* p_lvec, SV* p_gvec, int do_backward, double eta)
{
	double to_return;
	if(p_mum -> type == 0)
	{
		copy_SV(&((p_mum -> blade_layer).input_sp[0]), p_wvec);
		copy_SV(&((p_mum -> blade_layer).input_sp[1]), p_lvec);
		copy_SV(&((p_mum -> chest_layer).input_sp[0]), p_wvec);
		copy_SV(&((p_mum -> chest_layer).input_sp[1]), p_lvec);

		forward_activate_mult(&(p_mum -> blade_layer));
		forward_activate_mult(&(p_mum -> chest_layer));
		forward_BCL(&(p_mum -> bcl));
		//printf("%f\n", bcl.prob_w);
		
		to_return = (p_mum -> bcl).prob_w;

		if(do_backward)
		{
			backward_BCL(&(p_mum -> bcl));
			backward_gradient_mult(&(p_mum -> blade_layer), eta, p_mum -> lambda);
			backward_gradient_mult(&(p_mum -> chest_layer), eta, p_mum -> lambda);
		}
	}
	else if(p_mum -> type == 1)
	{
		SV temp_sv;
		init_SparseVec(&temp_sv);
		//void concat_SV(SV* sv1, SV* sv2, int l1, int l2, SV* dest);
		concat_SV(p_wvec, p_gvec, p_mum -> num_p_features, p_mum -> num_g_features, &temp_sv);
		copy_SV(&((p_mum -> blade_layer).input_sp[0]), &temp_sv);
		concat_SV(p_lvec, p_gvec, p_mum -> num_p_features, p_mum -> num_g_features, &temp_sv);
		copy_SV(&((p_mum -> blade_layer).input_sp[1]), &temp_sv);
		concat_SV(p_wvec, p_gvec, p_mum -> num_p_features, p_mum -> num_g_features, &temp_sv);
		copy_SV(&((p_mum -> chest_layer).input_sp[0]), &temp_sv);
		concat_SV(p_lvec, p_gvec, p_mum -> num_p_features, p_mum -> num_g_features, &temp_sv);
		copy_SV(&((p_mum -> chest_layer).input_sp[1]), &temp_sv);
		free_SparseVec(&temp_sv);

		forward_activate_mult(&(p_mum -> blade_layer));
		forward_activate_mult(&(p_mum -> chest_layer));
		forward_BCL(&(p_mum -> bcl));
		//printf("%f\n", bcl.prob_w);
		
		to_return = (p_mum -> bcl).prob_w;

		if(do_backward)
		{
			backward_BCL(&(p_mum -> bcl));
			backward_gradient_mult(&(p_mum -> blade_layer), eta, p_mum -> lambda);
			backward_gradient_mult(&(p_mum -> chest_layer), eta, p_mum -> lambda);
		}
	}
	else if(p_mum -> type == 2)
	{
		copy_SV(&((p_mum -> blade_layer).input_sp[0]), p_wvec);
		copy_SV(&((p_mum -> blade_layer).input_sp[1]), p_lvec);
		copy_SV(&((p_mum -> chest_layer).input_sp[0]), p_wvec);
		copy_SV(&((p_mum -> chest_layer).input_sp[1]), p_lvec);

		copy_SV(&((p_mum -> blade_layer_g).input_sp[0]), p_gvec);
		copy_SV(&((p_mum -> chest_layer_g).input_sp[0]), p_gvec);

		forward_activate_mult(&(p_mum -> blade_layer));
		forward_activate_mult(&(p_mum -> chest_layer));
		forward_activate_mult(&(p_mum -> blade_layer_g));
		forward_activate_mult(&(p_mum -> chest_layer_g));
		forward_BCL(&(p_mum -> bcl));
		
		to_return = (p_mum -> bcl).prob_w;

		//printf("%f\n", to_return);

		if(do_backward)
		{
			backward_BCL(&(p_mum -> bcl));
			backward_gradient_mult(&(p_mum -> blade_layer), eta, p_mum -> lambda);
			backward_gradient_mult(&(p_mum -> chest_layer), eta, p_mum -> lambda);
			backward_gradient_mult(&(p_mum -> blade_layer_g), eta, p_mum -> lambda);
			backward_gradient_mult(&(p_mum -> chest_layer_g), eta, p_mum -> lambda);
		}
	}
	return to_return;
}

void one_pass(MUM* p_mum, GRECORDS grs, double eta, double only_check, TR* p_tr)
{
	double ll_train = 0.0;
	double ll_val = 0.0;
	double ll_test = 0.0;

	double acc_train = 0.0;
	double acc_val = 0.0;
	double acc_test = 0.0;

	double realn_train = 0.0;
	double realn_val = 0.0;
	double realn_test = 0.0;
	int i;
	double prob_w;

	for(i = 0; i < grs.num_games; i++)
	{
		if(grs.all_games[i].tag == FOR_TRAINING)
		{
			if(only_check)
				prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 0, eta);
			else
				prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 1, eta);

			//printf("prob_w = %f\n", prob_w);
			prob_w = perturb_on_even(prob_w);

			realn_train++;
			ll_train += log(prob_w);
			acc_train += (prob_w >= 0.5 ? 1.0:0.0);

		}
		else if(grs.all_games[i].tag == FOR_VALIDATION)
		{
			prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 0, eta);
			prob_w = perturb_on_even(prob_w);
			realn_val++;
			ll_val += log(prob_w);
			acc_val += (prob_w >= 0.5 ? 1.0:0.0);
		}
		else if(grs.all_games[i].tag == FOR_TESTING)
		{
			prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 0, eta);
			prob_w = perturb_on_even(prob_w);
			realn_test++;
			ll_test += log(prob_w);
			acc_test += (prob_w >= 0.5 ? 1.0:0.0);
		}
	}

	p_tr -> realn_train = realn_train;
	p_tr -> realn_val = realn_val;
	p_tr -> realn_test = realn_test;

	p_tr -> ll_train = ll_train / realn_train;
	p_tr -> acc_train = acc_train / realn_train;

	p_tr -> ll_val = ll_val / realn_val;
	p_tr -> acc_val = acc_val / realn_val;

	p_tr -> ll_test = ll_test / realn_test;
	p_tr -> acc_test = acc_test / realn_test;

	//if(p_mum -> reg_type == 0)
	//	p_tr -> obj = p_tr -> ll_train; 
	if(p_mum -> type != 2)
		p_tr -> obj = p_tr -> ll_train - 0.5 * (p_mum -> lambda) * squared_norm_FCLM(&(p_mum -> blade_layer)) - 0.5 * (p_mum -> lambda) * squared_norm_FCLM(&(p_mum -> chest_layer));
	else
		p_tr -> obj = p_tr -> ll_train - 0.5 * (p_mum -> lambda) * squared_norm_FCLM(&(p_mum -> blade_layer)) - 0.5 * (p_mum -> lambda) * squared_norm_FCLM(&(p_mum -> chest_layer)) - 0.5 * (p_mum -> lambda) * squared_norm_FCLM(&(p_mum -> blade_layer_g)) - 0.5 * (p_mum -> lambda) * squared_norm_FCLM(&(p_mum -> chest_layer_g));

}

void print_test_result(TR tr, char* aux_str)
{
	printf("%s: Objective function: %f\n", aux_str, tr.obj);

	printf("%s: Avg. Training Log-likelihood: %f\n", aux_str, tr.ll_train);
	printf("%s: Avg. Training Accuracy: %f\n", aux_str, tr.acc_train);

	printf("%s: Avg. Validation Log-likelihood: %f\n", aux_str, tr.ll_val);
	printf("%s: Avg. Validation Accuracy: %f\n", aux_str, tr.acc_val);

	printf("%s: Avg. Test Log-likelihood: %f\n", aux_str, tr.ll_test);
	printf("%s: Avg. Test Accuracy: %f\n", aux_str, tr.acc_test);
}

void print_training_paras(PARAS myparas)
{
	printf("-------- Training Parameters --------\n");
	printf("eta: %f\n", myparas.eta);
	printf("eps: %f\n", myparas.eps);
	printf("lambda: %f\n", myparas.lambda);
	printf("fclm_scale: %f\n", myparas.fclm_scale);
	printf("activation function: %d\n", myparas.activation_function);
	printf("d: %d\n", myparas.d);
	printf("-------------------------------------\n\n");
}

void BT_rank_model(GRECORDS grs, double eta, double lambda, double eps, TR* p_tr)
{
	int i;
	double prob_w;

	int min_player = INT_MAX;
	int max_player = INT_MIN;
	int winner, loser;

	for(i = 0; i < grs.num_games; i++)
	{
		//printf("%d beats %d\n", grs.all_games[i].wvec.data[0].idx, grs.all_games[i].lvec.data[0].idx);
		winner = grs.all_games[i].wvec.data[0].idx;
		loser = grs.all_games[i].lvec.data[0].idx;

		assert(grs.all_games[i].wvec.data[0].val = 1.0);
		assert(grs.all_games[i].lvec.data[0].val = 1.0);

		min_player = min(winner, min(loser, min_player));
		max_player = max(winner, max(loser, max_player));

	}
	printf("Num of players: %d\n", max_player + 1);
	double* ranks = randvec(max_player + 1, 1.0);

	double ll_last = 1.0;

	int t = 0;
	while(1)
	{
		//printf("%d\n", ++t);
		double ll_train = 0.0;
		double ll_val = 0.0;
		double ll_test = 0.0;

		double acc_train = 0.0;
		double acc_val = 0.0;
		double acc_test = 0.0;

		double realn_train = 0.0;
		double realn_val = 0.0;
		double realn_test = 0.0;

		double obj = 0.0;


		for(i = 0; i < grs.num_games; i++)
		{
			winner = grs.all_games[i].wvec.data[0].idx;
			loser = grs.all_games[i].lvec.data[0].idx;

			prob_w = logistic_fun(ranks[winner] - ranks[loser]); 
			prob_w = perturb_on_even(prob_w);

			if(grs.all_games[i].tag == FOR_TRAINING)
			{
				ranks[winner] += (1.0 - prob_w) * eta;
				ranks[loser] -= (1.0 - prob_w) * eta;
				//if(only_check)
				//	prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 0, eta);
				//else
				//	prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 1, eta);

				////printf("prob_w = %f\n", prob_w);

				//realn_train++;
				//ll_train += log(prob_w);
				//acc_train += (prob_w > 0.5 ? 1.0:0.0);
				realn_train++;
				ll_train += log(prob_w);
				acc_train += (prob_w >= 0.5 ? 1.0:0.0);


			}
			else if(grs.all_games[i].tag == FOR_VALIDATION)
			{
				//prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 0, eta);
				realn_val++;
				ll_val += log(prob_w);
				acc_val += (prob_w >= 0.5 ? 1.0:0.0);
			}
			else if(grs.all_games[i].tag == FOR_TESTING)
			{
				//prob_w = feed_MatchupModel(p_mum, &(grs.all_games[i].wvec), &(grs.all_games[i].lvec), &(grs.all_games[i].gvec), 0, eta);
				realn_test++;
				ll_test += log(prob_w);
				acc_test += (prob_w >= 0.5 ? 1.0:0.0);
			}
		}
		//Regularization
		add_vec(ranks, ranks, max_player + 1,  -lambda);




		p_tr -> realn_train = realn_train;
		p_tr -> realn_val = realn_val;
		p_tr -> realn_test = realn_test;

		p_tr -> ll_train = ll_train / realn_train;
		p_tr -> acc_train = acc_train / realn_train;

		p_tr -> ll_val = ll_val / realn_val;
		p_tr -> acc_val = acc_val / realn_val;

		p_tr -> ll_test = ll_test / realn_test;
		p_tr -> acc_test = acc_test / realn_test;

		p_tr -> obj = p_tr -> ll_train * realn_train - 0.5 * lambda * pow(vec_norm(ranks, max_player + 1), 2.0);





		//print_test_result(*p_tr, "BT_inter");

		//printf("%f, %f, %f, %f\n", fabs(p_tr -> ll_train - ll_last), eps, p_tr -> ll_train, ll_last);
		//printf("-- %f -- \n", ll_last);
		if(ll_last < 0 && fabs(p_tr -> obj - ll_last) < eps)
			break;
		ll_last =  p_tr -> obj;
		if(++t > 1000)
		   break;	
	}


	//One more time
	double ll_train = 0.0;
	double ll_val = 0.0;
	double ll_test = 0.0;

	double acc_train = 0.0;
	double acc_val = 0.0;
	double acc_test = 0.0;

	double realn_train = 0.0;
	double realn_val = 0.0;
	double realn_test = 0.0;

	double obj = 0.0;


	for(i = 0; i < grs.num_games; i++)
	{
		winner = grs.all_games[i].wvec.data[0].idx;
		loser = grs.all_games[i].lvec.data[0].idx;

		prob_w = logistic_fun(ranks[winner] - ranks[loser]); 
		prob_w = perturb_on_even(prob_w);

		if(grs.all_games[i].tag == FOR_TRAINING)
		{
			realn_train++;
			ll_train += log(prob_w);
			acc_train += (prob_w >= 0.5 ? 1.0:0.0);
		}
		else if(grs.all_games[i].tag == FOR_VALIDATION)
		{
			realn_val++;
			ll_val += log(prob_w);
			acc_val += (prob_w >= 0.5 ? 1.0:0.0);
		}
		else if(grs.all_games[i].tag == FOR_TESTING)
		{
			realn_test++;
			ll_test += log(prob_w);
			acc_test += (prob_w >= 0.5 ? 1.0:0.0);
		}
	}

	p_tr -> realn_train = realn_train;
	p_tr -> realn_val = realn_val;
	p_tr -> realn_test = realn_test;

	p_tr -> ll_train = ll_train / realn_train;
	p_tr -> acc_train = acc_train / realn_train;

	p_tr -> ll_val = ll_val / realn_val;
	p_tr -> acc_val = acc_val / realn_val;

	p_tr -> ll_test = ll_test / realn_test;
	p_tr -> acc_test = acc_test / realn_test;

	p_tr -> obj = p_tr -> ll_train * realn_train - 0.5 * lambda * pow(vec_norm(ranks, max_player + 1), 2.0);

	free(ranks);
}

void naive_history_model(GRECORDS grs, TR* p_tr)
{
	int i, j;
	double prob_w;

	int min_player = INT_MAX;
	int max_player = INT_MIN;
	int winner, loser;

	for(i = 0; i < grs.num_games; i++)
	{
		//printf("%d beats %d\n", grs.all_games[i].wvec.data[0].idx, grs.all_games[i].lvec.data[0].idx);
		winner = grs.all_games[i].wvec.data[0].idx;
		loser = grs.all_games[i].lvec.data[0].idx;
		assert(grs.all_games[i].wvec.data[0].val = 1.0);
		assert(grs.all_games[i].lvec.data[0].val = 1.0);

		min_player = min(winner, min(loser, min_player));
		max_player = max(winner, max(loser, max_player));
	}

	double** hist = zerosarray(max_player + 1, max_player + 1);
	for(i = 0; i < grs.num_games; i++)
	{
		winner = grs.all_games[i].wvec.data[0].idx;
		loser = grs.all_games[i].lvec.data[0].idx;
		if(grs.all_games[i].tag == FOR_TRAINING)
			hist[winner][loser]++;
	}

	//Necessary smoothing
	for(i = 0; i <= max_player; i++)
		for(j = 0; j <= max_player; j++)
			hist[i][j] += 1.0;

	double ll_train = 0.0;
	double ll_val = 0.0;
	double ll_test = 0.0;

	double acc_train = 0.0;
	double acc_val = 0.0;
	double acc_test = 0.0;

	double realn_train = 0.0;
	double realn_val = 0.0;
	double realn_test = 0.0;

	for(i = 0; i < grs.num_games; i++)
	{
		winner = grs.all_games[i].wvec.data[0].idx;
		loser = grs.all_games[i].lvec.data[0].idx;
		prob_w = hist[winner][loser] / (hist[winner][loser] + hist[loser][winner]);
		prob_w = perturb_on_even(prob_w);
		if(grs.all_games[i].tag == FOR_TRAINING)
		{
			realn_train++;
			ll_train += log(prob_w);
			acc_train += (prob_w >= 0.5 ? 1.0:0.0);
		}
		else if(grs.all_games[i].tag == FOR_VALIDATION)
		{
			realn_val++;
			ll_val += log(prob_w);
			acc_val += (prob_w >= 0.5 ? 1.0:0.0);
		}
		else if(grs.all_games[i].tag == FOR_TESTING)
		{
			realn_test++;
			ll_test += log(prob_w);
			acc_test += (prob_w >= 0.5 ? 1.0:0.0);
		}
	}

	p_tr -> realn_train = realn_train;
	p_tr -> realn_val = realn_val;
	p_tr -> realn_test = realn_test;

	p_tr -> ll_train = ll_train / realn_train;
	p_tr -> acc_train = acc_train / realn_train;

	p_tr -> ll_val = ll_val / realn_val;
	p_tr -> acc_val = acc_val / realn_val;

	p_tr -> ll_test = ll_test / realn_test;
	p_tr -> acc_test = acc_test / realn_test;


	Array2Dfree(hist, max_player + 1, max_player + 1);
}

void logistic_rank_model(GRECORDS grs, double eta, double lambda, double eps, TR* p_tr, int type)
{
	int i;
	double prob_w;
	double* w;
	if(type != 1)
		w = randvec(grs.num_p_features, 1e-2);
	else
		w = randvec(grs.num_p_features + grs.num_g_features, 1e-2);
	SV tempw, templ;
	init_SparseVec(&tempw);
	init_SparseVec(&templ);
	double ll_last = 1.0;
	int t = 0;
	while(1)
	{
		double ll_train = 0.0;
		double ll_val = 0.0;
		double ll_test = 0.0;

		double acc_train = 0.0;
		double acc_val = 0.0;
		double acc_test = 0.0;

		double realn_train = 0.0;
		double realn_val = 0.0;
		double realn_test = 0.0;

		double obj = 0.0;

		for(i = 0; i < grs.num_games; i++)
		{
			if(type != 1)
			{
				copy_SV(&tempw, &(grs.all_games[i].wvec));
				copy_SV(&templ, &(grs.all_games[i].lvec));
			}
			else
			{
				concat_SV(&(grs.all_games[i].wvec), &(grs.all_games[i].gvec), grs.num_p_features, grs.num_g_features, &tempw);
				concat_SV(&(grs.all_games[i].lvec), &(grs.all_games[i].gvec), grs.num_p_features, grs.num_g_features, &templ);
			}

			prob_w = logistic_fun(Vec_inner_SV(w, &tempw) - Vec_inner_SV(w, &templ)); 
			prob_w = perturb_on_even(prob_w);
			//print_SparseVec(grs.all_games[i].wvec);
			//print_SparseVec(grs.all_games[i].lvec);
			//print_vec(w, grs.num_p_features);
			//printf("%f, (%f, %f)\n", prob_w, Vec_inner_SV(w, &(grs.all_games[i].wvec)), Vec_inner_SV(w, &(grs.all_games[i].lvec)));

			if(grs.all_games[i].tag == FOR_TRAINING)
			{
				Update_with_SV(w, &tempw, (1.0 - prob_w) * eta);
				Update_with_SV(w, &templ, -(1.0 - prob_w) * eta);
				realn_train++;
				ll_train += log(prob_w);
				acc_train += (prob_w >= 0.5 ? 1.0:0.0);


			}
			else if(grs.all_games[i].tag == FOR_VALIDATION)
			{
				realn_val++;
				ll_val += log(prob_w);
				acc_val += (prob_w >= 0.5 ? 1.0:0.0);
			}
			else if(grs.all_games[i].tag == FOR_TESTING)
			{
				realn_test++;
				ll_test += log(prob_w);
				acc_test += (prob_w >= 0.5 ? 1.0:0.0);
			}
		}

		//Regularization
		if(type != 1)
			add_vec(w, w, grs.num_p_features,  -lambda);
		else
			add_vec(w, w, grs.num_p_features + grs.num_g_features,  -lambda);

		p_tr -> realn_train = realn_train;
		p_tr -> realn_val = realn_val;
		p_tr -> realn_test = realn_test;

		p_tr -> ll_train = ll_train / realn_train;
		p_tr -> acc_train = acc_train / realn_train;

		p_tr -> ll_val = ll_val / realn_val;
		p_tr -> acc_val = acc_val / realn_val;

		p_tr -> ll_test = ll_test / realn_test;
		p_tr -> acc_test = acc_test / realn_test;

		if(type != 1)
			p_tr -> obj = p_tr -> ll_train * realn_train - 0.5 * lambda * pow(vec_norm(w, grs.num_p_features), 2.0);
		else
			p_tr -> obj = p_tr -> ll_train * realn_train - 0.5 * lambda * pow(vec_norm(w, grs.num_p_features + grs.num_g_features), 2.0);

		//printf("%d %f\n", t, p_tr -> obj);

		if(ll_last < 0 && fabs(p_tr -> obj - ll_last) < eps)
			break;
		ll_last =  p_tr -> obj;
		if(++t > 1000)
		   break;	
	}

	//One more time
	double ll_train = 0.0;
	double ll_val = 0.0;
	double ll_test = 0.0;

	double acc_train = 0.0;
	double acc_val = 0.0;
	double acc_test = 0.0;

	double realn_train = 0.0;
	double realn_val = 0.0;
	double realn_test = 0.0;

	double obj = 0.0;
	for(i = 0; i < grs.num_games; i++)
	{
		if(type != 1)
		{
			copy_SV(&tempw, &(grs.all_games[i].wvec));
			copy_SV(&templ, &(grs.all_games[i].lvec));
		}
		else
		{
			concat_SV(&(grs.all_games[i].wvec), &(grs.all_games[i].gvec), grs.num_p_features, grs.num_g_features, &tempw);
			concat_SV(&(grs.all_games[i].lvec), &(grs.all_games[i].gvec), grs.num_p_features, grs.num_g_features, &templ);
		}

		prob_w = logistic_fun(Vec_inner_SV(w, &tempw) - Vec_inner_SV(w, &templ)); 
		prob_w = perturb_on_even(prob_w);

		if(grs.all_games[i].tag == FOR_TRAINING)
		{
			realn_train++;
			ll_train += log(prob_w);
			acc_train += (prob_w >= 0.5 ? 1.0:0.0);


		}
		else if(grs.all_games[i].tag == FOR_VALIDATION)
		{
			realn_val++;
			ll_val += log(prob_w);
			acc_val += (prob_w >= 0.5 ? 1.0:0.0);
		}
		else if(grs.all_games[i].tag == FOR_TESTING)
		{
			realn_test++;
			ll_test += log(prob_w);
			acc_test += (prob_w >= 0.5 ? 1.0:0.0);
		}
	}

	p_tr -> realn_train = realn_train;
	p_tr -> realn_val = realn_val;
	p_tr -> realn_test = realn_test;

	p_tr -> ll_train = ll_train / realn_train;
	p_tr -> acc_train = acc_train / realn_train;

	p_tr -> ll_val = ll_val / realn_val;
	p_tr -> acc_val = acc_val / realn_val;

	p_tr -> ll_test = ll_test / realn_test;
	p_tr -> acc_test = acc_test / realn_test;

	if(type != 1)
		p_tr -> obj = p_tr -> ll_train * realn_train - 0.5 * lambda * pow(vec_norm(w, grs.num_p_features), 2.0);
	else
		p_tr -> obj = p_tr -> ll_train * realn_train - 0.5 * lambda * pow(vec_norm(w, grs.num_p_features + grs.num_g_features), 2.0);

	free(w);
	free_SparseVec(&tempw);
	free_SparseVec(&templ);
}
