//#include "BTbaseline.h"
//#include "GamePairHash.h"
//#include "BTbaseline.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//#include <assert.h>
//#include <time.h>
//#include <math.h>
//#include <float.h>
//#define FOR_TRAINING 0
//#define FOR_VALIDATION 1
//#define FOR_TESTING 2
//extern void shuffle_gele_array(GELE* array, size_t n);
//extern double logistic_fun(double a);
//extern double safe_log_logit(double a);
//void init_BModel(BModel* p_bm, int num_players, int d, GPHASH* aggregated_games)
//{
//	(p_bm -> num_players) = num_players;
//	(p_bm -> d) = d;
//	(p_bm -> sigma) = create_gp_hash(aggregated_games -> length, num_players);
//	int i;
//	for(i = 0; i < aggregated_games -> length; i++)
//	{
//		(p_bm -> sigma).array[i].key = (aggregated_games -> array)[i].key;
//		if((aggregated_games -> array)[i].val.fst >= (aggregated_games -> array)[i].val.snd)
//		
//			(p_bm -> sigma).array[i].val.fst = 1;
//		else
//			(p_bm -> sigma).array[i].val.fst = -1;
//		(p_bm -> sigma).array[i].val.snd = 0;
//	}
//	(p_bm -> X) = randarray(num_players, d, 1.0);
//}
//
//void free_BModel(BModel* p_bm)
//{
//	Array2Dfree(p_bm -> X, p_bm -> num_players, p_bm -> d);
//	free_gp_hash(p_bm -> sigma);
//}
//
//void copy_BModel(BModel* src, BModel* dest)
//{
//	dest -> num_players = src -> num_players;
//	dest -> d = src -> d;
//	Array2Dcopy(src -> X, dest -> X, src -> num_players, src -> d);
//	copy_gp_hash(&(src -> sigma), &(dest -> sigma));
//}
//
//double matchup_fun_bm(BModel bm, int a, int b, double* sigma_ab)
//{
//	//print_mat(bm.X, bm.num_players, bm.d);
//	GPAIR temp_gp;
//	temp_gp.fst = a;
//	temp_gp.snd = b;
//	int no_use;
//	int idx =  find_gp_key(temp_gp, bm.sigma, &no_use);
//	*sigma_ab = 1.0; 
//
//	if(idx >= 0)
//		(*sigma_ab) = (double)(bm.sigma.array)[idx].val.fst;
//
//	assert((*sigma_ab) == 1.0 || (*sigma_ab) == -1.0);
//	//return (*sigma_ab) * pow(vec_diff_norm2(bm.X[a], bm.X[b], bm.d), 2.0);
//	return (*sigma_ab) * vec_diff_norm2(bm.X[a], bm.X[b], bm.d);
//}
//
//BModel train_baseline_model(GPHASH* aggregated_games, PARAS myparas, int verbose)
//{
//	BModel bm, bm_best, bm_last;
//	init_BModel(&bm, aggregated_games -> num_players, myparas.d, aggregated_games);
//	init_BModel(&bm_best, aggregated_games -> num_players, myparas.d, aggregated_games);
//	init_BModel(&bm_last, aggregated_games -> num_players, myparas.d, aggregated_games);
//	int iteration_count = 0;
//	double llhood, llhood_prev, realn, avg_ll = 0.0;
//	double best_ll = -DBL_MAX;
//	GELE game_ele;
//	int a, b, na, nb;
//	double sigma_ab;
//	double coeff;
//	double mf, prob_a, prob_b;
//	double* temp_d = (double*)malloc(myparas.d * sizeof(double));
//	int time_bomb;
//
//	GELE space_free[aggregated_games -> num_used];
//	int t = 0;
//	int i;
//	int triple_id;
//	for(i = 0; i < aggregated_games -> length; i++)
//		if(!gele_is_empty((aggregated_games -> array)[i]))
//			space_free[t++] = (aggregated_games -> array)[i];
//
//	while(iteration_count++ < myparas.max_iter)
//	{
//		copy_BModel(&bm, &bm_last);
//		llhood_prev = llhood;
//		llhood = 0.0;
//		realn = 0.0;
//		shuffle_gele_array(space_free, (size_t)(aggregated_games -> num_used));
//		if(verbose)
//			printf("Baseline iteration %d\n", iteration_count);
//		for(triple_id = 0; triple_id < aggregated_games -> num_used; triple_id++)
//		{
//			game_ele = space_free[triple_id]; 
//			a = game_ele.key.fst;
//			b = game_ele.key.snd;
//			na = game_ele.val.fst;
//			nb = game_ele.val.snd;
//			//printf("(%d, %d, %d, %d)\n", a, b, na, nb);
//			assert(a < b);
//			mf =  matchup_fun_bm(bm, a, b, &sigma_ab);
//			//printf("mf: %f\n", mf);
//			//printf("really: %f\n", 1.0 / (1.0 + exp(-mf)));
//			prob_a = logistic_fun(mf);
//			prob_b = 1.0 - prob_a;
//			//printf("pa, pb: (%f, %f)\n", prob_a, prob_b);
//
//			coeff = (na * prob_b - nb * prob_a);
//			
//			memcpy(temp_d, bm.X[a], myparas.d);
//			add_vec(temp_d, bm.X[b], myparas.d, -1.0);
//			scale_vec(temp_d, myparas.d, sigma_ab / vec_diff_norm2(bm.X[a], bm.X[b], myparas.d));
//
//			add_vec(bm.X[a], temp_d, myparas.d, myparas.eta * coeff );
//			add_vec(bm.X[b], temp_d, myparas.d, -1.0 * myparas.eta * coeff);
//
//			realn += (na + nb);
//			llhood += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
//		}
//		avg_ll = llhood / realn;
//		//printf("realn: %f\n", realn);
//		if(verbose)
//			printf("Baseline Avg training log-likelihood: %f\n", avg_ll);
//		if(isnan(avg_ll))
//			break;
//
//		if(avg_ll > best_ll) 
//		{
//			best_ll = avg_ll;
//			copy_BModel(&bm_last, &bm_best);
//			time_bomb = 0;
//		}
//		else
//			time_bomb++;
//		if(time_bomb > myparas.bomb_thresh)
//			break;
//	}
//	copy_BModel(&bm_best, &bm);
//	free_BModel(&bm_best);
//	free_BModel(&bm_last);
//	free(temp_d);
//	return bm;
//}
//
//void test_baseline_model(BModel bm, GRECORDS grs, int* test_mask)
//{
//	int i, triple_id;
//	int a, b, w,  na, nb;
//	double prob_a, prob_b, mf, coeff, sigma_ab;
//	double ll_validate, ll_test, realn_validate, realn_test, correct_validate, correct_test = 0.0;
//	for(triple_id = 0; triple_id < grs.num_games; triple_id++)
//	{
//		a = grs.all_games[triple_id].pa;
//		b = grs.all_games[triple_id].pb;
//		w = grs.all_games[triple_id].pw;
//		if(a == w)
//		{
//			na = 1;
//			nb = 0;
//		}
//		else
//		{
//			na = 0;
//			nb = 1;
//		}
//		mf =  matchup_fun_bm(bm, a, b, &sigma_ab);
//		prob_a = logistic_fun(mf);
//		prob_b = 1.0 - prob_a;
//		coeff = (na * prob_b - nb * prob_a);
//		if(test_mask[triple_id] == FOR_VALIDATION)
//		{
//			realn_validate += (na + nb);
//			ll_validate += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
//			if((w == a && prob_a >= 0.5) || (w == b && prob_b > 0.5))
//				correct_validate++;
//		}
//		if(test_mask[triple_id] == FOR_TESTING)
//		{
//			realn_test += (na + nb);
//			ll_test += (na * safe_log_logit(-mf) + nb * safe_log_logit(mf));
//			if((w == a && prob_a >= 0.5) || (w == b && prob_b > 0.5))
//				correct_test++;
//		}
//	}
//	printf("Avg baseline validation log-likelihood: %f\n", ll_validate / realn_validate);
//	printf("Avg baseline validation accuracy: %f\n", (correct_validate / realn_validate));
//	printf("Avg baseline testing log-likelihood: %f\n", ll_test / realn_test);
//	printf("Avg baseline testing accuracy: %f\n", (correct_test / realn_test));
//
//}
