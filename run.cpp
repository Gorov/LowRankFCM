//
//  RE_FCT.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-31.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sstream>
#include <limits>
#include "RunRankingModel.h"
#include "RunBigramRanking.h"
#include "RunCombinedRanker.h"

#define SEM_EVAL 1l
#define ACE_2005 2

char train_file[MAX_STRING], dev_file[MAX_STRING], res_file[MAX_STRING];
char output_file[MAX_STRING], param_file[MAX_STRING];
char clus_file[MAX_STRING], baseemb_file[MAX_STRING], freq_file[MAX_STRING];
char model_file[MAX_STRING];
char feature_file[MAX_STRING];
char trainsub_file[MAX_STRING];
int iter = 1;
int finetuning = 1;
real alpha = 0.01;
real lambda = 0.01;
int rank1 = 100;

int ArgPos(char *str, int argc, char **argv);

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    output_file[0] = 0;
    string dir = "/Users/gflfof/Desktop/new work/path embedding/2014summer/data/";
    string tmp;
    if (true) {
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-dev", argc, argv)) > 0) strcpy(dev_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(res_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-baseemb", argc, argv)) > 0) strcpy(baseemb_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-rank1", argc, argv)) > 0) rank1 = atoi(argv[i + 1]);
        
//        if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
//        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    }
    
    if (true) {
        dir = "data/";
        //tmp = dir + "wsj.train.brown6.nextpos";
        //strcpy(train_file, tmp.c_str());
        //tmp = dir + "wsj.test.brown6.nextpos";
        //strcpy(dev_file, tmp.c_str());
        //tmp = dir + "ppa.wsj.nyt2011.cbow.bin.filtered";
        //strcpy(baseemb_file, tmp.c_str());
        
        PPA_Params* params = new PPA_Params();
        params -> rank1 = rank1;
        //params -> rank2 = 12;
        //params -> rank3 = 19;
        params -> rank2 = 20;
        params -> rank3 = 1;
        params -> clus = false;
        params -> verbnet = true;
        params -> wordnet = false;
        params -> nextpos = true;
        
        RunCombinedRanker* plearner = new RunCombinedRanker(baseemb_file, train_file, params);
        //        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
        plearner -> adagrad = true;
        plearner -> update_emb = false;
        plearner -> update_feat_emb = true;
        plearner -> update_lab_emb = false;
        //plearner -> eta = plearner -> eta0 = 0.05; //0.1; //0.02;
        plearner -> eta = plearner -> eta0 = 0.1;
        plearner -> lambda = lambda;
        
        plearner -> SetModels();
        plearner -> PrintModelInfo();
        
        iter = 20;
        plearner -> iter = iter;
        plearner -> lambda_prox = 0;
        
        plearner -> EvalClosest(dev_file);
        //        plearner -> EvalData(dev_file);
        
        plearner -> TrainData(train_file, dev_file);
        
        cout << "end" << endl;
    }
}
