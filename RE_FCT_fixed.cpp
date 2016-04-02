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
#include "FullFctModel.h"
#include "TestFullModel.h"
#include "TestFullModel2.h"
#include "TestTuckerModel.h"
#include "TestTuckerMtl.h"
#include "RunRankingModel.h"
#include "RunBigramRanking.h"
#include "TestBigramModelPrep.h"

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
    if (false) {
        if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-dev", argc, argv)) > 0) strcpy(dev_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(res_file, argv[i + 1]);
        if ((i = ArgPos((char *)"-baseemb", argc, argv)) > 0) strcpy(baseemb_file, argv[i + 1]);
        
//        if ((i = ArgPos((char *)"-epochs", argc, argv)) > 0) epochs = atoi(argv[i + 1]);
//        if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
//        if ((i = ArgPos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    }
    
    if (true) {
        dir = "/Users/gflfof/Desktop/new work/low_rank fcm/pp-data-english/";
        tmp = dir + "wsj.train.nextpos";
        strcpy(train_file, tmp.c_str());
        tmp = dir + "wsj.test.nextpos";
        strcpy(dev_file, tmp.c_str());
        tmp = dir + "ppa.wsj.nyt2011.cbow.bin.filtered";
        strcpy(baseemb_file, tmp.c_str());
        
        PPA_Params* params = new PPA_Params();
        params -> rank1 = 200;
        //params -> rank2 = 12;
        //params -> rank3 = 19;
        params -> rank2 = 20;
        params -> rank3 = 1;
        params -> clus = false;
        params -> verbnet = false;
        params -> wordnet = false;
        params -> nextpos = true;
        
        RunBigramRanking* plearner = new RunBigramRanking(baseemb_file, train_file, params);
        //        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
        plearner -> adagrad = true;
        plearner -> update_emb = false;
        plearner -> update_feat_emb = false;
        plearner -> update_lab_emb = false;
        plearner -> SetModels();
        plearner -> PrintModelInfo();
        
        iter = 10;
        plearner -> iter = iter;
        plearner -> eta = plearner -> eta0 = 0.1; //0.02;
        plearner -> lambda = 0;
        plearner -> lambda_prox = 0;
        
        plearner -> EvalClosest(dev_file);
        //        plearner -> EvalData(dev_file);
        
        plearner -> TrainData(train_file, dev_file);
        
        cout << "end" << endl;
    }
    
    if (false) {
        dir = "/Users/gflfof/Desktop/new work/low_rank fcm/pp-data-english/";
//        tmp = dir + "wsj.train.data";
        tmp = dir + "wsj.train.brown6data";
        tmp = dir + "wsj.train.verbnet";
        tmp = dir + "wsj.train.wordnet";
        tmp = dir + "wsj.train.nextpos.verbnet.wordnet";
        strcpy(train_file, tmp.c_str());
        tmp = dir + "wsj.test.brown6data";
        tmp = dir + "wsj.test.verbnet";
        tmp = dir + "wsj.test.wordnet";
        tmp = dir + "wsj.test.nextpos.verbnet.wordnet";
        strcpy(dev_file, tmp.c_str());
        tmp = dir + "ppa.wsj.nyt2011.cbow.bin.filtered";
        strcpy(baseemb_file, tmp.c_str());
        
        PPA_Params* params = new PPA_Params();
        params -> rank1 = 200;
        //params -> rank2 = 12;
        //params -> rank3 = 19;
        params -> rank2 = 20;
        params -> rank3 = 1;
        params -> clus = false;
        params -> verbnet = true;
        params -> wordnet = true;
        params -> nextpos = true;
        
        RunRankingModel* plearner = new RunRankingModel(baseemb_file, train_file, params);
        //        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
        plearner -> adagrad = true;
        plearner -> update_emb = false;
        plearner -> update_feat_emb = false;
        plearner -> update_lab_emb = false;
        plearner -> SetModels();
        plearner -> PrintModelInfo();
        
        iter = 10;
        plearner -> iter = iter;
        plearner -> eta = plearner -> eta0 = 0.1; //0.02;
        plearner -> lambda = 0;
        plearner -> lambda_prox = 0;
        
        plearner -> EvalClosest(dev_file);
//        plearner -> EvalData(dev_file);
        
        plearner -> TrainData(train_file, dev_file);
        
        cout << "end" << endl;
    }
    
//    if (false) {
////        tmp = dir + "SemEval.train.tmp2";
//        tmp = dir + "SemEval.train.fea.sst";
//        strcpy(train_file, tmp.c_str());
//        tmp = dir + "SemEval.test.fea.sst";
//        strcpy(dev_file, tmp.c_str());
//        tmp = dir + "predict.txt";
//        strcpy(res_file, tmp.c_str());
//        tmp = dir + "vectors.nyt2011.cbow.semeval.filtered";
//        strcpy(baseemb_file, tmp.c_str());
//    }
//    if (false) {
//        FeaParams* params = new FeaParams();
//        params -> rank1 = 50;
//        //params -> rank2 = 12;
//        //params -> rank3 = 19;
//        params -> rank2 = 20;
//        params -> rank3 = 32;
//        
//        TestTuckerMtl* plearner = new TestTuckerMtl(baseemb_file, train_file, params);
//        //        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
//        plearner -> adagrad = true;
//        plearner -> update_emb = true;
//        plearner -> update_feat_emb = true;
//        plearner -> update_lab_emb = true;
//        ((TestTuckerModel*)plearner) -> SetModels();
//        //        plearner -> InitSubmodels();
//        ((TestTuckerModel*)plearner) -> PrintModelInfo();
//        
//        //        plearner -> iter = atoi(argv[5]);
//        //        plearner -> eta = plearner -> eta0 = atof(argv[6]);
//        plearner -> iter = iter;
//        plearner -> eta = plearner -> eta0 = 0.05; //0.02;
//        plearner -> lambda = 0;
//        plearner -> lambda_prox = 0;
//        
//        plearner -> EvalData(dev_file, SEM_EVAL);
//        plearner -> TrainData(train_file, dev_file, SEM_EVAL);
//        //plearner -> TrainData(train_file, train_file);
//        
//        //        plearner -> EvalData(train_file, REALFCT_INST);
//        //        plearner -> EvalData(dev_file, res_file, SEM_EVAL);
//        
//        cout << "end" << endl;
//    }
//    if (false) {
//        FeaParams* params = new FeaParams();
//        params -> rank1 = 200;
//        params -> rank2 = 12;
//        params -> rank3 = 19;
//        TestTuckerModel* plearner = new TestTuckerModel(baseemb_file, train_file, params);
//        //        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
//        plearner -> adagrad = true;
//        plearner -> update_emb = false;
//        plearner -> update_feat_emb = true;
//        plearner -> update_lab_emb = false;
//        plearner -> SetModels();
//        //        plearner -> InitSubmodels();
//        plearner -> PrintModelInfo();
//        
//        plearner -> iter = 20;
//        plearner -> eta = plearner -> eta0 = 0.02;// 0.05;
//        plearner -> lambda = 0;
//        plearner -> lambda_prox = 0;
//        
////        plearner -> EvalData(dev_file, SEM_EVAL);
//        plearner -> TrainData(train_file, dev_file, SEM_EVAL);
//        //plearner -> TrainData(train_file, train_file);
//        
//        //        plearner -> EvalData(train_file, REALFCT_INST);
//        //        plearner -> EvalData(dev_file, res_file, SEM_EVAL);
//        
//        cout << "end" << endl;
//        return 0;
//    }
//    if (true) {
//        int fold = 0;
//        string dir = "/Users/gflfof/Desktop/new work/low_rank fcm/preposition_data/";
//        ostringstream oss;
//        oss.str("");
//        oss << dir << "output/" << "train." << fold << ".data";
//        //        oss << dir << "test.data";
//        strcpy(train_file, oss.str().c_str());
//        oss.str("");
//        oss << dir << "output/" << "test." << fold << ".data";
//        //        oss << dir << "train.data";
//        strcpy(dev_file, oss.str().c_str());
//        //        strcpy(res_file, argv[3]);
//        oss.str("");
//        //        oss << dir << "vectors.nyt2011.cbow.prep.filtered";
//        oss << dir << "GoogleNews-vectors-negative300.bin.prep.filtered";
//        strcpy(baseemb_file, oss.str().c_str());
//        
//        Prep_Params* params = new Prep_Params();
//        params -> bigram = true;
//        TestBigramModelPrep* plearner = new TestBigramModelPrep(baseemb_file, train_file, params);
//        //        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
//        plearner -> adagrad = true;
//        plearner -> update_emb = false;
//        
//        plearner -> SetModels();
//        //        plearner -> InitSubmodels();
//        plearner -> PrintModelInfo();
//        
//        plearner -> iter = 20;
//        plearner -> eta = plearner -> eta0 = 0.02;// 0.05;
//        plearner -> lambda = 0;
//        plearner -> lambda_prox = 0;
//        
//        //        plearner -> EvalData(dev_file, SEM_EVAL);
//        plearner -> TrainData(train_file, dev_file, SEM_EVAL);
//        //plearner -> TrainData(train_file, train_file);
//        
//        //        plearner -> EvalData(train_file, REALFCT_INST);
//        //        plearner -> EvalData(dev_file, res_file, SEM_EVAL);
//        
//        cout << "end" << endl;
//        return 0;
//    }
//    if (false) {
//        TestFullModel* plearner = new TestFullModel(baseemb_file, train_file);
////        FullFctModel* plearner = new FullFctModel(baseemb_file, train_file);
//        plearner -> adagrad = true;
//        plearner -> update_emb = true;
//        plearner -> update_feat_emb = true;
//        plearner -> SetModels();
////        plearner -> InitSubmodels();
//        plearner -> PrintModelInfo();
//        
////        plearner -> iter = atoi(argv[5]);
////        plearner -> eta = plearner -> eta0 = atof(argv[6]);
//        plearner -> iter = 20;
//        plearner -> eta = plearner -> eta0 = 0.02;// 0.05;
//        plearner -> lambda = 0;
//        plearner -> lambda_prox = 0;
//        
//        //        plearner -> EvalData(dev_file, SEMEVAL_INST);
//        plearner -> TrainData(train_file, dev_file, SEM_EVAL);
//        //plearner -> TrainData(train_file, train_file);
//        
//        //        plearner -> EvalData(train_file, REALFCT_INST);
////        plearner -> EvalData(dev_file, res_file, SEM_EVAL);
//        
//        cout << "end" << endl;
//        return 0;
//    }
}
