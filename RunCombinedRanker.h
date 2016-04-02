//
//  RunCombinedRanker.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-11.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_RunCombinedRanker_h
#define LR_FCEM_proj_RunCombinedRanker_h

#include "PrepInstance.h"
#include "LrRankingModel.h"
#include "LrTuckerRankingModel.h"
#include "RunRankingModel.h"
#include "FeatureFactory.h"

class RunCombinedRanker
{
public:
    bool adagrad;
    bool update_feat_emb;
    bool update_lab_emb;
    bool update_emb;
    
    bool enable_model_1;
    bool enable_model_2;
    
    bool debug;
    
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model[2];
    
    LabelEmbeddingModel* lab_model;
    vector<LrRankingModel*> lr_bigram_list;
    vector<LrTuckerRankingModel*> lr_unigram_list;
    
    int num_models;
    int word_emb_dim;
    int feat_emb_dim[2];
    int num_inst;
    int max_len;
    
    feat2int labeldict;
    vector<string> labellist;
    
    PPA_FeatureFactory fea_factory[2];
    
    int num_labels;
    int num_feats;
    BaseInstance* inst[2];
    PPA_Params fea_params[2];
    
    real eta0;
    real eta;
    real eta_real;
    real alpha_old;
    real alpha;
    real lambda;
    real lambda_prox;
    
    int iter;
    int cur_iter;
    
    int2int* feat_group;
    int num_group;
    int2int group_dict;
    
    ~RunCombinedRanker() {}
    
    RunCombinedRanker(char* embfile, char* trainfile, PPA_Params* params) {
        inst[0] = new PPAInstance();
        inst[1] = new PPAInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstanceOnly(ifstream& ifs, bool add);
    
    string ToLower(string& s);
    
    void Init(char* embfile, char* trainfile, PPA_Params* params);
    
    void ForwardProp();
    void BackProp();
    
    virtual void TrainData(string trainfile, string devfile);
    virtual void EvalData(string trainfile);
    
    void EvalClosest(string trainfile);
    
    void SetModels();
    void PrintModelInfo();
};


#endif
