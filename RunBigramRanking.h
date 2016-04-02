//
//  RunBigramRanking.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-9.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_RunBigramRanking_h
#define LR_FCEM_proj_RunBigramRanking_h

#include "PrepInstance.h"
#include "LrRankingModel.h"
#include "RunRankingModel.h"

class RunBigramRanking
{
public:
    bool adagrad;
    bool update_feat_emb;
    bool update_lab_emb;
    bool update_emb;
    
    bool debug;
    
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model;
    
    LabelEmbeddingModel* lab_model;
    vector<LrRankingModel*> lr_tensor_list;
    
    int num_models;
    int word_emb_dim;
    int feat_emb_dim;
    int num_inst;
    int max_len;
    
    feat2int labeldict;
    vector<string> labellist;
    
    int num_labels;
    int num_feats;
    BaseInstance* inst;
    PPA_Params fea_params;
    
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
    
    ~RunBigramRanking() {}
    
    RunBigramRanking(char* embfile, char* trainfile, PPA_Params* params) {
        inst = new PPAInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    
    int AddWordFeature(string feat_key, int pair_id, int pos);
    
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
