//
//  RunRankingModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-31.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_RunRankingModel_h
#define LR_FCEM_proj_RunRankingModel_h

#include "PrepInstance.h"
#include "LrTuckerRankingModel.h"

class PPA_Params {
public:
    bool sum;
    bool position;
    bool prep;
    bool postag;
    bool clus;
    bool context;
    
    bool nextpos;
    
    bool verbnet;
    bool wordnet;
    
    int rank1;
    int rank2;
    int rank3;
    
    int rank;
    
    int fea_dim;
    
    void PrintValue() {
        cout << "sum:" << sum << endl;
        cout << "position:" << position << endl;
        cout << "prep:" << prep << endl;
        cout << "postag:" << postag << endl;
        cout << "clus:" << clus << endl;
        
        cout << "nextpos:" << nextpos << endl;
        cout << "verbnet:" << verbnet << endl;
        cout << "wordnet:" << wordnet << endl;
        
        cout << "context:" << context << endl;
        
        cout << "fea_dim:" << fea_dim << endl;
    }
};

class RunRankingModel
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
    vector<LrTuckerRankingModel*> lr_tensor_list;
    
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
    
    ~RunRankingModel() {}
    
    RunRankingModel(char* embfile, char* trainfile, PPA_Params* params) {
        inst = new PPAInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    
    int AddWordFeature(string feat_key, int pair_id, int pos);
    
    string ProcSenseTag(string input_type);
    string ProcNeTag(string input_type);
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
