//
//  TestBigramModelPrep.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-24.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_TestBigramModelPrep_h
#define LR_FCEM_proj_TestBigramModelPrep_h

#include "LrTensorBigramModel.h"
#include "Instances.h"
#include "PrepInstance.h"

class Prep_Params {
public:
    bool sum;
    bool position;
    bool prep;
    bool clus;
    
    bool bigram;
    
    bool low_rank;
    
    int fea_dim;
    
    void PrintValue() {
        cout << "position:" << position << endl;
        cout << "prep:" << prep << endl;
        
        cout << "sum:" << position << endl;
        cout << "clus:" << prep << endl;
        cout << "bigram:" << prep << endl;
        cout << "low_rank:" << prep << endl;
        
        cout << "fea_dim:" << fea_dim << endl;
    }
};

class TestBigramModelPrep
{
public:
    string type;
    bool adagrad;
    bool update_emb;
    
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model;
    
    LabelEmbeddingModel* lab_model;
    vector<LrTensorBigramModel*> lr_tensor_list;
    
    int num_models;
    int word_emb_dim;
    int feat_emb_dim;
    int num_inst;
    int max_len;
    
    feat2int labeldict;
    vector<string> labellist;
    
    int num_labels;
    int num_feats;
    PrepBigramInstance* inst;
    Prep_Params fea_params;
    
    real eta0;
    real eta;
    real eta_real;
    real alpha_old;
    real alpha;
    real lambda;
    real lambda_prox;
    
    int iter;
    int cur_iter;
    
    TestBigramModelPrep() {inst = new PrepBigramInstance();}
    ~TestBigramModelPrep() {}
    
    TestBigramModelPrep(char* embfile, char* trainfile, Prep_Params* params) {
        type = "PREP";
        inst = new PrepBigramInstance();
        Init(embfile, trainfile, params);
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstance(ifstream& ifs, int type);
    
    int AddWordFeature(string feat_key, int pos);
    
    int AddFeature(string feat_key);
    
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata, Prep_Params* params);
    
    void ForwardProp();
    void BackProp();
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
    
    void SetModels();
    void PrintModelInfo();
};


#endif
