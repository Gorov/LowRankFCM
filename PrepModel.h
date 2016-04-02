//
//  PrepModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-18.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_PrepModel_h
#define LR_FCEM_proj_PrepModel_h

#include "LrFcemModel.h"
#include "LrTensorModel.h"
#include "LrTensorSparseModel.h"
#include "Instances.h"
#include "PrepInstance.h"

class Prep_Params {
public:
    bool sum;
    bool position;
    bool prep;
    bool clus;
    
    bool low_rank;
    
    bool tri_conv;
    bool linear;
    
    int fea_dim;
    
    void PrintValue() {
        cout << "position:" << position << endl;
        cout << "prep:" << prep << endl;
        
        cout << "fea_dim:" << fea_dim << endl;
        
        cout << "tri_conv:" << tri_conv << endl; 
        cout << "linear:" << linear << endl; 
    }
};

class PrepModel
{
public:
    string type;
    bool adagrad;
    bool update_emb;
    
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model;
    
    LabelEmbeddingModel* lab_model;
    vector<LrTensorModel*> lr_tensor_list;
    
    int num_models;
    //    int layer1_size;
    int word_emb_dim;
    int feat_emb_dim;
    int num_inst;
    int max_len;
    
    feat2int labeldict;
    vector<string> labellist;
    
    int num_labels;
    int num_feats;
    PrepInstance* inst;
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
    
    PrepModel() {inst = new PrepInstance();}
    ~PrepModel() {}
    
    PrepModel(char* embfile, char* trainfile, Prep_Params* params) {
        type = "PREP";
        inst = new PrepInstance();
        Init(embfile, trainfile, params);
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstance(ifstream& ifs, int type);
    
    int AddWordFeature(string feat_key, int pos);
    
//    int SearchFeature(string feat_key);
    int AddFeature(string feat_key);
    
    string ProcSenseTag(string input_type);
    string ProcNeTag(string input_type);
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata, Prep_Params* params);
    
    void ForwardProp();
    void BackProp();
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
    //    virtual void EvalData(string trainfile, string outfile, int type);
    
    void SetModels();
    void PrintModelInfo();
    void WeightDecay(real eta_real, real lambda);
    
    //    void PushWordFeature(string slot_key);
};


#endif
