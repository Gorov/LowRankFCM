//
//  TestTuckerModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-22.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_TestTuckerModel_h
#define LR_FCEM_proj_TestTuckerModel_h

#include "Instances.h"
#include "LrFcemModel.h"
#include "LrTensorModel.h"
#include "LrTensorSparseModel.h"
#include "LrTensorModelBasic.h"
#include "TestFullModel.h"
#include "FullFctModel.h"
#include "LrTensorTuckerModel.h"

class TestTuckerModel
{
public:
    string type;
    bool adagrad;
    bool update_feat_emb;
    bool update_lab_emb;
    bool update_emb;
    
    bool debug;
    
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model;
    
    LabelEmbeddingModel* lab_model;
    vector<LrTensorTuckerModel*> lr_tensor_list;
    
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
    FeaParams fea_params;
    
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
    
    TestTuckerModel() {inst = new BaseInstance();}
    ~TestTuckerModel() {}
    
    TestTuckerModel(char* embfile, char* trainfile, FeaParams* params) {
        type = "SEM_EVAL";
        //        inst = new BaseInstance();
        inst = new LrFcemInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstance(ifstream& ifs, int type);
    int LoadInstanceSemEval(ifstream& ifs);
    
    int AddWordFeature(string feat_key, int pos);
    
    string ProcSenseTag(string input_type);
    string ProcNeTag(string input_type);
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* trainfile, FeaParams* params);
    
    void ForwardProp();
    void BackProp();
    
    void PrintInstance(ofstream& ofs, LrFcemInstance* p_inst);
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
    
    void SetModels();
    void PrintModelInfo();
    void WeightDecay(real eta_real, real lambda);
};


#endif
