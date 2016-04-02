//
//  TestFullModel2.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-7.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_TestFullModel2_h
#define LR_FCEM_proj_TestFullModel2_h

#include "Instances.h"
//#include "FctCoarseModel.h"
//#include "FctDeepModel.h"
#include "LrFcemModel.h"
#include "LrTensorModel.h"
#include "LrTensorModelBasic.h"
#include "TestFullModel.h"
#include "FullFctModel.h"

class TestFullModel2
{
public:
    string type;
    bool adagrad;
    bool update_feat_emb;
    bool update_emb;
    
    bool debug;
    
    //    vector<FctCoarseModel*> coarse_fct_list;
    //    vector<FctDeepModel*> deep_fct_list;
    //    vector<FctConvolutionModel*> convolution_fct_list;
    //    vector<EmbeddingModel*> emb_model_list;
    vector<LrFcemModel*> lr_fcem_list;
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model;
    
    LabelEmbeddingModel* lab_model;
    vector<LrTensorModelBasic*> lr_tensor_list;
//    vector<LrTensorModel*> lr_tensor_list;
    
    int num_models;
    //    int layer1_size;
    int word_emb_dim;
    int feat_emb_dim;
    int num_inst;
    int max_len;
    
    feat2int slot2lr_fcem;
    vector<string> lr_fcem_slot_list;
    
    //    feat2int slot2coarse_model;
    //    vector<string> coarse_slot_list;
    //    feat2int slot2deep_model;
    //    vector<string> deep_slot_list;
    //    
    //    feat2int slot2convolution_model;
    //    vector<string> convolution_slot_list;
    
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
    
    TestFullModel2() {inst = new BaseInstance();}
    ~TestFullModel2() {}
    
    TestFullModel2(char* embfile, char* trainfile) {
        type = "SEM_EVAL";
        //        inst = new BaseInstance();
        inst = new LrFcemInstance();
        Init(embfile, trainfile);
        debug = false;
    }
    TestFullModel2(char* embfile, char* trainfile, char* configfile): fea_params(configfile) {
        type = "SEM_EVAL";
        //        inst = new BaseInstance();
        inst = new LrFcemInstance();
        Init(embfile, trainfile);
    }
    
    void BuildModelsFromData(char* trainfile);
    //    void InitSubmodels();
    
    //void EvalData(string trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    int LoadInstance(ifstream& ifs, int type);
    int LoadInstanceSemEval(ifstream& ifs);
    //    int SearchCoarseFctSlot(string slot_key);
    
    int AddWordFeature(string feat_key, int pos);
    
    //    void AddHeadFctModels();
    //    void AddInBetweenFctModels();
    //    void AddDepPathFctModels();
    //    void AddFctModelsFromFile();
    
    //    int AddDeepFctModel2List(string slot_key, string fea_key, bool add);
    
    string ProcSenseTag(string input_type);
    string ProcNeTag(string input_type);
    string ToLower(string& s);
    
    void Init(char* embfile, char* traindata);
    //    void Init(char* embfile, char* trainfile, int type);
    
    void ForwardProp();
    void BackProp();
    
    void PrintInstance(ofstream& ofs, LrFcemInstance* p_inst);
    
    //    void BackPropFea(int fea_id, int word_id, int class_id, int correct, real eta_real);
    //    void BackPropWord(int fea_id, int word_id);
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
    virtual void EvalData(string trainfile, string outfile, int type);
    
    void SetModels();
    void PrintModelInfo();
    void WeightDecay(real eta_real, real lambda);
};

#endif
