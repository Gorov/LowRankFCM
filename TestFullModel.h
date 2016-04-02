//
//  TestFullModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-3.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_TestFullModel_h
#define LR_FCEM_proj_TestFullModel_h

#include "Instances.h"
//#include "FctCoarseModel.h"
//#include "FctDeepModel.h"
#include "LrFcemModel.h"
#include "LrTensorModel.h"
#include "LrTensorSparseModel.h"
#include "LrTensorModelBasic.h"
#include "TestFullModel.h"
#include "FullFctModel.h"

#define FEA_SPARSE

#ifdef FEA_SPARSE
#define IN_BETWEEN_GROUP 0
#define ON_PATH_GROUP 1
#define CONTEXT_GROUP 2
#define HEAD1_GROUP 3
#define HEAD2_GROUP 4
#else
#define IN_BETWEEN_GROUP 0
#define ON_PATH_GROUP 0
#define CONTEXT_GROUP 0
#define HEAD1_GROUP 0
#define HEAD2_GROUP 0
#endif

class TestFullModel
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
//    vector<LrTensorModel*> lr_tensor_list;
    vector<LrSparseTensorModel*> lr_tensor_list;
    
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
    
    //for sparse model
    int2int* feat_group;
    int num_group;
    int2int group_dict;
    
    TestFullModel() {inst = new BaseInstance();}
    ~TestFullModel() {}
    
    TestFullModel(char* embfile, char* trainfile) {
        type = "SEM_EVAL";
        //        inst = new BaseInstance();
        inst = new LrFcemInstance();
        Init(embfile, trainfile);
        debug = false;
    }
    TestFullModel(char* embfile, char* trainfile, FeaParams* params) {
        type = "SEM_EVAL";
        //        inst = new BaseInstance();
        inst = new LrFcemInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    TestFullModel(char* embfile, char* trainfile, char* configfile): fea_params(configfile) {
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
    void Init(char* embfile, char* trainfile, FeaParams* params);
    
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
