//
//  FullFctModel.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_FullFctModel_h
#define RE_FCT_FullFctModel_h

#include "Instances.h"
//#include "FctCoarseModel.h"
//#include "FctDeepModel.h"
//#include "FctConvolutionModel.h"
#include "LrFcemModel.h"
#include "LrTensorModel.h"
#include "LrTensorModelBasic.h"

class FeaParams {
public:
    bool path;
    bool type;
    bool postag;
    bool dep;
    bool ner;
    bool sst;
    bool context;
    bool tag_fea;
    bool dep_fea;
    bool word_on_path;
    bool word_on_path_type;
    bool entity_type;
    bool head;
    bool hyper_emb;
    
    bool dep_path;
    bool pos_on_path;
    
    bool tri_conv;
    bool linear;
    
    int fea_dim;
    
    int rank1, rank2, rank3;
    
    FeaParams(){
        path = 0;
        type = 0;
        postag = 0;
        dep = 1;
        ner = 0;
        sst = 0;
        context = 1;
        tag_fea = 0;
        dep_fea = 0;
        word_on_path = 1;
        word_on_path_type = 0;
        entity_type = 0;
        head = 1;
        hyper_emb = 0;
        
        dep_path = 0;
        pos_on_path = 0;
        
        tri_conv = 0;
        linear = 0;
        
        fea_dim = 50;
    }
    FeaParams(char* filename) {
        {
            ifstream ifs(filename);
            char line[1000];
            int value;
            string key;
            string tmp;
            
            path = 0;
            type = 0;
            postag = 0;
            dep = 1;
            ner = 0;
            sst = 0;
            context = 1;
            tag_fea = 0;
            dep_fea = 0;
            word_on_path = 1;
            word_on_path_type = 0;
            entity_type = 0;
            head = 1;
            hyper_emb = 0;
            
            dep_path = 0;
            pos_on_path = 0;
            
            tri_conv = 0;
            linear = 0;
            
            fea_dim = 50;
            
            ifs.getline(line, 1000);
            while (!strcmp(line, "")) {
                istringstream iss(line);
                iss >> key;
                iss >> tmp;
                iss >> value;
                if (key.compare("path") == 0) {
                    path = value;
                }
                else if (key.compare("type") == 0) {
                    type = value;
                }
                else if (key.compare("postag") == 0) {
                    postag = value;
                }
                else if (key.compare("dep") == 0) {
                    dep = value;
                }
                else if (key.compare("ner") == 0) {
                    ner = value;
                }
                else if (key.compare("sst") == 0) {
                    sst = value;
                }else if (key.compare("context") == 0) {
                    context = value;
                }else if (key.compare("tag_fea") == 0) {
                    tag_fea = value;
                }else if (key.compare("dep_fea") == 0) {
                    dep_fea = value;
                }else if (key.compare("word_on_path") == 0) {
                    word_on_path = value;
                }else if (key.compare("word_on_path_type") == 0) {
                    word_on_path_type = value;
                }else if (key.compare("entity_type") == 0) {
                    entity_type = value;
                }else if (key.compare("head") == 0) {
                    head = value;
                }else if (key.compare("hyper_emb") == 0) {
                    hyper_emb = value;
                }else if (key.compare("dep_path") == 0) {
                    dep_path = value;
                }else if (key.compare("pos_on_path") == 0) {
                    pos_on_path = value;
                }else if (key.compare("fea_dim") == 0) {
                    fea_dim = value;
                }
                
            }
        }
    }
    
    void PrintValue() {
        cout << "path:" << path << endl;
        cout << "type:" << type << endl;
        cout << "postag:" << postag << endl;
        cout << "dep:" << dep << endl;
        cout << "context:" << context << endl;
        cout << "dep_fea:" << dep_fea << endl;
        cout << "head:" << head << endl;
        cout << "word_on_path:" << word_on_path << endl;
        cout << "word_on_path_type:" << word_on_path_type << endl;
        cout << "entity_type:" << entity_type << endl;
        cout << "hyper_emb:" << hyper_emb << endl;
        cout << "dep_path:" << dep_path << endl;
        cout << "pos_on_path:" << pos_on_path << endl;
        
        cout << "tri_conv:" << tri_conv << endl; 
    }
};


class FullFctModel
{
public:
    string type;
    bool adagrad;
    bool update_feat_emb;
    bool update_emb;
    
//    vector<FctCoarseModel*> coarse_fct_list;
//    vector<FctDeepModel*> deep_fct_list;
//    vector<FctConvolutionModel*> convolution_fct_list;
//    vector<EmbeddingModel*> emb_model_list;
    vector<LrFcemModel*> lr_fcem_list;
    WordEmbeddingModel* emb_model;
    FeatureEmbeddingModel* fea_model;
    
    LabelEmbeddingModel* lab_model;
    vector<LrTensorModelBasic*> lr_tensor_list;
    
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
    
    FullFctModel() {inst = new BaseInstance();}
    ~FullFctModel() {}
    
    FullFctModel(char* embfile, char* trainfile) {
        type = "SEM_EVAL";
//        inst = new BaseInstance();
        inst = new LrFcemInstance();
        Init(embfile, trainfile);
    }
    FullFctModel(char* embfile, char* trainfile, char* configfile): fea_params(configfile) {
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
