//
//  BaseComponentModel.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_BaseComponentModel_h
#define RE_FCT_BaseComponentModel_h

#include "FeatureEmbeddingModel.h"
#include "WordEmbeddingModel.h"
#include <tr1/unordered_map>
#include "Instances.h"
#include <math.h>
#include <stdlib.h>

#define MAX_EXP 86
#define MIN_EXP -86

//typedef std::tr1::unordered_map<string, int> feat2int;
typedef std::tr1::unordered_map<string, string> word2clus;

class BaseComponentModel
{
public:
    
    real alpha;
    real lambda;
    
    unsigned long num_labels;
    real* label_emb;
    real* label_bias;
    real* params_g;
    
    FeatureEmbeddingModel* fea_model;
    WordEmbeddingModel* emb_model;
    
    int word_emb_dim;
    int feat_emb_dim;
    
    real eta0;
    real eta;
    int iter;
    int cur_iter;
    unsigned long num_fea;
    
    bool update_fea_emb;
    bool update_emb;
    bool adagrad;
    
    BaseComponentModel() {};
    
    BaseComponentModel(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model) {
        //Init();
        this -> fea_model = fea_model;
        this -> emb_model = emb_model;
    }
    ~BaseComponentModel(){
        delete emb_model;
        delete fea_model;
        
        delete label_emb;
        delete label_bias;
    }
    
    virtual void ForwardOutputs(BaseInstance* b_inst) = 0;
    virtual long BackPropPhrase(BaseInstance* b_inst, real eta_real) = 0;
    
//    void SaveModel(string modelfile);
//    void LoadModel(string modelfile);
    
    virtual void ForwardProp(BaseInstance* b_inst) = 0;
    virtual void BackProp(BaseInstance* b_inst, real eta_real) = 0;
    
    void Init();
    virtual void PrintModelInfo() = 0;
};


#endif
