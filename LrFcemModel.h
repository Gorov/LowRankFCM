//
//  LrFcemModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrFcemModel_h
#define LR_FCEM_proj_LrFcemModel_h

#include "BaseComponentModel.h"

class LrFcemModel: public BaseComponentModel
{
public:
    
//    LrFcemInstance* inst;
    real* emb_p;
    real* part_emb_p;
    
    real* word_emb_sum;
    real* feat_emb_sum;
    real* word_grad_sum;
    real* feat_grad_sum;
    
    LrFcemModel() {//inst = new LrFcemInstance();
    };
    ~LrFcemModel() {}
    
    LrFcemModel(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model) {
        this -> emb_model = emb_model;
        word_emb_dim = emb_model -> dim;
        this -> fea_model = fea_model;
        feat_emb_dim = fea_model -> dim;
//        inst = new LrFcemInstance();
    }
    
    void OuterProd(vector<int>& words, int num_words, vector<int>& feats, int num_feats);
    
    void ForwardOuterProd(BaseInstance* b_inst);
    void ForwardOutputs(BaseInstance* b_inst);
    
    void BackPropViews(vector<int>& words, int num_words, vector<int>& feats, int num_feats, real eta_real);
    long BackPropOuterProd(BaseInstance* b_inst, real eta_real);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real);
    
    void Init(char* embfile, char* trainfile, int type);
    void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
//    void BackPropFea(BaseInstance* b_inst, int word_id, int class_id, int correct, real eta_real);
    
    void PrintModelInfo();
};


#endif
