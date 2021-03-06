//
//  LrTensorModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrTensorModel_h
#define LR_FCEM_proj_LrTensorModel_h

#include "BaseComponentModel.h"
#include "LabelEmbeddingModel.h"

class LrTensorModel: public BaseComponentModel
{
public:
    int rank;
    int max_sent_len;
    
    bool update_word;
    bool debug;
    
    real* grad_emb_map;
    real* emb_map;
    real* params_g;
    real* word_repr;
    real* feat_repr;
    real* label_repr;
    real* word_repr_sum;
    real* feat_repr_sum;
    
    real* emb_input; //O_i = \sum_t Uhwt_i Vhft_i
    real* part_emb_input;
    real* grad_s;
    
    LabelEmbeddingModel* lab_model;
    int lab_emb_dim;
    
    LrTensorModel() {//inst = new LrFcemInstance();
    };
    ~LrTensorModel() {}
    
    LrTensorModel(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model, LabelEmbeddingModel* lab_model, int rank) {
        this -> emb_model = emb_model;
        word_emb_dim = emb_model -> dim;
        this -> fea_model = fea_model;
        this -> lab_model = lab_model;
        this -> rank = rank;
        debug = false;
    }
    
    virtual void GetRepresentations(BaseInstance *b_inst, int position);
    
    void ForwardViews(BaseInstance* b_inst);
    void ForwardOutputs(BaseInstance* b_inst);
    
    long BackPropLabel(BaseInstance* b_inst, real eta_real);
//    void BackPropViews(vector<int>& words, int num_words, vector<int>& feats, int num_feats, real eta_real);
    virtual void BackPropViews(BaseInstance* b_inst, real eta_real);
//    long BackPropOuterProd(BaseInstance* b_inst, real eta_real);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real) {return 0;}
    
    virtual void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    //    void BackPropFea(BaseInstance* b_inst, int word_id, int class_id, int correct, real eta_real);
    
    void PrintModelInfo();
};



#endif
