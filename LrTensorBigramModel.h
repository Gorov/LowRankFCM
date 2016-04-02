//
//  LrTensorBigramModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-24.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrTensorBigramModel_h
#define LR_FCEM_proj_LrTensorBigramModel_h

#include "BaseComponentModel.h"
#include "LabelEmbeddingModel.h"

class LrTensorBigramModel: public BaseComponentModel
{
public:
    //int rank1, rank2, rank3;
    int rank1, rank3;
    int max_sent_len;
    
    bool update_word;
    bool update_lab_emb;
    bool debug;
    
    real* core_tensor;
    real* core_params_g;
    
    vector<int> rank_views;
    
    vector<real*> emb_maps;
    real* emb_map;
    real* params_g;
    vector<real*> word_repr_views;
//    real* word_repr;
//    real* feat_repr;
    real* label_repr;
    vector<real*> word_repr_sum_views;
//    real* word_repr_sum;
//    real* feat_repr_sum;
    
    real* emb_input; //O_i = \sum_t Uhwt_i Vhft_i
    real* part_emb_input;
    real* grad_s;
    
    real* structure_emb; //\sum_n e_n \otimes f_n emb_{ij} = \sum_n f_{ni} * e_{nj}
    real* part_struct_emb;
    
    LabelEmbeddingModel* lab_model;
    int lab_emb_dim;
    
    LrTensorBigramModel() {//inst = new LrFcemInstance();
    };
    ~LrTensorBigramModel() {}
    
    LrTensorBigramModel(WordEmbeddingModel* emb_model, LabelEmbeddingModel* lab_model, int rank1, int rank3) {
        this -> emb_model = emb_model;
        word_emb_dim = emb_model -> dim;
        this -> fea_model = NULL;
        this -> lab_model = lab_model;
        this -> rank1 = rank1;
//        this -> rank2 = rank2;
        this -> rank3 = rank3;
        debug = false;
    }
    
    virtual void GetRepresentations(BaseInstance *b_inst, int position);
    
    void ForwardViews(BaseInstance* b_inst);
    void ForwardStructureEmb(BaseInstance* b_inst);
    void ForwardOutputs(BaseInstance* b_inst);
    
    long BackPropLabel(BaseInstance* b_inst, real eta_real);
    virtual void BackPropViews(BaseInstance* b_inst, real eta_real);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real) {return 0;}
    
    virtual void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    
    void PrintModelInfo();
};


#endif
