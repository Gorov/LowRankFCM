//
//  LrTuckerRankingModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-29.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrTuckerRankingModel_h
#define LR_FCEM_proj_LrTuckerRankingModel_h

#include "PrepInstance.h"
#include "LrTensorTuckerModel.h"

class LrTuckerRankingModel: public BaseComponentModel
{
public:
    int rank1, rank2, rank3;
    int max_sent_len;
    
    int max_list_len;
    
    bool update_word;
    bool update_lab_emb;
    bool debug;
    
    real* core_tensor;
    real* core_params_g;
    
    //    real* grad_emb_map;
    real* emb_map;
    real* params_g;
    
    vector<real*> vec_word_repr;
    vector<real*> vec_feat_repr;
    real* label_repr;
    vector<real*> vec_word_repr_sum;
    vector<real*> vec_feat_repr_sum;
    
    real* emb_input; //O_i = \sum_t Uhwt_i Vhft_i
    real* part_emb_input;
    
    vector<real*> vec_structure_emb; //\sum_n e_n \otimes f_n emb_{ij} = \sum_n f_{ni} * e_{nj}
    vector<real*> vec_part_struct_emb;
    
    LabelEmbeddingModel* lab_model;
    int lab_emb_dim;
    
    LrTuckerRankingModel() {//inst = new LrFcemInstance();
    };
    ~LrTuckerRankingModel() {}
    
    LrTuckerRankingModel(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model, LabelEmbeddingModel* lab_model, int rank1, int rank2, int rank3) {
        this -> emb_model = emb_model;
        word_emb_dim = emb_model -> dim;
        this -> fea_model = fea_model;
        this -> lab_model = lab_model;
        this -> rank1 = rank1;
        this -> rank2 = rank2;
        this -> rank3 = rank3;
        debug = false;
    }
    
    virtual void GetRepresentations(BaseInstance *b_inst, int position) {}
    void GetRepresentations(BaseInstance *b_inst, int pair_id, int position);
    
    void ForwardViews(BaseInstance* b_inst, int pair_id);
    void ForwardStructureEmb(BaseInstance* b_inst, int pair_id);
    void ForwardOutputs(BaseInstance* b_inst);
    
    long BackPropLabel(BaseInstance* b_inst, real eta_real);
    virtual void BackPropViews(BaseInstance* b_inst, real eta_real) {}
    void BackPropViews(BaseInstance* b_inst, real eta_real, int pair_id);
    long BackPropPhrase(BaseInstance* b_inst, real eta_real) {return 0;}
    
    virtual void InitModel();
    
    void ForwardProp(BaseInstance* b_inst);
    void BackProp(BaseInstance* b_inst, real eta_real);
    
    void PrintModelInfo();
};

#endif
