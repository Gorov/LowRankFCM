//
//  LrTuckerNCE.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-7.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrTuckerNCE_h
#define LR_FCEM_proj_LrTuckerNCE_h

#include "LrTensorTuckerModel.h"

class LrTuckerNCE: public LrTensorTuckerModel
{
public:
    int num_neg;
    vector<int> neg_samples;
    
    LrTuckerNCE() {//inst = new LrFcemInstance();
    };
    ~LrTuckerNCE() {}
    
    LrTuckerNCE(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model, LabelEmbeddingModel* lab_model, int rank1, int rank2, int rank3)
    :LrTensorTuckerModel(fea_model, emb_model, lab_model, rank1, rank2, rank3){
        debug = false;
    }
    
    void ForwardOutputs(BaseInstance* b_inst);
    
    long BackPropLabel(BaseInstance* b_inst, real eta_real);
    
    virtual void InitModel();
    
    void PrintModelInfo();
};


#endif
