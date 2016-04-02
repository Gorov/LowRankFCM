//
//  LrTuckerMtl.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-26.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrTuckerMtl_h
#define LR_FCEM_proj_LrTuckerMtl_h

#include "LrTensorTuckerModel.h"

class LrTuckerMtl: public LrTensorTuckerModel
{
public:
    int start[2];
    int length[2];
    LrTuckerMtl():LrTensorTuckerModel() {};
    ~LrTuckerMtl() {}
    
    LrTuckerMtl(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model, LabelEmbeddingModel* lab_model, int rank1, int rank2, int rank3) 
    : LrTensorTuckerModel(fea_model, emb_model, lab_model, rank1, rank2, rank3) {}
    
    void ForwardOutputs(BaseInstance* b_inst, int task_id);
    long BackPropLabel(BaseInstance* b_inst, real eta_real, int task_id);
    
    void ForwardProp(BaseInstance* b_inst, int task_id);
    void BackProp(BaseInstance* b_inst, real eta_real, int task_id);
    
    void PrintModelInfo();
};

#endif
