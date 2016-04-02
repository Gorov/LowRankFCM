//
//  LrFcemMultitask.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-29.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrFcemMultitask_h
#define LR_FCEM_proj_LrFcemMultitask_h

#include "LrFcemModel.h"

class LrFcemMultitask: public LrFcemModel
{
public:
    int start[2];
    int length[2];
    LrFcemMultitask() {};
    ~LrFcemMultitask() {}
    
    LrFcemMultitask(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model) 
    : LrFcemModel(fea_model, emb_model) {}
    
    void ForwardOutputs(BaseInstance* b_inst, int task_id);
    long BackPropOuterProd(BaseInstance* b_inst, real eta_real, int task_id);
    
    void ForwardProp(BaseInstance* b_inst, int task_id);
    void BackProp(BaseInstance* b_inst, real eta_real, int task_id);
};

#endif
