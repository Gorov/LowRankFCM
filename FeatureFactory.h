//
//  FeatureFactory.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-11.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_FeatureFactory_h
#define LR_FCEM_proj_FeatureFactory_h

#include "PrepInstance.h"
#include "FeatureEmbeddingModel.h"
#include "RunRankingModel.h"

class FeatureFactory {
public:    
    static void ExtractFeatures(BaseInstance* p_inst) {};
};

class PPA_FeatureFactory: FeatureFactory {
public:
    PPA_Params* fea_params;
    FeatureEmbeddingModel* fea_model;
    void Init(FeatureEmbeddingModel* fea_model, PPA_Params* fea_params) {
        this -> fea_model = fea_model;
        this -> fea_params = fea_params;
    }
    
    void ExtractFeatures(PPAInstance* p_inst, bool add);
    void ExtractBigramFeatures(PPAInstance* p_inst, bool add);
    
    int AddWordFeature(PPAInstance* p_inst, string feat_key, int pair_id, int pos);
};


#endif
