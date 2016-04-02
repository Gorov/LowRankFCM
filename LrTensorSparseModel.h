//
//  LrTensorSparseModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-10.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LrTensorSparseModel_h
#define LR_FCEM_proj_LrTensorSparseModel_h

#include "BaseComponentModel.h"
#include "LabelEmbeddingModel.h"
#include "LrTensorModel.h"

//typedef std::tr1::unordered_map<int, int> int2int;

class LrSparseTensorModel: public LrTensorModel
{
public:
    int2int* feat_group;
    int group_size;
    
    int num_group;
    bool sparse_map;
    
    LrSparseTensorModel() {//inst = new LrFcemInstance();
    };
    ~LrSparseTensorModel() {}
    
    LrSparseTensorModel(FeatureEmbeddingModel* fea_model, WordEmbeddingModel* emb_model, LabelEmbeddingModel* lab_model, int2int* feat_group, int rank, int group_size)
    :LrTensorModel(fea_model, emb_model, lab_model, rank) {
        this -> feat_group = feat_group;
        this -> group_size = group_size;
    }
    
    void InitModel();
    
    void GetRepresentations(BaseInstance *b_inst, int position);
    void BackPropViews(BaseInstance* b_inst, real eta_real);
    
    void PrintModelInfo();
};

#endif
