//
//  FeatureEmbedingModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_FeatureEmbedingModel_h
#define LR_FCEM_proj_FeatureEmbedingModel_h

#include "AbsEmbeddingModel.h"

using namespace std;

typedef str2int feat2int;
typedef std::tr1::unordered_map<int, int> int2int;

class FeatureEmbeddingModel: public AbsEmbeddingModel
{
public:
    
    vector<int> basefeat_list;
    
    //for sparse model
//    int2int* feat_group;
//    int num_group;
//    int2int group_dict;
    
    FeatureEmbeddingModel() {};
    
    FeatureEmbeddingModel(char* modelname) {
        LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    FeatureEmbeddingModel(char* modelname, bool unnorm) {
        LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    int AddFeature(string feat);
//    int AddFeature(string feat, int group);
    int SearchFeature(string feat);
    int LoadEmb(char* modelname);
    int InitEmb(char* freqfile, int dim);
    int InitEmb(int dim);
    int InitEmb();
    void SaveEmb(char* modelfile);
};


#endif
