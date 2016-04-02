//
//  LabelEmbeddingModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_LabelEmbeddingModel_h
#define LR_FCEM_proj_LabelEmbeddingModel_h

#include "AbsEmbeddingModel.h"

using namespace std;

typedef str2int label2int;

class LabelEmbeddingModel: public AbsEmbeddingModel
{
public:
    
    vector<int> basefeat_list;
    
    LabelEmbeddingModel() {};
    
    LabelEmbeddingModel(char* modelname) {
        LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    LabelEmbeddingModel(char* modelname, bool unnorm) {
        LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    int AddLabel(string feat);
    int SearchLabel(string feat);
    int LoadEmb(char* modelname);
    int InitEmb(char* freqfile, int dim);
    int InitEmb(int dim);
    int InitEmb();
    void SaveEmb(char* modelfile);
};


#endif
