//
//  WordEmbeddingModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_WordEmbeddingModel_h
#define LR_FCEM_proj_WordEmbeddingModel_h

#include "AbsEmbeddingModel.h"

#define EMB 0
#define HALFLM 1
#define LM 2
#define NORMEMB 3
#define HSLM 4
#define NORMLM 5
#define RANDEMB 10

using namespace std;

typedef str2int word2int;

const string sstags[42] = {
    "stative",
    "possession",
    "all",
    "phenomenon",
    "process",
    "attribute",
    "creation",
    "competition",
    "ppl",
    "motive",
    "shape",
    "perception",
    "relation",
    "event",
    "group",
    "consumption",
    "Tops",
    "0",
    "state",
    "other",
    "location",
    "animal",
    "communication",
    "weather",
    "body",
    "pert",
    "plant",
    "object",
    "food",
    "social",
    "artifact",
    "emotion",
    "change",
    "substance",
    "cognition",
    "act",
    "motion",
    "person",
    "contact",
    "time",
    "feeling",
    "quantity"};

class WordEmbeddingModel: public AbsEmbeddingModel
{
public:
    
    WordEmbeddingModel() {};
    
    WordEmbeddingModel(char* modelname) {
        LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    WordEmbeddingModel(char* modelname, bool unnorm) {
        if (unnorm) LoadEmbUnnorm(modelname);
        else LoadEmb(modelname);
        vocab_size = vocabdict.size();
    }
    
    int LoadEmb(char* modelname);
    int LoadEmbUnnorm(char* modelname);
    int InitEmb(char* freqfile, int dim);
    void SaveEmb(char* modelfile);
    void SaveEmbTxt(char* modelfile, real alpha);
};


#endif
