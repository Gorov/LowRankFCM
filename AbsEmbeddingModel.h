//
//  AbsEmbeddingModel.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_AbsEmbeddingModel_h
#define LR_FCEM_proj_AbsEmbeddingModel_h

#include <tr1/unordered_map>
#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "Commons.h"

using namespace std;

typedef std::tr1::unordered_map<string, int> str2int;
typedef float real;

class AbsEmbeddingModel
{
public:
    str2int vocabdict;
    vector<string> vocablist;
    long vocab_size;
    int dim;
    real *syn0;
    
    real *params_g;
    vector<int> freqlist;
    
    AbsEmbeddingModel() {};
    
    virtual int InitEmb(char* vocabfile, int dim) = 0;
    virtual int LoadEmb(char* modelname) = 0;
    virtual void SaveEmb(char* modelfile) = 0;
};


#endif
