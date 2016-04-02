//
//  TestTuckerMtl.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-26.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_TestTuckerMtl_h
#define LR_FCEM_proj_TestTuckerMtl_h

#include "TestTuckerModel.h"
#include "LrTuckerMtl.h"

class TestTuckerMtl: public TestTuckerModel
{
public:
    feat2int labeldicts[2];
    vector<string> labellists[2];
    int num_labels[2];
    
    TestTuckerMtl():TestTuckerModel() {}
    ~TestTuckerMtl() {}
    
    TestTuckerMtl(char* embfile, char* trainfile, FeaParams* params){
        type = "SEM_EVAL";
        inst = new LrFcemInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    
    void BuildModelsFromData(char* trainfile);
    
    void Init(char* embfile, char* trainfile, FeaParams* params);
    
    void ForwardProp(int task_id);
    void BackProp(int task_id);
    
    void TrainData(string trainfile, string devfile, int type);
    void EvalData(string trainfile, int type);
    void EvalData(string trainfile, int type, int task_id);
    
    void SetModels();
    void PrintModelInfo();
};


#endif
