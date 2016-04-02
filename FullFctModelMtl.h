//
//  FullFctModelMtl.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-29.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_FullFctModelMtl_h
#define LR_FCEM_proj_FullFctModelMtl_h

#include "LrFcemMultitask.h"
#include "FullFctModel.h"

class FullFctModelMtl: public FullFctModel
{
public:
    feat2int labeldicts[2];
    vector<string> labellists[2];
    vector<LrFcemMultitask*> lr_fcem_mtl_list;
    
    int num_labels[2];
    
    FullFctModelMtl(): FullFctModel() {}
    ~FullFctModelMtl() {}
    
    FullFctModelMtl(char* embfile, char* trainfile){
        type = "SEM_EVAL";
        inst = new LrFcemInstance();
        Init(embfile, trainfile);
    }
//    FullFctModelMtl(char* embfile, char* trainfile, char* configfile)
//    : FullFctModel(embfile, trainfile, configfile) {}
    
    void BuildModelsFromData(char* trainfile);
    
    void Init(char* embfile, char* traindata);
    //    void Init(char* embfile, char* trainfile, int type);
    void TrainData(string trainfile, string devfile, int type);
    void EvalData(string trainfile, int type);
    void EvalData(string trainfile, string outfile, int type) {}
    
    void ForwardProp(int task_id);
    void BackProp(int task_id);
    
    void SetModels();
    void PrintModelInfo();
};


#endif
