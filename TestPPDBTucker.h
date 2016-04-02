//
//  TestPPDBTucker.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-7.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_TestPPDBTucker_h
#define LR_FCEM_proj_TestPPDBTucker_h

#include "TestTuckerModel.h"
#include "LrTuckerNCE.h"

class FeaParams_NCE : public FeaParams
{
    public:
    int num_neg;
    FeaParams_NCE():FeaParams() {}
    void PrintValue() {
        FeaParams::PrintValue();
        cout << "num_neg" << num_neg << endl;
    }
};

class TestPPDBTucker : public TestTuckerModel
{
public:
    int num_neg;
    
    //for NCE
    word2int word_dict;
    vector<string> vocab;
    unsigned long long next_random;
    const int table_size = 1e8;
    int *table;
    long vocab_size;
    int* freqtable;
    
    TestPPDBTucker() : TestTuckerModel() {}
    ~TestPPDBTucker() {}
    
    TestPPDBTucker(char* embfile, char* trainfile, FeaParams_NCE* params) {
        type = "SEM_EVAL";
        inst = new LrFcemInstance();
        Init(embfile, trainfile, params);
        debug = false;
    }
    
    void BuildModelsFromData(char* trainfile);
    
    int LoadInstance(ifstream& ifs);
    int LoadInstanceInit(ifstream& ifs);
    
    int AddWordFeature(string feat_key, int pos);
    
    void Init(char* embfile, char* traindata);
    void Init(char* embfile, char* trainfile, FeaParams_NCE* params);
    
    void ForwardProp();
    void BackProp();
    
    void PrintInstance(ofstream& ofs, LrFcemInstance* p_inst);
    
    virtual void TrainData(string trainfile, string devfile, int type);
    virtual void EvalData(string trainfile, int type);
    
    string ToLower(string& s);
    
    void SetModels();
    void PrintModelInfo();
    
    void InitFreqTable(char* filename);
    void InitUnigramTable();
    long SampleNegative();
};


#endif
