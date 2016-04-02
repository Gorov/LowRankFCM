//
//  Instances.h
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#ifndef RE_FCT_Instances_h
#define RE_FCT_Instances_h

#include <vector>
#include <sstream>
#include <fstream>
#include <string.h>
#include "WordEmbeddingModel.h"
using namespace std;

#define MAX_SENT_LEN 300
#define MAX_FEAT_LEN 200

class BaseInstance{
public:
    string label;
    int label_id;
    int len;
    
    string entitytype;
    
    vector<string> words;
    vector<string> tags;
    vector<string> word_nes;
    vector<string> word_types;
    vector<int> dep_paths;
    vector<string> dep_labels;
    vector<string> dirs;
    vector<int> word_ids;
    vector<real> scores;
    
    vector<string> ne1_words;
    vector<int> ne1_ids;
    int ne1_len;
    vector<string> ne2_words;
    vector<int> ne2_ids;
    int ne2_len;
    vector<string> ne1_types;
    vector<string> ne2_types;
    vector<string> ne1_nes;
    vector<string> ne2_nes;
    vector<int> ne1_type_ids;
    vector<int> ne2_type_ids;
    
    string path;
    int frame_id;
    
    BaseInstance() {
        words.resize(MAX_SENT_LEN);
        tags.resize(MAX_SENT_LEN);
        word_nes.resize(MAX_SENT_LEN);
        word_types.resize(MAX_SENT_LEN);
        dep_labels.resize(MAX_SENT_LEN);
        dep_paths.resize(MAX_SENT_LEN);
        dirs.resize(MAX_SENT_LEN);
        word_ids.resize(MAX_SENT_LEN);
        ne1_words.resize(MAX_SENT_LEN);
        ne2_words.resize(MAX_SENT_LEN);
        ne1_types.resize(MAX_SENT_LEN);
        ne2_types.resize(MAX_SENT_LEN);
        ne1_nes.resize(MAX_SENT_LEN);
        ne2_nes.resize(MAX_SENT_LEN);
        ne1_ids.resize(MAX_SENT_LEN);
        ne2_ids.resize(MAX_SENT_LEN);
        ne1_type_ids.resize(MAX_SENT_LEN);
        ne2_type_ids.resize(MAX_SENT_LEN);
    }
};

class LrFcemInstance : public BaseInstance {
public:
//    vector<string> words;
//    vector<int> word_ids;
//    //    int len;
//    int count;
//    
//    BaseInstance* p_base_inst;
    
    vector<vector<int> > fct_fea_ids;
    vector<int> fct_nums_fea;
    
    LrFcemInstance():BaseInstance() {
        words.resize(MAX_SENT_LEN);
        word_ids.resize(MAX_SENT_LEN);
        fct_fea_ids.resize(MAX_SENT_LEN);
        fct_nums_fea.resize(MAX_SENT_LEN);
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            fct_fea_ids[i].resize(20);
        }
    }
    void Clear() {
        len = ne1_len = ne2_len = 0;
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            fct_nums_fea[i] = 0;
        }
    }
    
    void PushFctFea(int fea_id, int pos) {
        fct_fea_ids[pos][fct_nums_fea[pos]] = fea_id;
        fct_nums_fea[pos]++;
    }
};

#endif
