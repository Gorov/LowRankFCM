//
//  PrepInstance.h
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-18.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#ifndef LR_FCEM_proj_PrepInstance_h
#define LR_FCEM_proj_PrepInstance_h

#include "Instances.h"
#define MAX_FEAVEC_LEN 50
class PrepInstance: public LrFcemInstance{
public:
    //    string label;
    //    int label_id;
    //    
    //    int len;
    vector<string> clus;
    string prep_word;
    //    string entitytype;
    
    //    vector<string> words;
    //    vector<int> word_ids;
    //    vector<real> scores;
    
    PrepInstance()
    :LrFcemInstance(){
        clus.resize(MAX_SENT_LEN);
    }
};

class PrepBigramInstance: public PrepInstance{
public:
    //    string label;
    //    int label_id;
    //    
    //    int len;

    vector<pair<int, int> > word_pairs;
    vector<vector<int> > word_pair_feats;
    int num_pairs;
    //    string entitytype;
    
    //    vector<string> words;
    //    vector<int> word_ids;
    //    vector<real> scores;
    
    PrepBigramInstance()
    :PrepInstance(){
        word_pairs.resize(MAX_SENT_LEN);
    }
};

class PPAInstance: public LrFcemInstance{
public:
    vector<string> clus;
    string prep_word;
    
    vector<vector<string> > word_pairs;
    vector<vector<int> > id_pairs;
    
    vector<vector<vector<int> > > fea_vec_pairs;
    vector<vector<int> > fea_num_pairs;
    int list_len;
    int label_pos;
    
    string child_clus;
    vector<string> nextpos;
    vector<string> headpos;
    vector<int> verbnet_len;
    vector<vector<string> > preps;
    
    
    PPAInstance()
    :LrFcemInstance(){
        clus.resize(MAX_SENT_LEN);
        word_pairs.resize(MAX_SENT_LEN);
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            word_pairs[i].resize(2);
        }
        id_pairs.resize(MAX_SENT_LEN);
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            id_pairs[i].resize(2);
        }
        
        nextpos.resize(MAX_SENT_LEN);
        headpos.resize(MAX_SENT_LEN);
        verbnet_len.resize(MAX_SENT_LEN);
        preps.resize(MAX_SENT_LEN);
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            preps[i].resize(10);
        }
        
        
        fea_vec_pairs.resize(MAX_SENT_LEN);
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            fea_vec_pairs[i].resize(2);
            fea_vec_pairs[i][0].resize(MAX_FEAVEC_LEN);
            fea_vec_pairs[i][1].resize(MAX_FEAVEC_LEN);
        }
        
        fea_num_pairs.resize(MAX_SENT_LEN);
        for (int i = 0; i < MAX_SENT_LEN; i++) {
            fea_num_pairs[i].resize(2);
        }
    }
    
    void Clear() {
        len = ne1_len = ne2_len = 0;
        for (int pair_id = 0; pair_id < MAX_SENT_LEN; pair_id++) {
            fea_num_pairs[pair_id][0] = 0;
            fea_num_pairs[pair_id][1] = 0;
        }
    }
    
    void PushFctFea(int fea_id, int pair_id, int pos) {
        fea_vec_pairs[pair_id][pos][fea_num_pairs[pair_id][pos]] = fea_id;
        fea_num_pairs[pair_id][pos]++;
    }
};
#endif
