//
//  RunCombinedRanker.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-11.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "RunCombinedRanker.h"

void RunCombinedRanker::Init(char* embfile, char* trainfile, PPA_Params* params)
{
    enable_model_1 = true;
    enable_model_2 = false;
    
    fea_params[0].sum = fea_params[1].sum = true;
    
    fea_params[0].position = fea_params[1].position = true;
    fea_params[0].prep = fea_params[1].prep = true;
    fea_params[0].postag = fea_params[1].postag = false;
    
    fea_params[0].clus = fea_params[1].clus = params -> clus;
    //fea_params[0].clus = true;
    //fea_params[1].clus = false;
    
    fea_params[0].nextpos = fea_params[1].nextpos = params -> nextpos;
    fea_params[0].verbnet = fea_params[1].verbnet = params -> verbnet;
    //fea_params[0].verbnet = false;
    //fea_params[1].verbnet = true;
    fea_params[0].wordnet = fea_params[1].wordnet = params -> wordnet;
    
    fea_params[0].fea_dim = fea_params[1].fea_dim = params -> fea_dim;
    
    fea_params[0].rank1 = fea_params[1].rank1 = params -> rank1;
    fea_params[0].rank2 = fea_params[1].rank2 = params -> rank2;
	fea_params[0].rank2 = fea_params[0].rank1;
    fea_params[0].rank3 = fea_params[1].rank3 = params -> rank3 = 1;
    
    emb_model = new WordEmbeddingModel(embfile);
    
    BuildModelsFromData(trainfile);
    num_labels = (int)labeldict.size();
    inst[0] -> scores.resize(max_len);
    inst[1] -> scores.resize(max_len);
    
    alpha = 1.0;
    lambda = 0.0;
}

void RunCombinedRanker::BuildModelsFromData(char* trainfile) {
    
    fea_model[0] = new FeatureEmbeddingModel();
    fea_model[1] = new FeatureEmbeddingModel();
    lab_model = new LabelEmbeddingModel();
    
    fea_factory[0].Init(fea_model[0], &fea_params[0]);
    fea_factory[1].Init(fea_model[1], &fea_params[1]);
    
    feat_group = new int2int();
    num_group = 0;
    group_dict.clear();
    
    ifstream ifs(trainfile);
    num_inst = 0;
    while (LoadInstanceInit(ifs)) {
        num_inst++;
    }
    
    num_feats = (int)fea_model[0] -> vocabdict.size();
    //    fea_model -> InitEmb(fea_params.fea_dim);
    fea_model[0] -> InitEmb(fea_params[0].rank2);
    fea_model[1] -> InitEmb();
    //fea_params[0].rank2 = (int)fea_model[0] -> vocabdict.size();
    fea_params[1].rank2 = (int)fea_model[1] -> vocabdict.size();
    
    num_labels = (int)lab_model -> vocabdict.size();
    lab_model -> InitEmb();
    
    num_labels = (int)labeldict.size();
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
    
    feat_emb_dim[0] = fea_model[0] -> dim;
    feat_emb_dim[1] = fea_model[1] -> dim;
    word_emb_dim = emb_model -> dim;
    if (enable_model_1)
    {
        LrRankingModel* pmodel = new LrRankingModel(fea_model[0], emb_model, lab_model, fea_params[0].rank1);
        //LrRankingModel* pmodel = new LrRankingModel(fea_model[0], emb_model, lab_model, fea_params[0].rank1, fea_params[0].rank2);
        
        pmodel -> max_list_len = max_len;
        pmodel -> max_sent_len = 3;
        pmodel -> num_labels = num_labels;
        pmodel -> num_fea = num_feats;
        pmodel -> update_word = update_emb;
        pmodel -> InitModel();
        
        lr_bigram_list.push_back(pmodel);
    }
    if (enable_model_2)
    {
        LrTuckerRankingModel* pmodel = new LrTuckerRankingModel(fea_model[1], emb_model, lab_model, fea_params[1].rank1, fea_params[1].rank2, fea_params[1].rank3);
        
        pmodel -> max_list_len = max_len;
        pmodel -> max_sent_len = 3;
        pmodel -> num_labels = num_labels;
        pmodel -> num_fea = num_feats;
        pmodel -> update_word = update_emb;
        pmodel -> InitModel();
        
        lr_unigram_list.push_back(pmodel);
    }
    
    num_models = (int)lr_bigram_list.size() + (int)lr_unigram_list.size();
}

int RunCombinedRanker::LoadInstanceInit(ifstream &ifs) {
    int ret = LoadInstanceOnly(ifs, true);
    if (ret == 0) return 0;
    
    PPAInstance* p_inst[2];
    p_inst[0] = (PPAInstance*) inst[0];
    p_inst[1] = (PPAInstance*) inst[1];
    if (enable_model_1) fea_factory[0].ExtractBigramFeatures(p_inst[0], true);
    if (enable_model_2) fea_factory[1].ExtractFeatures(p_inst[1], true);
    
    return 1;
}

int RunCombinedRanker::LoadInstanceOnly(ifstream &ifs, bool add) {
    word2int::iterator iter2;
    string token, clus;
    char line_buf[5000], line_buf2[2000];
    vector<string> preps;
    preps.resize(10);
    vector<string> senses;
    senses.resize(100);
    vector<string> head_senses;
    head_senses.resize(100);
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    PPAInstance* p_inst[2];
    p_inst[0] = (PPAInstance*) inst[0];
    p_inst[1] = (PPAInstance*) inst[1];
    p_inst[0] -> Clear();
    p_inst[1] -> Clear();
    {
        istringstream iss(line_buf);
        iss >> token;
        ToLower(token);
        inst[0] -> words[0] = inst[1] -> words[0] = token;
        
        if (!add) {
            iter2 = emb_model -> vocabdict.find(inst[0] -> words[0]);
            if (iter2 != emb_model -> vocabdict.end()) {
                inst[0] -> word_ids[0] = iter2 -> second;
                inst[1] -> word_ids[0] = iter2 -> second;
            }
            else {
                inst[0] -> word_ids[0] = -1;
                inst[1] -> word_ids[0] = -1;
            }
        }
        
        if (fea_params[0].clus ||fea_params[1].clus ) {
            iss >> p_inst[0] -> child_clus;
            p_inst[1] -> child_clus = p_inst[0] -> child_clus; 
        }
        
        iss >> p_inst[0] -> prep_word;
        p_inst[1] -> prep_word = p_inst[0] -> prep_word;
        
        iss >> p_inst[0] -> list_len;
        p_inst[1] -> list_len = p_inst[0] -> list_len;
    }
    for (int i = 0; i < p_inst[0] -> list_len; i++)
    {
        ifs.getline(line_buf, 5000, '\n');
        istringstream iss2(line_buf);
        string token;
        int label;
        
        iss2 >> label;
        if (label == 1) inst[0] -> label_id = inst[1] -> label_id = p_inst[0] -> label_pos = p_inst[1] -> label_pos = i;
        
        p_inst[0] -> word_pairs[i][0] = p_inst[1] -> word_pairs[i][0] = inst[0] -> words[0];
        
        if (!add) {
            p_inst[0] -> id_pairs[i][0] = inst[0] -> word_ids[0];
            p_inst[1] -> id_pairs[i][0] = inst[1] -> word_ids[0];
        }
        
        iss2 >> token;
        ToLower(token);
        p_inst[0] -> word_pairs[i][1] = token;
        p_inst[1] -> word_pairs[i][1] = token;
        
        if (!add) {
            iter2 = emb_model -> vocabdict.find(token);
            if (iter2 != emb_model -> vocabdict.end()) {
                p_inst[0] -> id_pairs[i][1] = iter2 -> second;
                p_inst[1] -> id_pairs[i][1] = iter2 -> second;
            }
            else {
                p_inst[0] -> id_pairs[i][1] = -1;
                p_inst[1] -> id_pairs[i][1] = -1;
            }
        }
        
        if (fea_params[0].clus || fea_params[1].clus ) {
            iss2 >> p_inst[0] -> clus[i];
            p_inst[1] -> clus[i] = p_inst[0] -> clus[i];
        }
        
        if (fea_params[0].nextpos || fea_params[1].nextpos) {
            iss2 >> p_inst[0] -> nextpos[i];
            p_inst[1] -> nextpos[i] = p_inst[0] -> nextpos[i];
        }
        
        if (fea_params[0].verbnet || fea_params[1].verbnet) {
            ifs.getline(line_buf2, 2000, '\n');
            istringstream iss3(line_buf2);
            iss3 >> p_inst[0] -> verbnet_len[i];
            p_inst[1] -> verbnet_len[i] = p_inst[0] -> verbnet_len[i];
            if (p_inst[1] -> verbnet_len[i] == -1) {
                p_inst[0] -> headpos[i] = p_inst[1] -> headpos[i] = "N";    
            }
            else {
                p_inst[0] -> headpos[i] = p_inst[1] -> headpos[i] = "V";
                for (int j = 0; j < p_inst[0] -> verbnet_len[i]; j++) {
                    iss3 >> p_inst[0] -> preps[i][j];
                    p_inst[1] -> preps[i][j] = p_inst[0] -> preps[i][j];
                }
            }
        }
        
        inst[0] -> len = p_inst[0] -> list_len;
        inst[1] -> len = p_inst[1] -> list_len;
        if (inst[0] -> len > max_len) {max_len = inst[0] -> len;}
    }
    
    return 1;
}

int RunCombinedRanker::LoadInstance(ifstream &ifs) {
    int ret = LoadInstanceOnly(ifs, false);
    if (ret == 0) return 0;
    
    PPAInstance* p_inst[2];
    p_inst[0] = (PPAInstance*) inst[0];
    p_inst[1] = (PPAInstance*) inst[1];
    if (enable_model_1) fea_factory[0].ExtractBigramFeatures(p_inst[0], false);
    if (enable_model_2) fea_factory[1].ExtractFeatures(p_inst[1], false);
    
    return 1;    
}


string RunCombinedRanker::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

void RunCombinedRanker::ForwardProp()
{
    real sum;
    int c;
    PPAInstance* p_inst = (PPAInstance*) inst[0];
    for (int i = 0; i < p_inst -> list_len; i++) {
        inst[0] -> scores[i] = 0.0;
    }
    for (int i = 0; i < p_inst -> list_len; i++) {
        inst[1] -> scores[i] = 0.0;
    }
    if (enable_model_1) 
    for (int i = 0; i < lr_bigram_list.size(); i++) {
        lr_bigram_list[i] -> ForwardProp(inst[0]);
    }
    if (enable_model_2) 
    for (int i = 0; i < lr_unigram_list.size(); i++) {
        lr_unigram_list[i] -> ForwardProp(inst[1]);
    }
    
    for (c = 0; c < p_inst -> list_len; c++) {
        inst[0] -> scores[c] += inst[1] -> scores[c];
        //cout << c << ":" << inst[0] -> scores[c] << " " << inst[1] -> scores[c];
    }
    //cout << endl;
    
    if (debug) {
        for (c = 0; c < p_inst -> list_len; c++) {
            cout << c << ":" << inst[0] -> scores[c];
        }
        cout << endl;
    }
    
    sum = 0.0;
    for (c = 0; c < p_inst -> list_len; c++) {
        float tmp;
        if (p_inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (p_inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(p_inst -> scores[c]);
        p_inst -> scores[c] = tmp;
        sum += p_inst -> scores[c];
        
    }
    for (c = 0; c < p_inst -> list_len; c++) {
        inst[0] -> scores[c] /= sum;
        inst[1] -> scores[c] = inst[0] -> scores[c];
    }
}

void RunCombinedRanker::BackProp()
{
    alpha_old = alpha;
    real eta_real = eta / alpha;
    if (enable_model_1) 
    for (int i = 0; i < lr_bigram_list.size(); i++) {
        lr_bigram_list[i] -> BackProp(inst[0], eta_real);
    }
    if (enable_model_2) 
    for (int i = 0; i < lr_unigram_list.size(); i++) {
        lr_unigram_list[i] -> BackProp(inst[1], eta_real);
    }
}

void RunCombinedRanker::TrainData(string trainfile, string devfile) {
    if (enable_model_1) 
    if (lr_bigram_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_bigram_list[0] -> vec_emb_map[0][0]);
        printf("L-emb2: %lf\n", lr_bigram_list[0] -> vec_emb_map[0][fea_params[0].rank1]);
        printf("L-emb1: %lf\n", lr_bigram_list[0] -> vec_emb_map[1][0]);
        printf("L-emb2: %lf\n", lr_bigram_list[0] -> vec_emb_map[1][fea_params[0].rank1]);
        printf("L-emb1: %lf\n", lr_bigram_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_bigram_list[0] -> fea_model -> syn0[fea_params[0].rank2]);
        printf("core-tensor: %lf\n", lr_bigram_list[0] -> core_tensor[0]);
    }
    if (enable_model_2) 
        if (lr_unigram_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_unigram_list[0] -> emb_map[0]);
            printf("L-emb2: %lf\n", lr_unigram_list[0] -> emb_map[fea_params[1].rank1]);
            printf("L-emb1: %lf\n", lr_unigram_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_unigram_list[0] -> fea_model -> syn0[fea_params[1].rank2]);
            printf("core-tensor: %lf\n", lr_unigram_list[0] -> core_tensor[0]);
        }
    int count = 0;
    int total = num_inst * iter;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        while (LoadInstance(ifs)) {
            ForwardProp();
            BackProp();
            count++;
            if (count % 4000 == 0) EvalData(devfile);
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        
        if (enable_model_1) 
        if (lr_bigram_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_bigram_list[0] -> vec_emb_map[0][0]);
            printf("L-emb2: %lf\n", lr_bigram_list[0] -> vec_emb_map[0][fea_params[0].rank1]);
            printf("L-emb1: %lf\n", lr_bigram_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_bigram_list[0] -> fea_model -> syn0[fea_params[0].rank2]);
            printf("core-tensor: %lf\n", lr_bigram_list[0] -> core_tensor[0]);
        }
        if (enable_model_2) 
            if (lr_unigram_list.size() != 0) {
                printf("L-emb1: %lf\n", lr_unigram_list[0] -> emb_map[0]);
                printf("L-emb2: %lf\n", lr_unigram_list[0] -> emb_map[fea_params[1].rank1]);
                printf("L-emb1: %lf\n", lr_unigram_list[0] -> fea_model -> syn0[0]);
                printf("L-emb2: %lf\n", lr_unigram_list[0] -> fea_model -> syn0[fea_params[1].rank2]);
                printf("core-tensor: %lf\n", lr_unigram_list[0] -> core_tensor[0]);
            }
        EvalData(trainfile);
        EvalData(devfile);
    }
    if (enable_model_1) 
        if (lr_bigram_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_bigram_list[0] -> vec_emb_map[0][0]);
            printf("L-emb2: %lf\n", lr_bigram_list[0] -> vec_emb_map[0][fea_params[0].rank1]);
            printf("L-emb1: %lf\n", lr_bigram_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_bigram_list[0] -> fea_model -> syn0[fea_params[0].rank2]);
            printf("core-tensor: %lf\n", lr_bigram_list[0] -> core_tensor[0]);
        }
    if (enable_model_2) 
        if (lr_unigram_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_unigram_list[0] -> emb_map[0]);
            printf("L-emb2: %lf\n", lr_unigram_list[0] -> emb_map[fea_params[1].rank1]);
            printf("L-emb1: %lf\n", lr_unigram_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_unigram_list[0] -> fea_model -> syn0[fea_params[1].rank2]);
            printf("core-tensor: %lf\n", lr_unigram_list[0] -> core_tensor[0]);
        }
}

void RunCombinedRanker::EvalClosest(string trainfile) {
    int total = 0;
    int right = 0;
    real acc = 0.0;
    ifstream ifs(trainfile.c_str());
    
    PPAInstance* p_inst = (PPAInstance*) inst[0];
    while (LoadInstance(ifs)) {
        if (p_inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        
        if (p_inst -> list_len - 1 == p_inst -> label_id) {
            right++;
        }
    }
    acc = real(right) / total;
    
    cout << std::setprecision(4) << right << "\t" << total << "\t" << acc * 100 << endl;
    ifs.close();
}

void RunCombinedRanker::EvalData(string trainfile) {
    int total = 0;
    int right = 0;
    real acc = 0.0;
    double max, max_p;
    
    PPAInstance* p_inst = (PPAInstance*) inst[0];
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        if (inst[0] -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 0; i < p_inst -> list_len; i++){
            if (inst[0] -> scores[i] > max) {
                max = inst[0] -> scores[i];
                max_p = i;
            }
        }
        if (max_p == inst[0] -> label_id) {
            right++;
        }
    }
    acc = real(right) / total;
    
    cout << std::setprecision(4) << right << "\t" << total << "\t" << acc * 100 << endl;
    ifs.close();
}

void RunCombinedRanker::SetModels() {
    cout << "Adagrad: " << adagrad << endl;
    cout << "update_emb: " << update_emb << endl;
    cout << "update_feat_emb: " << update_feat_emb << endl;
    cout << "update_lab_emb: " << update_lab_emb << endl;
    if (enable_model_1) 
    for (int i = 0; i < lr_bigram_list.size(); i++) {
        lr_bigram_list[i] -> adagrad = adagrad;
        lr_bigram_list[i] -> lambda = lambda;
        lr_bigram_list[i] -> update_emb = true;//update_emb;
        lr_bigram_list[i] -> update_word = true;//update_emb;
        lr_bigram_list[i] -> update_fea_emb = update_feat_emb;
        lr_bigram_list[i] -> update_lab_emb = update_lab_emb;
        lr_bigram_list[i] -> debug = debug;
    }
    if (enable_model_2) 
    for (int i = 0; i < lr_unigram_list.size(); i++) {
        lr_unigram_list[i] -> adagrad = adagrad;
        lr_unigram_list[i] -> lambda = lambda;
        lr_unigram_list[i] -> update_emb = update_emb;
        lr_unigram_list[i] -> update_word = update_emb;
        lr_unigram_list[i] -> update_fea_emb = update_feat_emb;
        lr_unigram_list[i] -> update_lab_emb = update_lab_emb;
        lr_unigram_list[i] -> debug = debug;
    }
}

void RunCombinedRanker::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    if (fea_model[0] != NULL) cout << "Number of Features: " << fea_model[0] -> vocab_size << endl;
    if (fea_model[1] != NULL) cout << "Number of Features: " << fea_model[1] -> vocab_size << endl;
    cout << "Max length of sentences: " << max_len << endl;
    if (enable_model_1) 
    for (int i = 0; i < lr_bigram_list.size(); i++) {
        lr_bigram_list[i] -> PrintModelInfo();
    }
    if (enable_model_2) 
    for (int i = 0; i < lr_unigram_list.size(); i++) {
        lr_unigram_list[i] -> PrintModelInfo();
    }
}
