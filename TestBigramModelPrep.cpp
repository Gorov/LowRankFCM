//
//  TestBigramModelPrep.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-24.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "TestBigramModelPrep.h"

void TestBigramModelPrep::Init(char* embfile, char* trainfile, Prep_Params* params)
{
    fea_params.position = true;
    fea_params.prep = true;
    fea_params.clus = true;
    fea_params.sum = true;
    
    fea_params.low_rank = false;
    
    fea_params.bigram = true;
    
    fea_params.fea_dim = params -> fea_dim;
    
    emb_model = new WordEmbeddingModel(embfile);
    word_emb_dim = emb_model -> dim;
    
    BuildModelsFromData(trainfile);
    num_labels = (int)labeldict.size();
    inst -> scores.resize(num_labels);
    
    alpha = 1.0;
    lambda = 0.0;
}

void TestBigramModelPrep::BuildModelsFromData(char* trainfile) {
    fea_model = new FeatureEmbeddingModel();
    lab_model = new LabelEmbeddingModel();
    
    ifstream ifs(trainfile);
    num_inst = 0;
    while (LoadInstanceInit(ifs)) {
        num_inst++;
    }
    
    num_feats = (int)fea_model -> vocabdict.size();
    fea_model -> InitEmb(fea_params.fea_dim);
    
    num_labels = (int)lab_model -> vocabdict.size();
    lab_model -> InitEmb(num_labels);
    
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
    
    feat_emb_dim = fea_model -> dim;
    word_emb_dim = emb_model -> dim;
    
    //    num_group = group_dict.size();
    LrTensorBigramModel* pmodel = new LrTensorBigramModel(emb_model, lab_model, emb_model -> dim, num_labels);
    
    pmodel -> num_labels = num_labels;
    pmodel -> max_sent_len = 100;
    pmodel -> update_word = update_emb;
    pmodel -> InitModel();
    
    //    lr_fcem_list.push_back(pmodel);
    lr_tensor_list.push_back(pmodel);
    num_models = (int)lr_tensor_list.size();
}

int TestBigramModelPrep::LoadInstanceInit(ifstream &ifs) {
    int id;
    int fea_id;
    char line_buf[5000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) {
            id = (int)labeldict.size();
            labeldict[inst -> label] = id;
        }
        
        iter = lab_model -> vocabdict.find(inst -> label);
        if (iter == lab_model -> vocabdict.end()) {
            id = (int)lab_model -> vocabdict.size();
            lab_model -> vocabdict[inst -> label] = id;
        }
        
        if (fea_params.prep) {
            iss >> ((PrepInstance*)inst) -> prep_word;
        }
        if (fea_params.clus) {
            iss >> inst -> clus[3];
            iss >> inst -> clus[6];
        }
    }
    {
        ifs.getline(line_buf, 5000, '\n');
        istringstream iss2(line_buf);
        int count = 0;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            
            count++;
        }
        
        inst -> len = count + 1;
        if (inst -> len > max_len) {max_len = inst -> len;}
    }
    
    return 1;
}

int TestBigramModelPrep::LoadInstance(ifstream& ifs, int type) {
    return LoadInstance(ifs);
}

int TestBigramModelPrep::LoadInstance(ifstream &ifs) {
    word2int::iterator iter2;
    char line_buf[5000];
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    
    inst -> Clear();
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) inst -> label_id = -1;
        else inst -> label_id = iter -> second;
        
        if (fea_params.prep) {
            iss >> ((PrepInstance*)inst) -> prep_word;
        }
        if (fea_params.clus) {
            iss >> inst -> clus[3];
            iss >> inst -> clus[6];
        }
    }
    {
        ifs.getline(line_buf, 5000, '\n');
        
        istringstream iss2(line_buf);
        int count = 0;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            iter2 = emb_model -> vocabdict.find(token);
            if (iter2 != emb_model -> vocabdict.end()) inst -> word_ids[count] = iter2 -> second;
            else inst -> word_ids[count] = -1;
            
            count++;
        }
        inst -> word_pairs[0].first = inst -> word_ids[3];
        inst -> word_pairs[0].second = inst -> word_ids[6];
        inst -> num_pairs = 1;
        inst -> len = count + 1;
    }
    return 1;
}

int TestBigramModelPrep::AddWordFeature(string feat_key, int pos) {
    int fea_id;
    fea_id = fea_model -> SearchFeature(feat_key);
    if (fea_id == -1) return -1;
    else {
        inst -> PushFctFea(fea_id, pos);
        return fea_id;
    }
}

string TestBigramModelPrep::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

void TestBigramModelPrep::ForwardProp()
{
    real sum;
    int c;
    for (int i = 0; i < num_labels; i++) {
        inst -> scores[i] = 0.0;
    }
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> ForwardProp(inst);
    }
    
    sum = 0.0;
    for (c = 0; c < num_labels; c++) {
        float tmp;
        if (inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(inst -> scores[c]);
        inst -> scores[c] = tmp;
        sum += inst -> scores[c];
        
    }
    for (c = 0; c < num_labels; c++) {
        inst -> scores[c] /= sum;
    }
}

void TestBigramModelPrep::BackProp()
{
    alpha_old = alpha;
    //    alpha = alpha * ( 1 - eta * lambda );
    real eta_real = eta / alpha;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> BackProp(inst, eta_real);
    }
}

void TestBigramModelPrep::TrainData(string trainfile, string devfile, int type) {
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> emb_map[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> emb_map[word_emb_dim]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> core_tensor[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> core_tensor[word_emb_dim]);
    }
    int count = 0;
    int total = num_inst * iter;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs, type)) {
            ForwardProp();
            BackProp();
            count++;
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        
        if (lr_tensor_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> emb_map[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> emb_map[word_emb_dim]);
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> core_tensor[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> core_tensor[word_emb_dim]);
        }
        EvalData(trainfile, type);
        EvalData(devfile, type);
    }
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> emb_map[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> emb_map[word_emb_dim]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> core_tensor[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> core_tensor[word_emb_dim]);
    }
    
    //ofs.close();
}

void TestBigramModelPrep::EvalData(string trainfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    double max, max_p;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        if (inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        if (inst -> label_id != 0) positive++;
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 0; i < num_labels; i++){
            if (inst -> scores[i] > max) {
                max = inst -> scores[i];
                max_p = i;
            }
        }
        if (max_p != 0) pos_pred ++;
        if (max_p == inst -> label_id) {
            right++;
            if (inst -> label_id != 0) tp++;
        }
    }
    cout << "Acc: " << (float)right / total << endl;
    ifs.close();
}

void TestBigramModelPrep::SetModels() {
    cout << "Adagrad: " << adagrad << endl;
    cout << "update_emb: " << update_emb << endl;
    cout << "update_feat_emb: " << true << endl;
    
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> adagrad = adagrad;
        lr_tensor_list[i] -> update_emb = update_emb;
        lr_tensor_list[i] -> update_word = update_emb;
        lr_tensor_list[i] -> update_fea_emb = true;
        lr_tensor_list[i] -> update_lab_emb = false;
        lr_tensor_list[i] -> debug = false;
    }
}

void TestBigramModelPrep::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    cout << "Max length of sentences: " << max_len << endl;
    
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> PrintModelInfo();
    }
}

