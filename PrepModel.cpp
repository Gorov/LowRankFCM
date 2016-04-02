//
//  PrepModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-18.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "PrepModel.h"

void PrepModel::Init(char* embfile, char* trainfile, Prep_Params* params)
{
    fea_params.position = true;
    fea_params.prep = true;
    fea_params.clus = true;
    fea_params.sum = true;
    
    fea_params.low_rank = true;
    
    fea_params.tri_conv = false;
    fea_params.linear = false;//true;
    
    //    fea_params.fea_dim = 5;
    fea_params.fea_dim = params -> fea_dim;
    
    emb_model = new WordEmbeddingModel(embfile);
    word_emb_dim = emb_model -> dim;
    
    BuildModelsFromData(trainfile);
    num_labels = (int)labeldict.size();
    inst -> scores.resize(num_labels);
    
    alpha = 1.0;
    lambda = 0.0;
}

void PrepModel::BuildModelsFromData(char* trainfile) {
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
    lab_model -> InitEmb(fea_params.fea_dim);
    
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
    
    feat_emb_dim = fea_model -> dim;
    word_emb_dim = emb_model -> dim;

//    num_group = group_dict.size();
    LrTensorModel* pmodel = new LrTensorModel(fea_model, emb_model, lab_model, fea_params.fea_dim);
//    LrSparseTensorModel* pmodel = new LrSparseTensorModel(fea_model, emb_model, lab_model, feat_group, fea_params.fea_dim, fea_params.fea_dim / num_group);
    
    pmodel -> num_labels = num_labels;
    pmodel -> max_sent_len = 100;
//    pmodel -> sparse_map = true;
    pmodel -> update_word = update_emb;
    pmodel -> InitModel();
    
    //    lr_fcem_list.push_back(pmodel);
    lr_tensor_list.push_back(pmodel);
    num_models = (int)lr_tensor_list.size();
}

int PrepModel::LoadInstanceInit(ifstream &ifs) {
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
            
            if (fea_params.sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                fea_id = fea_model -> AddFeature(slot_key);
            }
            
            if (fea_params.position){// && (count == 3 || count == 6)) {
                ostringstream oss;
                oss << "POS_" << count;
                slot_key = oss.str();
                fea_id = fea_model -> AddFeature(slot_key);
                
                if (fea_params.clus) {
                    if (count == 3) {
                        oss.str("");
                        oss << "CLUS6_" << inst -> clus[6];
                        slot_key = oss.str();
                        fea_id = fea_model -> AddFeature(slot_key);
                    } 
                    else if(count == 6) {
                        oss.str("");
                        oss << "CLUS3_" << inst -> clus[3];
                        slot_key = oss.str();
                        fea_id = fea_model -> AddFeature(slot_key);
                    }
                    
                    oss.str("");
                    oss << "POS_" << count << "CLUS3_" << inst -> clus[3] << "CLUS6_" << inst -> clus[6];
                    slot_key = oss.str();
                    fea_id = fea_model -> AddFeature(slot_key);
                    
                }
                
                if (fea_params.prep) {
                    oss.str("");
                    oss << "PREP_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    fea_id = fea_model -> AddFeature(slot_key);
                    
                    oss.str("");
                    oss << "POS_PREP_" << count << "_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    fea_id = fea_model -> AddFeature(slot_key);
                    
                    if (fea_params.clus) {
                        if (count == 3) {
                            oss.str("");
                            oss << "PREP_" << inst -> prep_word << "_CLUS6_" << inst -> clus[6];
                            slot_key = oss.str();
                            fea_id = fea_model -> AddFeature(slot_key);
                        } 
                        else if(count == 6) {
                            oss.str("");
                            oss << "PREP_" << inst -> prep_word << "_CLUS3_" << inst -> clus[3];
                            slot_key = oss.str();
                            fea_id = fea_model -> AddFeature(slot_key);
                        }
                    }
                }
            }
            count++;
        }
        
        inst -> len = count + 1;
        if (inst -> len > max_len) {max_len = inst -> len;}
    }
    
    return 1;
}

int PrepModel::LoadInstance(ifstream& ifs, int type) {
    return LoadInstance(ifs);
}

int PrepModel::LoadInstance(ifstream &ifs) {
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
            
            if (fea_params.sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                AddWordFeature(slot_key, count);
            }
            
            if (fea_params.position){// && (count == 3 || count == 6)) {
                ostringstream oss;
                oss << "POS_" << count; //bug here
                slot_key = oss.str();
                AddWordFeature(slot_key, count);
                
                if (fea_params.clus) {
                    if (count == 3) {
                        oss.str("");
                        oss << "CLUS6_" << inst -> clus[6];
                        slot_key = oss.str();
                        AddWordFeature(slot_key, count);
                    } 
                    else if(count == 6) {
                        oss.str("");
                        oss << "CLUS3_" << inst -> clus[3];
                        slot_key = oss.str();
                        AddWordFeature(slot_key, count);
                    }
                    oss.str("");
                    oss << "POS_" << count << "CLUS3_" << inst -> clus[3] << "CLUS6_" << inst -> clus[6];
                    slot_key = oss.str();
                    AddWordFeature(slot_key, count);
                }
                
                if (fea_params.prep) {
                    oss.str("");
                    oss << "PREP_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    AddWordFeature(slot_key, count);
                    
                    oss.str("");
                    oss << "POS_PREP_" << count << "_" << ((PrepInstance*)inst) -> prep_word;
                    slot_key = oss.str();
                    AddWordFeature(slot_key, count);
                    
                    if (fea_params.clus) {
                        if (count == 3) {
                            oss.str("");
                            oss << "PREP_" << inst -> prep_word << "_CLUS6_" << inst -> clus[6];
                            slot_key = oss.str();
                            AddWordFeature(slot_key, count);
                        } 
                        else if(count == 6) {
                            oss.str("");
                            oss << "PREP_" << inst -> prep_word << "_CLUS3_" << inst -> clus[3];
                            slot_key = oss.str();
                            AddWordFeature(slot_key, count);
                        }
                    }
                }
            }
            count++;
        }
        inst -> len = count + 1;
    }
    return 1;
}

int PrepModel::AddWordFeature(string feat_key, int pos) {
    int fea_id;
    fea_id = fea_model -> SearchFeature(feat_key);
    if (fea_id == -1) return -1;
    else {
        inst -> PushFctFea(fea_id, pos);
        return fea_id;
    }
}

string PrepModel::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

void PrepModel::ForwardProp()
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

void PrepModel::BackProp()
{
    alpha_old = alpha;
    //    alpha = alpha * ( 1 - eta * lambda );
    real eta_real = eta / alpha;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> BackProp(inst, eta_real);
    }
}

void PrepModel::TrainData(string trainfile, string devfile, int type) {
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[word_emb_dim]);
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
            
            if (count % 100 == 0) {
                WeightDecay(eta, lambda);
            }
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        //if(i >= 5) for (int a = 0; a < num_labels * num_slots * layer1_size; a++) label_emb[a] = (1 -lambda_prox) * label_emb[a];
        //if(i >= 5) if(update_emb) for (int a = 0; a < emb_model->vocab_size * layer1_size; a++) emb_model -> syn0[a] = (1 - lambda_prox) * emb_model -> syn0[a];
        if (lr_tensor_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[word_emb_dim]);
        }
        EvalData(trainfile, type);
        EvalData(devfile, type);
    }
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[word_emb_dim]);
    }
    
    //ofs.close();
}

void PrepModel::EvalData(string trainfile, int type) {
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

void PrepModel::SetModels() {
    cout << "Adagrad: " << adagrad << endl;
    cout << "update_emb: " << update_emb << endl;
    cout << "update_feat_emb: " << true << endl;
    
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> adagrad = adagrad;
        lr_tensor_list[i] -> update_emb = update_emb;
        lr_tensor_list[i] -> update_word = update_emb;
        lr_tensor_list[i] -> update_fea_emb = true;
        lr_tensor_list[i] -> debug = false;
    }
}

void PrepModel::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    //    cout << "Number of FCT Slots: " << deep_fct_list[0] -> fct_slotdict.size() << endl;
    cout << "Max length of sentences: " << max_len << endl;
    
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> PrintModelInfo();
    }
}

void PrepModel::WeightDecay(real eta_real, real lambda) {
}
