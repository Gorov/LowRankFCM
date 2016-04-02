//
//  RunBigramRanking.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-9.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "RunBigramRanking.h"

void RunBigramRanking::Init(char* embfile, char* trainfile, PPA_Params* params)
{
    fea_params.sum = true;
    
    fea_params.position = true;
    fea_params.prep = true;
    fea_params.postag = false;
    
    fea_params.clus = params -> clus;
    
    fea_params.context = true;
    
    fea_params.nextpos = params -> nextpos;
    fea_params.verbnet = params -> verbnet;
    fea_params.wordnet = params -> wordnet;
    
    //    fea_params.fea_dim = 5;
    fea_params.fea_dim = params -> fea_dim;
    
    fea_params.rank1 = params -> rank1;
    fea_params.rank2 = params -> rank2;
    fea_params.rank3 = params -> rank3 = 1;
    
    emb_model = new WordEmbeddingModel(embfile);
    
    BuildModelsFromData(trainfile);
    num_labels = (int)labeldict.size();
    inst -> scores.resize(max_len);
    
    alpha = 1.0;
    lambda = 0.0;
}

void RunBigramRanking::BuildModelsFromData(char* trainfile) {
    
    fea_model = new FeatureEmbeddingModel();
    lab_model = new LabelEmbeddingModel();
    
    feat_group = new int2int();
    num_group = 0;
    group_dict.clear();
    
    ifstream ifs(trainfile);
    num_inst = 0;
    while (LoadInstanceInit(ifs)) {
        num_inst++;
    }
    
    num_feats = (int)fea_model -> vocabdict.size();
    //    fea_model -> InitEmb(fea_params.fea_dim);
    fea_model -> InitEmb();
    fea_params.rank2 = (int)fea_model -> vocabdict.size();
    
    num_labels = (int)lab_model -> vocabdict.size();
    //    lab_model -> InitEmb(fea_params.fea_dim);
    lab_model -> InitEmb();
    
    num_labels = (int)labeldict.size();
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
    
    feat_emb_dim = fea_model -> dim;
    word_emb_dim = emb_model -> dim;
    LrRankingModel* pmodel = new LrRankingModel(fea_model, emb_model, lab_model, fea_params.rank1, fea_params.rank2);
    
    pmodel -> max_list_len = max_len;
    pmodel -> max_sent_len = 3;
    pmodel -> num_labels = num_labels;
    pmodel -> num_fea = num_feats;
    pmodel -> update_word = update_emb;
    pmodel -> InitModel();
    
    lr_tensor_list.push_back(pmodel);
    num_models = (int)lr_tensor_list.size();
}

int RunBigramRanking::LoadInstanceInit(ifstream &ifs) {
    int fea_id;
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
    
    PPAInstance* p_inst = (PPAInstance*) inst;
    p_inst -> Clear();
    {
        istringstream iss(line_buf);
        iss >> token;
        ToLower(token);
        inst -> words[0] = token;
        
        if (fea_params.clus) {
            iss >> p_inst -> clus[0];
        }
        
        iss >> p_inst -> prep_word;
        
        iss >> p_inst -> list_len;
    }
    for (int i = 0; i < p_inst -> list_len; i++)
    {
        ifs.getline(line_buf, 5000, '\n');
        istringstream iss2(line_buf);
        string token, slot_key;
        string headpos;
        string nextpos;
        int verbnet_len;
        int label;
        
        iss2 >> label;
        if (label == 1) inst -> label_id = p_inst -> label_pos = i;
        
        p_inst -> word_pairs[i][0] = inst -> words[0];
        iss2 >> token;
        ToLower(token);
        p_inst -> word_pairs[i][1] = token;
        
        if (fea_params.clus) {
            iss2 >> clus;
        }
        
        if (fea_params.nextpos) {
            iss2 >> nextpos;
        }
        
        if (fea_params.verbnet) {
            ifs.getline(line_buf2, 2000, '\n');
            istringstream iss3(line_buf2);
            iss3 >> verbnet_len;
            if (verbnet_len == -1) headpos = "N";
            else {
                headpos = "V";
                for (int j = 0; j < verbnet_len; j++) {
                    iss3 >> preps[j];
                }
            }
        }
        
        if (fea_params.sum) {
            ostringstream oss;
            oss << "BIAS";
            slot_key = oss.str();
            fea_id = fea_model -> AddFeature(slot_key);
        }
        
        if (fea_params.clus) {
            ostringstream oss;
            oss << "CHILD_CLUS_" << p_inst -> clus[0];
            slot_key = oss.str();
            fea_id = fea_model -> AddFeature(slot_key);
        }
        
        if (fea_params.nextpos) {
            ostringstream oss;
            oss << "NEXT_POSTAG_" << nextpos;
            slot_key = oss.str();
            fea_id = fea_model -> AddFeature(slot_key);
        }
        
        if (fea_params.position) {
            if (i == p_inst -> list_len - 1) {
                slot_key = "CLOSEST";
                fea_id = fea_model -> AddFeature(slot_key);
            }
            ostringstream oss;
            oss << "DIST_" << (p_inst -> list_len - 1 - i);
            slot_key = oss.str();
            fea_id = fea_model -> AddFeature(slot_key);
        }
        
        if (fea_params.prep) {
            ostringstream oss;
            oss.str("");
            oss << "PREP_" << ((PPAInstance*)inst) -> prep_word;
            slot_key = oss.str();
            fea_id = fea_model -> AddFeature(slot_key);
        }
        
        if (fea_params.verbnet) {
            ostringstream oss;
            oss << "HEADTAG" << headpos;
            slot_key = oss.str();
            fea_id = fea_model -> AddFeature(slot_key);
            int match = 0;
            for (int k = 0; k < verbnet_len; k++) {
                if (strcmp(preps[k].c_str(), p_inst -> prep_word.c_str()) == 0) {
                    match = 1;
                }
                oss.str("");
                oss << "HEAD_VERBPREP_" << preps[k];
                slot_key = oss.str();
                fea_id = fea_model -> AddFeature(slot_key);
            }
            if (match == 1) {
                oss.str("");
                oss << "HEAD_VERBPREP_MATCHED";
                slot_key = oss.str();
                fea_id = fea_model -> AddFeature(slot_key);
            }
        }
        
        inst -> len = p_inst -> list_len;
        if (inst -> len > max_len) {max_len = inst -> len;}
    }
    
    return 1;
}

int RunBigramRanking::LoadInstance(ifstream &ifs) {
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
    
    PPAInstance* p_inst = (PPAInstance*) inst;
    p_inst -> Clear();
    {
        istringstream iss(line_buf);
        iss >> token;
        ToLower(token);
        inst -> words[0] = token;
        
        iter2 = emb_model -> vocabdict.find(inst -> words[0]);
        if (iter2 != emb_model -> vocabdict.end()) inst -> word_ids[0] = iter2 -> second;
        else inst -> word_ids[0] = -1;
        
        if (fea_params.clus) {
            iss >> p_inst -> clus[0];
        }
        
        iss >> p_inst -> prep_word;
        
        iss >> p_inst -> list_len;
    }
    for (int i = 0; i < p_inst -> list_len; i++)
    {
        ifs.getline(line_buf, 5000, '\n');
        istringstream iss2(line_buf);
        string token, slot_key;
        string headpos;
        string nextpos;
        int verbnet_len;
        int label;
        
        iss2 >> label;
        if (label == 1) inst -> label_id = p_inst -> label_pos = i;
        
        p_inst -> word_pairs[i][0] = inst -> words[0];
        p_inst -> id_pairs[i][0] = inst -> word_ids[0];
        
        iss2 >> token;
        ToLower(token);
        p_inst -> word_pairs[i][1] = token;
        iter2 = emb_model -> vocabdict.find(token);
        if (iter2 != emb_model -> vocabdict.end()) p_inst -> id_pairs[i][1] = iter2 -> second;
        else p_inst -> id_pairs[i][1] = -1;
        
        if (fea_params.clus) {
            iss2 >> clus;
        }
        
        if (fea_params.nextpos) {
            iss2 >> nextpos;
        }
        
        if (fea_params.verbnet) {
            ifs.getline(line_buf2, 2000, '\n');
            istringstream iss3(line_buf2);
            iss3 >> verbnet_len;
            if (verbnet_len == -1) headpos = "N";
            else {
                headpos = "V";
                for (int j = 0; j < verbnet_len; j++) {
                    iss3 >> preps[j];
                }
            }
        }
        
        if (fea_params.sum) {
            slot_key = "BIAS";
            AddWordFeature(slot_key, i, 0);
        }
        
        if (fea_params.clus) {
            ostringstream oss;
            oss << "CHILD_CLUS_" << p_inst -> clus[0];
            slot_key = oss.str();
            AddWordFeature(slot_key, i, 0);
        }
        
        if (fea_params.nextpos) {
            ostringstream oss;
            oss << "NEXT_POSTAG_" << nextpos;
            slot_key = oss.str();
            AddWordFeature(slot_key, i, 0);
        }
        
        if (fea_params.position) {
            if (i == p_inst -> list_len - 1) {
                slot_key = "CLOSEST";
                AddWordFeature(slot_key, i, 0);
            }
            ostringstream oss;
            oss << "DIST_" << (p_inst -> list_len - 1 - i);
            slot_key = oss.str();
            AddWordFeature(slot_key, i, 0);
        }
        
        if (fea_params.prep) {
            ostringstream oss;
            oss.str("");
            oss << "PREP_" << ((PPAInstance*)inst) -> prep_word;
            slot_key = oss.str();
            AddWordFeature(slot_key, i, 0);
        }
        
        if (fea_params.verbnet) {
            ostringstream oss;
            oss << "HEADTAG" << headpos;
            slot_key = oss.str();
            AddWordFeature(slot_key, i, 0);
            int match = 0;
            for (int k = 0; k < verbnet_len; k++) {
                if (strcmp(preps[k].c_str(), p_inst -> prep_word.c_str()) == 0) {
                    match = 1;
                }
                oss.str("");
                oss << "HEAD_VERBPREP_" << preps[k];
                slot_key = oss.str();
                AddWordFeature(slot_key, i, 0);
            }
            if (match == 1) {
                oss.str("");
                oss << "HEAD_VERBPREP_MATCHED";
                slot_key = oss.str();
                AddWordFeature(slot_key, i, 0);
            }
        }
        
        inst -> len = p_inst -> list_len;
    }
    
    return 1;
}


string RunBigramRanking::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

int RunBigramRanking::AddWordFeature(string feat_key, int pair_id, int pos) {
    int fea_id;
    fea_id = fea_model -> SearchFeature(feat_key);
    if (fea_id == -1) return -1;
    else {
        ((PPAInstance*)inst) -> PushFctFea(fea_id, pair_id, pos);
        return fea_id;
    }
}

void RunBigramRanking::ForwardProp()
{
    real sum;
    int c;
    for (int i = 0; i < num_labels; i++) {
        inst -> scores[i] = 0.0;
    }
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> ForwardProp(inst);
    }
    
    if (debug) {
        for (c = 0; c < num_labels; c++) {
            cout << c << ":" << inst -> scores[c];
        }
        cout << endl;
    }
    
    PPAInstance* p_inst = (PPAInstance*) inst;
    sum = 0.0;
    for (c = 0; c < p_inst -> list_len; c++) {
        float tmp;
        if (inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(inst -> scores[c]);
        inst -> scores[c] = tmp;
        sum += inst -> scores[c];
        
    }
    for (c = 0; c < p_inst -> list_len; c++) {
        inst -> scores[c] /= sum;
    }
}

void RunBigramRanking::BackProp()
{
    alpha_old = alpha;
    real eta_real = eta / alpha;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> BackProp(inst, eta_real);
    }
}

void RunBigramRanking::TrainData(string trainfile, string devfile) {
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> vec_emb_map[0][0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> vec_emb_map[0][fea_params.rank1]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[fea_params.rank2]);
    }
    //    ofstream ofs("feat.txt");
    int count = 0;
    int total = num_inst * iter;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs)) {
            ForwardProp();
            BackProp();
            count++;
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        
        if (lr_tensor_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> vec_emb_map[0][0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> vec_emb_map[0][fea_params.rank1]);
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[fea_params.rank2]);
        }
        EvalData(trainfile);
        EvalData(devfile);
    }
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> vec_emb_map[0][0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> vec_emb_map[0][fea_params.rank1]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[fea_params.rank2]);
    }
    
    //ofs.close();
}

void RunBigramRanking::EvalClosest(string trainfile) {
    int total = 0;
    int right = 0;
    real acc = 0.0;
    ifstream ifs(trainfile.c_str());
    
    PPAInstance* p_inst = (PPAInstance*) inst;
    while (LoadInstance(ifs)) {
        if (inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        
        if (p_inst -> list_len - 1 == inst -> label_id) {
            right++;
        }
    }
    acc = real(right) / total;
    
    cout << std::setprecision(4) << right << "\t" << total << "\t" << acc * 100 << endl;
    ifs.close();
}

void RunBigramRanking::EvalData(string trainfile) {
    int total = 0;
    int right = 0;
    real acc = 0.0;
    double max, max_p;
    
    PPAInstance* p_inst = (PPAInstance*) inst;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs)) {
        if (inst -> label_id == -1) {
            continue;
        }
        //continue;
        total++;
        
        ForwardProp();
        max = -1;
        max_p = -1;
        for (int i = 0; i < p_inst -> list_len; i++){
            if (inst -> scores[i] > max) {
                max = inst -> scores[i];
                max_p = i;
            }
        }
        if (max_p == inst -> label_id) {
            right++;
        }
    }
    acc = real(right) / total;
    
    cout << std::setprecision(4) << right << "\t" << total << "\t" << acc * 100 << endl;
    ifs.close();
}

void RunBigramRanking::SetModels() {
    cout << "Adagrad: " << adagrad << endl;
    cout << "update_emb: " << update_emb << endl;
    cout << "update_feat_emb: " << update_feat_emb << endl;
    cout << "update_lab_emb: " << update_lab_emb << endl;
    
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> adagrad = adagrad;
        lr_tensor_list[i] -> update_emb = update_emb;
        lr_tensor_list[i] -> update_word = update_emb;
        lr_tensor_list[i] -> update_fea_emb = update_feat_emb;
        lr_tensor_list[i] -> update_lab_emb = update_lab_emb;
        lr_tensor_list[i] -> debug = debug;
    }
}

void RunBigramRanking::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    if (fea_model != NULL) cout << "Number of Features: " << fea_model -> vocab_size << endl;;
    cout << "Max length of sentences: " << max_len << endl;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> PrintModelInfo();
    }
}
