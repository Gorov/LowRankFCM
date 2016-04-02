//
//  TestTuckerMtl.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-26.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "TestTuckerMtl.h"

void TestTuckerMtl::Init(char* embfile, char* trainfile, FeaParams* params)
{
    fea_params.head = true;
    fea_params.word_on_path = true;
    fea_params.word_on_path_type = true;
    fea_params.entity_type = true;
    
    fea_params.postag = true;
    fea_params.dep = true;
    fea_params.ner = true;
    fea_params.sst = true;
    fea_params.context = true;
    
    fea_params.tag_fea = true;
    fea_params.dep_fea = true;
    
    fea_params.hyper_emb = false;
    
    fea_params.dep_path = false;
    fea_params.pos_on_path = false;
    
    fea_params.tri_conv = false;
    fea_params.linear = false;//true;
    
    //    fea_params.fea_dim = 5;
    fea_params.fea_dim = params -> fea_dim;
    
    fea_params.rank1 = params -> rank1;
    fea_params.rank2 = params -> rank2;
    fea_params.rank3 = params -> rank3;
    
    emb_model = new WordEmbeddingModel(embfile);
    
    BuildModelsFromData(trainfile);
    num_labels[0] = (int)labeldicts[0].size();
    num_labels[1] = (int)labeldicts[1].size();
    inst -> scores.resize(num_labels[0] + num_labels[1]);
    
    alpha = 1.0;
    lambda = 0.0;
}

void TestTuckerMtl::BuildModelsFromData(char* trainfile) {
    labeldict["NA"] = 0;
    
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
    fea_model -> InitEmb(fea_params.rank2);
//    fea_model -> InitEmb();
    
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    
    //re-arranging labels
    labeldict.clear();
    labellists[0].clear(); labellists[1].clear();
    labeldicts[0].clear(); labeldicts[1].clear();
    labeldicts[0]["T1:NA"] = (int)labellists[0].size();
    labellists[0].push_back("T1:NA");
    labeldicts[1]["T2:NA"] = (int)labellists[1].size();
    labellists[1].push_back("T2:NA");
    
    lab_model -> vocabdict.clear();
    lab_model -> vocablist.clear();
    lab_model -> vocabdict["T1:NA"] = (int)lab_model -> vocablist.size();
    lab_model -> vocablist.push_back("T1:NA");
    
    for (vector<string>::iterator iter = labellist.begin(); iter != labellist.end(); iter++) {
        if (strcmp("T1:NA", iter -> c_str()) == 0 || strcmp("T2:NA", iter -> c_str()) == 0) {
            continue;
        }
        if(iter -> find("T1:") == 0) {
            labeldicts[0][*iter] = (int)labellists[0].size();
            lab_model -> vocabdict[*iter] = (int)labellists[0].size();
            labellists[0].push_back(*iter);
//            cout << *iter << ":" << lab_model -> vocabdict[*iter] << endl;
            lab_model -> vocablist.push_back(*iter);
        }
        else if (iter -> find("T2:") == 0) {
            labeldicts[1][*iter] = (int)labellists[1].size();
            lab_model -> vocabdict[*iter] = (int)labellists[1].size();
            labellists[1].push_back(*iter);
//            cout << *iter << ":" << lab_model -> vocabdict[*iter] << endl;
        }
    }
    
    int tmp_start = (int)labellists[0].size();
    int tmp_id;
    for (vector<string>::iterator iter = labellists[1].begin(); iter != labellists[1].end(); iter++) {
        lab_model -> vocablist.push_back(*iter);
        tmp_id = labeldicts[1][*iter];
        labeldicts[1][*iter] = tmp_id + tmp_start;
        lab_model -> vocabdict[*iter] = tmp_id + tmp_start;
    }
    
    for (feat2int::iterator iter = lab_model -> vocabdict.begin(); iter != lab_model -> vocabdict.end(); iter++) {
        cout << iter -> first << ":" << iter -> second << endl;
    }
    
    num_labels[0] = (int)labeldicts[0].size();
    num_labels[1] = (int)labeldicts[1].size();
    lab_model -> vocab_size = lab_model -> vocabdict.size();
    
    //    lab_model -> InitEmb(fea_params.fea_dim);
    lab_model -> InitEmb(fea_params.rank3);
    
    ifs.close();
    
    //    feat_emb_dim = fea_params.fea_dim;
    feat_emb_dim = fea_model -> dim;
    word_emb_dim = emb_model -> dim;
    
    LrTuckerMtl* pmodel = new LrTuckerMtl(fea_model, emb_model, lab_model, fea_params.rank1, fea_params.rank2, fea_params.rank3);
    int start = 0;
    for (int i = 0; i < 2; i++) {
        pmodel -> start[i] = start;
        pmodel -> length[i] = num_labels[i];
        start += num_labels[i];
    }
    
    pmodel -> num_labels = num_labels[0] + num_labels[1];
    pmodel -> max_sent_len = 100;
    pmodel -> update_word = update_emb;
    pmodel -> InitModel();
    
    lr_tensor_list.push_back(pmodel);
    num_models = (int)lr_tensor_list.size();
}

void TestTuckerMtl::ForwardProp(int task_id)
{
    real sum;
    int c;
    int start, end;
    start = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id];
    end = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id] + ((LrTuckerMtl*)lr_tensor_list[0]) -> length[task_id];
    for (int i = start; i < end; i++) {
        inst -> scores[i] = 0.0;
    }
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        ((LrTuckerMtl*)lr_tensor_list[i]) -> ForwardProp(inst, task_id);
    }
    
    if (debug) {
        for (c = start; c < end; c++) {
            cout << c << ":" << inst -> scores[c];
        }
        cout << endl;
    }
    
    sum = 0.0;
    for (c = start; c < end; c++) {
        float tmp;
        if (inst -> scores[c] > MAX_EXP) tmp = exp(MAX_EXP);
        else if (inst -> scores[c] < MIN_EXP) tmp = exp(MIN_EXP);
        else tmp = exp(inst -> scores[c]);
        inst -> scores[c] = tmp;
        sum += inst -> scores[c];
        
    }
    for (c = start; c < end; c++) {
        inst -> scores[c] /= sum;
    }
}

void TestTuckerMtl::BackProp(int task_id)
{
    int start, end;
    start = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id];
    end = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id] + ((LrTuckerMtl*)lr_tensor_list[0]) -> length[task_id];
    alpha_old = alpha;
    real eta_real = eta / alpha;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        ((LrTuckerMtl*)lr_tensor_list[i]) -> BackProp(inst, eta_real, task_id);
    }
}

void TestTuckerMtl::TrainData(string trainfile, string devfile, int type) {
    feat2int::iterator lab_iter;
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> emb_map[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> emb_map[fea_params.rank1]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.rank3]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[fea_params.rank2]);
    }
    //    ofstream ofs("feat.txt");
    int count = 0;
    int total = num_inst * iter;
    int task_id;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs, type)) {
            if (strcmp("T1:", inst -> label.substr(0,3).c_str()) == 0) {
                task_id = 0;
            }
            else task_id = 1;
            lab_iter = labeldicts[task_id].find(inst -> label);
            if (lab_iter == labeldicts[task_id].end()) inst -> label_id = -1;
            else inst -> label_id = lab_iter -> second;
            
            ForwardProp(task_id);
            BackProp(task_id);
            count++;
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        
        if (lr_tensor_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> emb_map[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> emb_map[fea_params.rank1]);
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.rank3]);
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[fea_params.rank2]);
        }
        EvalData(trainfile, type);
        EvalData(devfile, type);
    }
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> emb_map[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> emb_map[fea_params.rank1]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.rank3]);
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> fea_model -> syn0[fea_params.rank2]);
    }
    
    //ofs.close();
}

void TestTuckerMtl::EvalData(string trainfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    double max, max_p;
    real prec, rec;
    int task_id;
    feat2int::iterator lab_iter;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        if (strcmp("T1:", inst -> label.substr(0,3).c_str()) == 0) {
            task_id = 0;
        }
        else task_id = 1;
        lab_iter = labeldicts[task_id].find(inst -> label);
        if (lab_iter == labeldicts[task_id].end()) inst -> label_id = -1;
        else inst -> label_id = lab_iter -> second;
        
        if (inst -> label_id == -1) {
            continue;
        }
        total++;
        
        if (inst -> label_id != 0) positive++;
        ForwardProp(task_id);
        max = -1;
        max_p = -1;
        int start = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id];
        int end = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id] + ((LrTuckerMtl*)lr_tensor_list[0]) -> length[task_id];
        for (int i = start; i < end; i++){
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
    rec = (real)tp / positive;
    prec = (real)tp / pos_pred;
    
    real f1 = 2 * prec * rec / (prec + rec);
    //    cout << "F1:" << f1 << endl;
    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
}

void TestTuckerMtl::EvalData(string trainfile, int type, int eval_id) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    double max, max_p;
    real prec, rec;
    int task_id;
    feat2int::iterator lab_iter;
    ifstream ifs(trainfile.c_str());
    while (LoadInstance(ifs, type)) {
        if (strcmp("T1:", inst -> label.substr(0,3).c_str()) == 0) {
            task_id = 0;
        }
        else task_id = 1;
        if (task_id != eval_id) {
            continue;
        }
        lab_iter = labeldicts[task_id].find(inst -> label);
        if (lab_iter == labeldicts[task_id].end()) inst -> label_id = -1;
        else inst -> label_id = lab_iter -> second;
        
        if (inst -> label_id == -1) {
            continue;
        }
        total++;
        
        if (inst -> label_id != ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id]) positive++;
        ForwardProp(task_id);
        max = -1;
        max_p = -1;
        int start = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id];
        int end = ((LrTuckerMtl*)lr_tensor_list[0]) -> start[task_id] + ((LrTuckerMtl*)lr_tensor_list[0]) -> length[task_id];
        for (int i = start; i < end; i++){
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
    rec = (real)tp / positive;
    prec = (real)tp / pos_pred;
    
    real f1 = 2 * prec * rec / (prec + rec);
    //    cout << "F1:" << f1 << endl;
    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
}

void TestTuckerMtl::SetModels() {
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

void TestTuckerMtl::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels[0] << " " << num_labels[1] << " "
    << num_labels[0] + num_labels[1] << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    if (fea_model != NULL) cout << "Number of Features: " << fea_model -> vocab_size << endl;;
    cout << "Max length of sentences: " << max_len << endl;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> PrintModelInfo();
    }
}
