//
//  FullFctModel.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "FullFctModel.h"

void FullFctModel::Init(char* embfile, char* trainfile)
{
    fea_params.head = false;
    fea_params.word_on_path = true;
    fea_params.word_on_path_type = true;
    fea_params.entity_type = true;
    
    fea_params.postag = true;
    fea_params.dep = true;
    fea_params.ner = true;
    fea_params.sst = true;
    fea_params.context = false;
    
    fea_params.tag_fea = true;
    fea_params.dep_fea = false;
//    fea_params.head = true;
//    fea_params.word_on_path = true;
//    fea_params.word_on_path_type = true;
//    fea_params.entity_type = false;
//    
//    fea_params.postag = true;
//    fea_params.dep = true;
//    fea_params.ner = true;
//    fea_params.sst = true;
//    fea_params.context = true;
//    
//    fea_params.tag_fea = true;
//    fea_params.dep_fea = true;
    
    fea_params.hyper_emb = false;
    
    fea_params.dep_path = false;
    fea_params.pos_on_path = false;
    
    fea_params.tri_conv = false;
    fea_params.linear = false;//true;
    
    fea_params.fea_dim = 5;
    
    emb_model = new WordEmbeddingModel(embfile);

    BuildModelsFromData(trainfile);
    num_labels = (int)labeldict.size();
    inst -> scores.resize(num_labels);
    
    alpha = 1.0;
    lambda = 0.0;
}

void FullFctModel::BuildModelsFromData(char* trainfile) {
//    if (type == "SEM_EVAL") {
//        labeldict["Other"] = 0;
//    }
//    else labeldict["NA"] = 0;
    //labeldict["NA"] = 0;
    labeldict["Other"] = 0;
    
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
    
//    fea_model -> InitEmb();
    
    num_labels = (int)labeldict.size();
    labellist.resize(labeldict.size());
    for (feat2int::iterator iter = labeldict.begin(); iter != labeldict.end(); iter++) {
        labellist[iter -> second] = iter -> first;
    }
    ifs.close();
    
//    feat_emb_dim = fea_params.fea_dim;
    feat_emb_dim = fea_model -> dim;
    word_emb_dim = emb_model -> dim;
//    LrFcemModel* pmodel = new LrFcemModel(fea_model, emb_model);
    LrTensorModelBasic* pmodel = new LrTensorModelBasic(fea_model, emb_model, lab_model, fea_params.fea_dim);
    
    pmodel -> num_labels = num_labels;
    pmodel -> max_sent_len = 100;
    pmodel -> InitModel();
    
//    lr_fcem_list.push_back(pmodel);
    lr_tensor_list.push_back(pmodel);
    num_models = (int)lr_fcem_list.size() + (int)lr_tensor_list.size();
}

int FullFctModel::LoadInstanceInit(ifstream &ifs) {
    int id;
    int fea_id;
    int beg1 = 0, end1 = 0, beg2 = 0, end2 = 0;
    char line_buf[5000], line_buf2[1000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    ((LrFcemInstance*) inst) -> Clear();
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) {
            id = (int)labeldict.size();
//            cout << inst->label << endl;
            labeldict[inst -> label] = id;
        }
        
        iter = lab_model -> vocabdict.find(inst -> label);
        if (iter == lab_model -> vocabdict.end()) {
            id = (int)lab_model -> vocabdict.size();
            lab_model -> vocabdict[inst -> label] = id;
        }
        
        iss >> beg1; iss >> end1;
        inst -> ne1_len = end1 + 1 - beg1;
        for (int i = 0; i < inst -> ne1_len; i++) {
            iss >> inst -> ne1_words[i];
            ToLower(inst -> ne1_words[i]);
        }
        iss >> beg2; iss >> end2;
        inst -> ne2_len = end2 + 1 - beg2;
        for (int i = 0; i < inst -> ne2_len; i++) {
            iss >> inst -> ne2_words[i];
            ToLower(inst -> ne2_words[i]);
        }
    }
    {
        ifs.getline(line_buf, 5000, '\n');
        {
            ifs.getline(line_buf2, 1000, '\n');
            istringstream iss3(line_buf2);
            int tmpint;
            string ne_tag;
            iss3 >> tmpint;
            for (int i = 0; i < inst -> ne1_len; i++) {
                iss3 >> inst -> ne1_types[i];
                inst -> ne1_types[i] = ProcSenseTag(inst -> ne1_types[i]);
                if(fea_params.ner) {
                    iss3 >> ne_tag;
                    inst -> ne1_nes[i] = ProcNeTag(ne_tag);
                }
            }
        }
        {
            ifs.getline(line_buf2, 1000, '\n');
            istringstream iss4(line_buf2);
            int tmpint;
            string ne_tag;
            iss4 >> tmpint;
            for (int i = 0; i < inst -> ne2_len; i++) {
                iss4 >> inst -> ne2_types[i];
                inst -> ne2_types[i] = ProcSenseTag(inst -> ne2_types[i]);
                if (fea_params.ner) {
                    iss4 >> ne_tag;
                    inst -> ne2_nes[i] = ProcNeTag(ne_tag);
                }
            }
        }
        inst -> entitytype = inst -> ne1_types[inst -> ne1_len - 1]
        + '\t' 
        + inst -> ne2_types[inst -> ne2_len - 1];
        
        istringstream iss2(line_buf);
        int count = 0;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            if (fea_params.postag) {
                iss2 >> tag; tag = tag.substr(0,2);
            }
            if (fea_params.ner) {
                iss2 >> tag; inst -> word_nes[count] = ProcNeTag(tag);
            }
            if (fea_params.sst) {
                iss2 >> tag; inst -> word_types[count] = ProcSenseTag(tag);
            }
            if (fea_params.dep) {
                iss2 >> inst -> dep_paths[count];
            }
            
            if (fea_params.word_on_path) {
                if (count < max(beg1, beg2) && count > min(end1, end2)) {
                    slot_key = "in_between_only";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                    
                    if (fea_params.postag) {
//                        slot_key = "FCT_in_between_" + tag;
                    }
                    
                    if (fea_params.word_on_path_type && fea_params.entity_type) {
                        if (fea_params.tag_fea && fea_params.sst) {
                            //slot_key = "in_between_self\t" + inst -> word_types[count];
                            //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                        }
                    
                        /*slot_key = "in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        model_id = AddDeepFctModel(slot_key);
                        slot_key = "FCT_in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                        
                        slot_key = "in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        model_id = AddDeepFctModel(slot_key);
                        slot_key = "FCT_in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/

                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "in_between_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            fea_model -> AddFeature(slot_key);
                            slot_key = "in_between_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            fea_model -> AddFeature(slot_key);
                            slot_key = "in_between_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            fea_model -> AddFeature(slot_key);
                        }
                    }
                }
            }
            if (fea_params.dep_fea) {
                if (inst -> dep_paths[count] > 0) {
                    /*slot_key = "on_path_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_bias";
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/
                    slot_key = "on_path_only";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                        
                    if (fea_params.tag_fea && fea_params.sst) {
                        //slot_key = "on_path_self\t" + inst -> word_types[count];
                        //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], true);
                    }
                    
                    /*slot_key = "on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                    
                    slot_key = "on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = AddDeepFctModel(slot_key);
                    slot_key = "FCT_on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);*/
                    if (fea_params.entity_type) {
                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "on_path_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            fea_model -> AddFeature(slot_key);
                            slot_key = "on_path_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            fea_model -> AddFeature(slot_key);
                            slot_key = "on_path_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            fea_model -> AddFeature(slot_key);
                        }
                    }
                }
            }
            
            if (fea_params.context) {
                if(count == beg1 - 1){
                    slot_key = "ne1_left";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
//                    slot_key = "FCT_ne1_left" + inst -> ne1_types[inst -> ne1_len - 1] + "\tne1";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
//                    slot_key = "FCT_ne1_left" + inst -> ne2_types[inst -> ne2_len - 1] + "\tne2";
//                    deep_fct_list[model_id] -> AddRealFCTSlot(slot_key);
                }
                if(count == beg1 - 2){
                    slot_key = "ne1_left2";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                }
                if(count == end1 + 1){
                    slot_key = "ne1_right";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                }
                if(count == end1 + 2){
                    slot_key = "ne1_right2";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                }
                if(count == beg2 - 1){
                    slot_key = "ne2_left";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                }
                if(count == beg2 - 2){
                    slot_key = "ne2_left2";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
                }
                if(count == end2 + 1){
                    slot_key = "ne2_right";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1) {
                        fea_model -> basefeat_list.push_back(fea_id);    
                    }
                }
                if(count == end2 + 2){
                    slot_key = "ne2_right2";
                    fea_id = fea_model -> AddFeature(slot_key);
                    if (fea_id == fea_model -> vocabdict.size() - 1 && fea_model -> basefeat_list[fea_model->basefeat_list.size() - 1] != fea_id) fea_model -> basefeat_list.push_back(fea_id);
                }
            }
            
            count++;
        }
        
        if (fea_params.head) {
            if (fea_params.hyper_emb) {
                slot_key = "ne1_hyper";
                fea_model -> AddFeature(slot_key);
            }
            
            slot_key = "ne1_head_only";
            fea_id = fea_model -> AddFeature(slot_key);
            if (fea_id == fea_model -> vocabdict.size() - 1) fea_model -> basefeat_list.push_back(fea_id);
            
            if (fea_params.entity_type) {
                /*slot_key = "ne1_head\t" + inst -> ne1_types[inst -> ne1_len - 1];
                 AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
                 
                 slot_key = "ne1_head_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
                 AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);*/
                
                slot_key = "ne1_head_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                fea_model -> AddFeature(slot_key);
                
                slot_key = "ne1_head_ner2\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                fea_model -> AddFeature(slot_key);
                
                slot_key = "ne1_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                fea_model -> AddFeature(slot_key);
                //slot_key = "ne1_head_nepair\t" + inst -> entitytype;
                //AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], true);
            }
            
            if (fea_params.hyper_emb) {
                slot_key = "ne2_hyper";
                fea_model -> AddFeature(slot_key);
            }
            
            slot_key = "ne2_head_only";
            fea_id = fea_model -> AddFeature(slot_key);
            if (fea_id == fea_model -> vocabdict.size() - 1 && fea_model -> basefeat_list[fea_model->basefeat_list.size() - 1] != fea_id) {
                fea_model -> basefeat_list.push_back(fea_id);
            }
            
            if (fea_params.entity_type) {
                /*slot_key = "ne2_head\t" + inst -> ne2_types[inst -> ne2_len - 1];
                 AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
                 
                 slot_key = "ne2_head_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
                 AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);*/
                
                slot_key = "ne2_head_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                fea_model -> AddFeature(slot_key);
                
                slot_key = "ne2_head_ner2\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                fea_model -> AddFeature(slot_key);
                
                slot_key = "ne2_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                fea_model -> AddFeature(slot_key);
                //slot_key = "ne2_head_nepair\t" + inst -> entitytype;
                //AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], true);
            }
        }
        
        inst -> len = count + 1;
        if (inst -> len > max_len) {max_len = inst -> len;}
    }
    
    return 1;
}

int FullFctModel::LoadInstance(ifstream &ifs, int type) {
    return LoadInstance(ifs);
}

int FullFctModel::LoadInstance(ifstream &ifs) {
//    int id, model_id;
    word2int::iterator iter2;
    int beg1 = 0, end1 = 0, beg2 = 0, end2 = 0;
    char line_buf[5000], line_buf2[1000];
    vector<int> trigram_id;
    trigram_id.resize(3);
    ifs.getline(line_buf, 1000, '\n');
    if (!strcmp(line_buf, "")) {
        return 0;
    }
    ((LrFcemInstance*) inst) -> Clear();
    {
        istringstream iss(line_buf);
        iss >> inst -> label;
        feat2int::iterator iter = labeldict.find(inst -> label);
        if (iter == labeldict.end()) inst -> label_id = -1;
        else inst -> label_id = iter -> second;
        
        iss >> beg1; iss >> end1;
        inst -> ne1_len = end1 + 1 - beg1;
        for (int i = 0; i < inst -> ne1_len; i++) {
            iss >> inst -> ne1_words[i];
            ToLower(inst -> ne1_words[i]);
        }
        for (int i = 0; i < inst -> ne1_len; i++) {
            iter2 = emb_model -> vocabdict.find(inst -> ne1_words[i]);
            if (iter2 != emb_model -> vocabdict.end()) inst -> ne1_ids[i] = iter2 -> second;
            else inst -> ne1_ids[i] = -1;
        }
        iss >> beg2; iss >> end2;
        inst -> ne2_len = end2 + 1 - beg2;
        for (int i = 0; i < inst -> ne2_len; i++) {
            iss >> inst -> ne2_words[i];
            ToLower(inst -> ne2_words[i]);
        }
        for (int i = 0; i < inst -> ne2_len; i++) {
            iter2 = emb_model -> vocabdict.find(inst -> ne2_words[i]);
            if (iter2 != emb_model -> vocabdict.end()) inst -> ne2_ids[i] = iter2 -> second;
            else inst -> ne2_ids[i] = -1;
        }
    }
    {
        ifs.getline(line_buf, 5000, '\n');
        {
            ifs.getline(line_buf2, 1000, '\n');
            istringstream iss3(line_buf2);
            int tmpint;
            iss3 >> tmpint;
            for (int i = 0; i < inst -> ne1_len; i++) {
                iss3 >> inst -> ne1_types[i];
                inst -> ne1_types[i] = ProcSenseTag(inst -> ne1_types[i]);
                if (fea_params.ner) {
                    iss3 >> inst -> ne1_nes[i]; inst -> ne1_nes[i] = ProcNeTag(inst -> ne1_nes[i]);
                }
                
                iter2 = emb_model -> vocabdict.find("SST:" + inst -> ne1_types[i]);
                if (iter2 != emb_model -> vocabdict.end()) inst -> ne1_type_ids[i] = iter2 -> second;
                else inst -> ne1_type_ids[i] = -1;
            }
        }
        {
            ifs.getline(line_buf2, 1000, '\n');
            istringstream iss4(line_buf2);
            int tmpint;
            iss4 >> tmpint;
            for (int i = 0; i < inst -> ne2_len; i++) {
                iss4 >> inst -> ne2_types[i];
                inst -> ne2_types[i] = ProcSenseTag(inst -> ne2_types[i]);
                if (fea_params.ner) {
                    iss4 >> inst -> ne2_nes[i]; inst -> ne2_nes[i] = ProcNeTag(inst -> ne2_nes[i]);
                }
                
                iter2 = emb_model -> vocabdict.find("SST:" + inst -> ne2_types[i]);
                if (iter2 != emb_model -> vocabdict.end()) inst -> ne2_type_ids[i] = iter2 -> second;
                else inst -> ne2_type_ids[i] = -1;
            }
        }
        inst -> entitytype = inst -> ne1_types[inst -> ne1_len - 1]
        + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
        
        istringstream iss2(line_buf);
        int count = 0;
        string token, tag, slot_key;
        
        while (iss2 >> token) {
            ToLower(token);
            inst -> words[count] = token;
            iter2 = emb_model -> vocabdict.find(token);
            if (iter2 != emb_model -> vocabdict.end()) inst -> word_ids[count] = iter2 -> second;
            else inst -> word_ids[count] = -1;
            
            if (fea_params.postag) {
                iss2 >> tag; inst -> tags[count] = tag.substr(0,2);
            }
            if (fea_params.ner) {
                iss2 >> tag; inst -> word_nes[count] = ProcNeTag(tag);
            }
            if (fea_params.sst) {
                iss2 >> tag; inst -> word_types[count] = ProcSenseTag(tag);
            }
            if (fea_params.dep) {
                iss2 >> inst -> dep_paths[count];
            }
            if (fea_params.word_on_path) {
                if (count < max(beg1, beg2) && count > min(end1, end2)) {
                    /*slot_key = "in_between_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                    model_id = SearchDeepFctSlot(slot_key);
                    if (model_id >= 0) {
                        FctDeepModel* p_model = deep_fct_list[model_id];
                        RealFctPathInstance* p_inst = p_model -> inst;
                        p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                        slot_key = "FCT_bias";
                        id = p_model -> SearchRealFCTSlot(slot_key);
                        p_inst -> PushFctFea(id, p_inst -> count);
                        p_inst -> count++;
                    }*/
                    slot_key = "in_between_only";
                    AddWordFeature(slot_key, count);
                    
                    if (fea_params.entity_type) {
                        if (fea_params.tag_fea && fea_params.sst) {
                            //slot_key = "in_between_self\t" + inst -> word_types[count];
                            //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                        }
                        
                        slot_key = "in_between_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        AddWordFeature(slot_key, count);
                        
                        slot_key = "in_between_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        AddWordFeature(slot_key, count);
                    
                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "in_between_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddWordFeature(slot_key, count);
                            slot_key = "in_between_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            AddWordFeature(slot_key, count);
                            slot_key = "in_between_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddWordFeature(slot_key, count);
                        }
                    }
                }
            }
            if (fea_params.dep_fea) {
                if (inst -> dep_paths[count] > 0) {                    
                    slot_key = "on_path_only";
                    AddWordFeature(slot_key, count);
                        
                    if (fea_params.entity_type) {
                        if (fea_params.tag_fea && fea_params.sst) {
                            //slot_key = "on_path_self\t" + inst -> word_types[count];
                            //AddCoarseFctModel2List(slot_key, inst -> word_ids[count], false);
                        }
                        
                        slot_key = "on_path_ne1\t" + inst -> ne1_types[inst -> ne1_len - 1];
                        AddWordFeature(slot_key, count);
                        
                        slot_key = "on_path_ne2\t" + inst -> ne2_types[inst -> ne2_len - 1];
                        AddWordFeature(slot_key, count);
                        
                        if (fea_params.tag_fea && fea_params.ner) {
                            slot_key = "on_path_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddWordFeature(slot_key, count);
                            slot_key = "on_path_ne1_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                            AddWordFeature(slot_key, count);
                            slot_key = "on_path_ne2_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                            AddWordFeature(slot_key, count);
                        }
                        
                        /*slot_key = "on_path_nepair\t" + inst -> ne1_types[inst -> ne1_len - 1] + '\t' + inst -> ne2_types[inst -> ne2_len - 1];
                         model_id = SearchDeepFctSlot(slot_key);
                         if (model_id >= 0) {
                         FctDeepModel* p_model = deep_fct_list[model_id];
                         RealFctPathInstance* p_inst = p_model -> inst;
                         p_inst -> word_ids[p_inst -> count] = inst -> word_ids[count];
                         slot_key = "FCT_bias";
                         id = p_model -> SearchRealFCTSlot(slot_key);
                         p_inst -> PushFctFea(id, p_inst -> count);
                         p_inst -> count++;
                         }*/
                    }
                }
            }
            
            if (fea_params.context) {
                if(count == beg1 - 1){
                    slot_key = "ne1_left";
                    AddWordFeature(slot_key, count);
                }
                if(count == beg1 - 2){
                    slot_key = "ne1_left2";
                    AddWordFeature(slot_key, count);
                }
                if(count == end1 + 1){
                    slot_key = "ne1_right";
                    AddWordFeature(slot_key, count);
                }
                if(count == end1 + 2){
                    slot_key = "ne1_right2";
                    AddWordFeature(slot_key, count);
                }
                if(count == beg2 - 1){
                    slot_key = "ne2_left";
                    AddWordFeature(slot_key, count);
                }
                if(count == beg2 - 2){
                    slot_key = "ne2_left2";
                    AddWordFeature(slot_key, count);
                }
                if(count == end2 + 1){
                    slot_key = "ne2_right";
                    AddWordFeature(slot_key, count);              
                }
                if(count == end2 + 2){
                    slot_key = "ne2_right2";
                    AddWordFeature(slot_key, count);
                }
            }
            
            count++;
        }
        inst -> len = count + 1;
        
        if (fea_params.head) {
            if (fea_params.hyper_emb){
                slot_key = "ne1_hyper";
            }
            
            slot_key = "ne1_head_only";
            AddWordFeature(slot_key, end1);
            
            if (fea_params.entity_type) {
                slot_key = "ne1_head\t" + inst -> ne1_types[inst -> ne1_len - 1];
                //            AddWordFeature(slot_key, end1);
                
                slot_key = "ne1_head_pair\t" + inst -> ne2_types[inst -> ne2_len - 1];
                //            AddWordFeature(slot_key, end1);
                
                //slot_key = "ne1_head_nepair\t" + inst -> entitytype;
                //AddCoarseFctModel2List(slot_key, inst -> ne1_ids[inst -> ne1_len - 1], false);
                slot_key = "ne1_head_ner\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                AddWordFeature(slot_key, end1);
                
                slot_key = "ne1_head_ner2\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                AddWordFeature(slot_key, end1);
                
                slot_key = "ne1_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                AddWordFeature(slot_key, end1);
            }
            
            if (fea_params.hyper_emb) {
                slot_key = "ne2_hyper";
            }

            slot_key = "ne2_head_only";
            AddWordFeature(slot_key, end2);
            if (fea_params.entity_type) {
                slot_key = "ne2_head\t" + inst -> ne2_types[inst -> ne2_len - 1];
                AddWordFeature(slot_key, end2);
                
                slot_key = "ne2_head_pair\t" + inst -> ne1_types[inst -> ne1_len - 1];
                AddWordFeature(slot_key, end2);
                
                //slot_key = "ne2_head_nepair\t" + inst -> entitytype;
                //AddCoarseFctModel2List(slot_key, inst -> ne2_ids[inst -> ne2_len - 1], false);
                slot_key = "ne2_head_ner\t" + inst -> ne2_nes[inst -> ne2_len - 1];
                AddWordFeature(slot_key, end2);
                
                slot_key = "ne2_head_ner2\t" + inst -> ne1_nes[inst -> ne1_len - 1];
                AddWordFeature(slot_key, end2);
                
                slot_key = "ne2_head_ner_pair\t" + inst -> ne1_nes[inst -> ne1_len - 1] + '\t' + inst -> ne2_nes[inst -> ne2_len - 1];
                AddWordFeature(slot_key, end2);
            }
        }
        if (fea_params.dep_path)
        {
            ifs.getline(line_buf, 5000, '\n');
            istringstream iss3(line_buf);
            int path_len;
            string tmp_str;
            int token_pos;
            iss3 >> path_len;
            string path_len_fea;
            if (fea_params.pos_on_path) {
                ostringstream oss;
                if (path_len < 5) oss << "path_len\t" << path_len;
                else oss << "path_len\t5";
                path_len_fea = oss.str();
            }
            for (int i = 0; i < path_len - 1; i++) {
                iss3 >> tmp_str;
                iss3 >> tmp_str;
                iss3 >> tmp_str;
                iss3 >> token_pos;
                
                if (fea_params.pos_on_path) {
//                    ostringstream oss;
//                    oss << "on_path_pos\t" << inst -> entitytype << "\t" << i;
//                    AddCoarseFctModel2List(oss.str(), inst -> word_ids[token_pos], false);
                    
//                    AddCoarseFctModel2List(path_len_fea, inst -> word_ids[token_pos], false);
//                    slot_key = path_len_fea + "\t" + inst -> entitytype;
//                    AddCoarseFctModel2List(slot_key, inst -> word_ids[token_pos], false);
                }
            }
        }
    }
    return 1;
}

string FullFctModel::ProcSenseTag(string input_type) {
    size_t idx = input_type.find_first_of(".");
    string ret = input_type.substr(idx + 1);
    return ret; 
}

string FullFctModel::ProcNeTag(string input_type) {
    size_t idx = input_type.find_first_of(":");
    string ret = input_type.substr(idx + 1);
    //idx = input_type.find_first_of(":");
    idx = ret.find_first_of(":");
    if (idx != -1) {
        ret = ret.substr(0, idx);
    }
    return ret; 
}

string FullFctModel::ToLower(string& s) {
    for (int i = 0; i < s.length(); i++) {
        if (s[i] >= 'A' && s[i] <= 'Z') s[i] += 32;
    }
    return s;
}

int FullFctModel::AddWordFeature(string feat_key, int pos) {
    int fea_id;
    fea_id = fea_model -> SearchFeature(feat_key);
    if (fea_id == -1) return -1;
    else {
        ((LrFcemInstance*)inst) -> PushFctFea(fea_id, pos);
        return fea_id;
    }
}

void FullFctModel::ForwardProp()
{
    real sum;
    int c;
    for (int i = 0; i < num_labels; i++) {
        inst -> scores[i] = 0.0;
    }
    for (int i = 0; i < lr_fcem_list.size(); i++) {
        lr_fcem_list[i] -> ForwardProp(inst);
    }
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> ForwardProp(inst);
    }
    if (isnan(inst -> scores[0])) {
        for (c = 0; c < num_labels; c++) {
            cout << c << ":" << inst -> scores[c];
        }
        cout << endl;
        exit(-1);
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
//    for (c = 0; c < num_labels; c++) {
//        cout << c << ":" << inst -> scores[c];
//    }
//    cout << endl;
    for (c = 0; c < num_labels; c++) {
        inst -> scores[c] /= sum;
    }
}

void FullFctModel::BackProp()
{
    alpha_old = alpha;
//    alpha = alpha * ( 1 - eta * lambda );
    real eta_real = eta / alpha;
    for (int i = 0; i < lr_fcem_list.size(); i++) {
        lr_fcem_list[i] -> BackProp(inst, eta_real);
    }
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> BackProp(inst, eta_real);
    }
}

void FullFctModel::TrainData(string trainfile, string devfile, int type) {
    if (lr_fcem_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_fcem_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", lr_fcem_list[0] -> label_emb[word_emb_dim]);
    }
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
    }
    ofstream ofs("feat.txt");
    int count = 0;
    int total = num_inst * iter;
//    cout << feat_emb_dim << endl;
//    for (int k = 0; k < 2 * feat_emb_dim; k++) {
//        cout << fea_model->syn0[k] << endl;
//    }
//    cout << endl;
    for (int i = 0; i < iter; i++) {
        cout << "Iter " << i << endl;
        cur_iter = i;
        ifstream ifs(trainfile.c_str());
        //int count = 0;
        while (LoadInstance(ifs, type)) {
            PrintInstance(ofs, (LrFcemInstance*)inst);
            ForwardProp();
            BackProp();
            count++;
            
//            if (count % 100 == 0) {
//                WeightDecay(eta, lambda);
//            }
        }
        if(!adagrad) eta = eta0 * (1 - count / (double)(total + 1));
        if (eta < eta0 * 0.0001) eta = eta0 * 0.0001;
        ifs.close();
        //if(i >= 5) for (int a = 0; a < num_labels * num_slots * layer1_size; a++) label_emb[a] = (1 -lambda_prox) * label_emb[a];
        //if(i >= 5) if(update_emb) for (int a = 0; a < emb_model->vocab_size * layer1_size; a++) emb_model -> syn0[a] = (1 - lambda_prox) * emb_model -> syn0[a];
        if (lr_fcem_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_fcem_list[0] -> label_emb[0]);
            printf("L-emb2: %lf\n", lr_fcem_list[0] -> label_emb[word_emb_dim]);
        }
        if (lr_tensor_list.size() != 0) {
            printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
            printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
        }
        EvalData(trainfile, type);
        EvalData(devfile, type);
    }
    if (lr_fcem_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_fcem_list[0] -> label_emb[0]);
        printf("L-emb2: %lf\n", lr_fcem_list[0] -> label_emb[word_emb_dim]);
    }
    if (lr_tensor_list.size() != 0) {
        printf("L-emb1: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[0]);
        printf("L-emb2: %lf\n", lr_tensor_list[0] -> lab_model -> syn0[fea_params.fea_dim]);
    }
        
    //ofs.close();
}

void FullFctModel::EvalData(string trainfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    double max, max_p;
    real prec, rec;
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
//    cout << right << " " << tp << " "  << positive << " " << pos_pred << endl;
//    cout << "Acc: " << (float)right / total << endl;
    rec = (real)tp / positive;
    prec = (real)tp / pos_pred;
//    cout << "Prec: " << prec << endl;
//    cout << "Rec:" << rec << endl;
    real f1 = 2 * prec * rec / (prec + rec);
//    cout << "F1:" << f1 << endl;
    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
}

void FullFctModel::PrintInstance(ofstream& ofs, LrFcemInstance* p_inst) {
    ofs << p_inst -> label << "\t" << p_inst -> label_id << endl;
    for (int i = 0; i < p_inst -> len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            ofs << p_inst -> word_ids[i] << endl;
            for (int j = 0; j < p_inst -> fct_nums_fea[i]; j++) {
                if (j != 0) {
                    ofs << "\t";
                }
                ofs << p_inst -> fct_fea_ids[i][j];
            }
            ofs << endl;
        }
    }
    ofs << endl;
}

void FullFctModel::EvalData(string trainfile, string outfile, int type) {
    int total = 0;
    int right = 0;
    int positive = 0;
    int tp = 0;
    int pos_pred = 0;
    int id = 8000;
    double max, max_p;
    real prec, rec;
    ifstream ifs(trainfile.c_str());
    ofstream ofs(outfile.c_str());
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
        ofs << (id + total) << "\t" << labellist[max_p] << endl;// << "\t" << inst -> scores[max_p] << endl;
        if (max_p != 0) pos_pred ++;
        if (max_p == inst -> label_id) {
            right++;
            if (inst -> label_id != 0) tp++;
        }
    }
    cout << right << " " << tp << " "  << positive << " " << pos_pred << endl;
    cout << "Acc: " << (float)right / total << endl;
    rec = (real)tp / positive;
    prec = (real)tp / pos_pred;
    cout << "Prec: " << prec << endl;
    cout << "Rec:" << rec << endl;
    real f1 = 2 * prec * rec / (prec + rec);
    cout << "F1:" << f1 << endl;
    cout << std::setprecision(4) << prec * 100 << "\t" << rec * 100 << "\t" << f1 * 100 << endl;
    ifs.close();
    ofs.close();
}

void FullFctModel::SetModels() {
    cout << "Adagrad: " << adagrad << endl;
    cout << "update_emb: " << update_emb << endl;
    cout << "update_feat_emb: " << update_feat_emb << endl;
    
    for (int i = 0; i < lr_fcem_list.size(); i++) {
        lr_fcem_list[i] -> adagrad = adagrad;
        lr_fcem_list[i] -> update_emb = update_emb;
        lr_fcem_list[i] -> update_fea_emb = update_feat_emb;
    }
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> adagrad = adagrad;
        lr_tensor_list[i] -> update_emb = update_emb;
        lr_tensor_list[i] -> update_fea_emb = update_feat_emb;
    }
    
}

void FullFctModel::PrintModelInfo() {
    cout << "Number of Labels: " << num_labels << endl;
    cout << "Number of Instances: " << num_inst << endl;
    cout << "Number of Models: " << num_models << endl;
    if (fea_model != NULL) cout << "Number of Features: " << fea_model -> vocab_size << endl;;
    cout << "Max length of sentences: " << max_len << endl;
    for (int i = 0; i < lr_tensor_list.size(); i++) {
        lr_tensor_list[i] -> PrintModelInfo();
    }
}

void FullFctModel::WeightDecay(real eta_real, real lambda) {
//    for (int i = 0; i < deep_fct_list.size(); i++) {
//        deep_fct_list[i] -> WeightDecay(eta_real, lambda);
//    }
}

