//
//  FeatureFactory.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-11.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include "FeatureFactory.h"

void PPA_FeatureFactory::ExtractFeatures(PPAInstance *p_inst, bool add) {
    p_inst -> Clear();
    int fea_id;
    for (int i = 0; i < p_inst -> list_len; i++)
    {
        string slot_key;
        
        for (int j = 0; j < 2; j++) {  
            if (fea_params -> sum) {
                ostringstream oss;
                oss << "BIAS";
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            
            if (fea_params -> clus) {
                ostringstream oss;
                oss << "HEAD_CLUS_" << p_inst -> child_clus;
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
                
                oss.str("");
                oss << "CHILD_CLUS_" << p_inst -> clus[0];
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
                
                oss.str("");
                oss << "PAIR_CLUS_" << p_inst -> clus[0] << "_" << p_inst -> child_clus;
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            
            if (fea_params -> nextpos) {
                ostringstream oss;
                oss << "NEXT_POSTAG_" << p_inst -> nextpos[i] << "_POSITION_" << j;
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            
            if (j == 0) {
                ostringstream oss;
                oss << "CHILD";
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            else if (j == 1) {
                ostringstream oss;
                oss << "HEAD";
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            
            if (fea_params -> position) {
                if (i == p_inst -> list_len - 1) {
                    slot_key = "CLOSEST";
                    if (add) {
                        fea_id = fea_model -> AddFeature(slot_key);
                    }
                    else {
                        AddWordFeature(p_inst, slot_key, i, j);
                    }
                }
                ostringstream oss;
                oss << "DIST_" << (p_inst -> list_len - 1 - i) << "_POSITION_" << j;
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            
            if (fea_params -> prep) {
                ostringstream oss;
                oss.str("");
                oss << "PREP_" << p_inst -> prep_word << "_POSITION_" << j;
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
            }
            
            if (fea_params -> verbnet) {
                ostringstream oss;
                oss << "HEADTAG" << p_inst -> headpos[i] << "_POSITION_" << j;
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, j);
                }
                
                //                if (j == 1) {
                int match = 0;
                for (int k = 0; k < p_inst -> verbnet_len[i]; k++) {
                    if (strcmp(p_inst -> preps[i][k].c_str(), p_inst -> prep_word.c_str()) == 0) {
                        match = 1;
                    }
                    oss.str("");
                    oss << "HEAD_VERBPREP_" << p_inst -> preps[i][k] << "_POSITION_" << j;
                    slot_key = oss.str();
                    if (add) {
                        fea_id = fea_model -> AddFeature(slot_key);
                    }
                    else {
                        AddWordFeature(p_inst, slot_key, i, j);
                    }
                }
                if (match == 1) {
                    oss.str("");
                    oss << "HEAD_VERBPREP_MATCHED" << "_POSITION_" << j;
                    slot_key = oss.str();
                    if (add) {
                        fea_id = fea_model -> AddFeature(slot_key);
                    }
                    else {
                        AddWordFeature(p_inst, slot_key, i, j);
                    }
                }
                //                }
            }
//            if (fea_params -> wordnet) {
//                ostringstream oss;
//                for (int k = 0; k < wordnet_len; k++) {
//                    oss.str("");
//                    oss << "CHILD_SENSE_" << senses[k] << "_POSITION_" << j;
//                    slot_key = oss.str();
//                    fea_id = fea_model -> AddFeature(slot_key);
//                }
//                for (int k = 0; k < head_wordnet_len; k++) {
//                    oss.str("");
//                    oss << "HEAD_SENSE_" << head_senses[k] << "_POSITION_" << j;
//                    slot_key = oss.str();
//                    fea_id = fea_model -> AddFeature(slot_key);
//                }
//            }
        }
    }  
}

void PPA_FeatureFactory::ExtractBigramFeatures(PPAInstance *p_inst, bool add) {
    p_inst -> Clear();
    int fea_id;
    for (int i = 0; i < p_inst -> list_len; i++)
    {
        string slot_key;
        
        if (fea_params -> sum) {
            ostringstream oss;
            oss << "BIAS";
            slot_key = oss.str();
            if (add) {
                fea_id = fea_model -> AddFeature(slot_key);
            }
            else {
                AddWordFeature(p_inst, slot_key, i, 0);
            }
        }
        
        if (fea_params -> clus) {
            ostringstream oss;
            oss << "CHILD_CLUS_" << p_inst -> child_clus;
            slot_key = oss.str();
            if (add) {
                fea_id = fea_model -> AddFeature(slot_key);
            }
            else {
                AddWordFeature(p_inst, slot_key, i, 0);
            }
        }
        
        if (fea_params -> nextpos) {
            ostringstream oss;
            oss << "NEXT_POSTAG_" << p_inst -> nextpos[i];
            slot_key = oss.str();
            if (add) {
                fea_id = fea_model -> AddFeature(slot_key);
            }
            else {
                AddWordFeature(p_inst, slot_key, i, 0);
            }
        }
        
        if (fea_params -> position) {
            if (i == p_inst -> list_len - 1) {
                slot_key = "CLOSEST";
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, 0);
                }
            }
            ostringstream oss;
            oss << "DIST_" << (p_inst -> list_len - 1 - i);
            slot_key = oss.str();
            if (add) {
                fea_id = fea_model -> AddFeature(slot_key);
            }
            else {
                AddWordFeature(p_inst, slot_key, i, 0);
            }
        }
        
        if (fea_params -> prep) {
            ostringstream oss;
            oss.str("");
            oss << "PREP_" << p_inst -> prep_word;
            slot_key = oss.str();
            if (add) {
                fea_id = fea_model -> AddFeature(slot_key);
            }
            else {
                AddWordFeature(p_inst, slot_key, i, 0);
            }
        }
        
        if (fea_params -> verbnet) {
            ostringstream oss;
            oss << "HEADTAG" << p_inst -> headpos[i];
            slot_key = oss.str();
            if (add) {
                fea_id = fea_model -> AddFeature(slot_key);
            }
            else {
                AddWordFeature(p_inst, slot_key, i, 0);
            }
            int match = 0;
            for (int k = 0; k < p_inst -> verbnet_len[i]; k++) {
                if (strcmp(p_inst -> preps[i][k].c_str(), p_inst -> prep_word.c_str()) == 0) {
                    match = 1;
                }
                oss.str("");
                oss << "HEAD_VERBPREP_" << p_inst -> preps[i][k];
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, 0);
                }
            }
            if (match == 1) {
                oss.str("");
                oss << "HEAD_VERBPREP_MATCHED";
                slot_key = oss.str();
                if (add) {
                    fea_id = fea_model -> AddFeature(slot_key);
                }
                else {
                    AddWordFeature(p_inst, slot_key, i, 0);
                }
            }
        }
    }
}

int PPA_FeatureFactory::AddWordFeature(PPAInstance* p_inst, string feat_key, int pair_id, int pos) {
    int fea_id;
    fea_id = fea_model -> SearchFeature(feat_key);
    if (fea_id == -1) return -1;
    else {
        p_inst -> PushFctFea(fea_id, pair_id, pos);
        return fea_id;
    }
}

