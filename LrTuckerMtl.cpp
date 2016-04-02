//
//  LrTuckerMtl.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-26.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include "LrTuckerMtl.h"

void LrTuckerMtl::ForwardOutputs(BaseInstance* b_inst, int task_id)
{
    int a, c;
    long long l1, l2;
    real sum;
    
    ForwardStructureEmb(b_inst);
    for (c = 0; c < rank3; c++) emb_input[c] = 0.0;
    
    for (c = 0; c < rank3; c++) {
        l1 = c * rank1 * rank2;
        sum = 0;
        for (int i = 0; i < rank2; i++) {
            l2 = i * rank1;
            for (int j = 0; j < rank1; j++) {
                sum += structure_emb[l2 + j] * core_tensor[l1 + l2 + j];
                if (debug && isnan(sum)) {
                    cout << sum << endl;
                }
            }
        }
        emb_input[c] += sum;
        if (debug && isnan(emb_input[c])) {
            cout << emb_input[c] << endl;
        }
    }
    
    word2int::iterator iter;
    for (c = start[task_id]; c < start[task_id] + length[task_id]; c++) {
        l1 = c * rank3;
        sum = 0;
        for (a = 0; a < rank3; a++) {
            sum += lab_model -> syn0[a + l1] * emb_input[a];
            if (isnan(sum)) {
                cout << sum << endl;
                exit(-1);
            }
        }
        b_inst -> scores[c] += sum;
    }
}


long LrTuckerMtl::BackPropLabel(BaseInstance* b_inst, real eta_real, int task_id) {
    int c, y;
    long long l1, l2;
    
    for (int a = 0; a < rank3; a++) part_emb_input[a] = 0.0;
    for (c = start[task_id]; c < start[task_id] + length[task_id]; c++) {
        l1 = c * rank3;
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        grad_s[c] = (y - b_inst -> scores[c]);
        for (int a = 0; a < rank3; a++) {
            part_emb_input[a] += grad_s[c] * lab_model -> syn0[l1 + a];
            if (update_lab_emb) {
                if (!adagrad) {
                    lab_model -> syn0[l1 + a] += eta_real * grad_s[c] * emb_input[a];
                }
                else {
                    lab_model -> params_g[l1 + a] += grad_s[c] * emb_input[a] * grad_s[c] * emb_input[a];
                    lab_model -> syn0[l1 + a] += eta_real / sqrt(lab_model -> params_g[l1 + a]) * grad_s[c] * emb_input[a];
                }
            }
        }
        if (debug) cout << lab_model -> syn0[l1] << endl;
    }
    
    for (int a = 0; a < rank2 * rank1; a++) part_struct_emb[a] = 0.0;
    for (c = 0; c < rank3; c++) {
        l1 = c * rank1 * rank2;
        for (int i = 0; i < rank2; i++) {
            l2 = i * rank1;
            for (int j = 0; j < rank1; j++) {
                part_struct_emb[l2 + j] += part_emb_input[c] * core_tensor[l1 + l2 + j];
                if (!adagrad) {
                    core_tensor[l1 + l2 + j] += eta_real * part_emb_input[c] * structure_emb[l2 + j];
                }
                else {
                    core_params_g[l1 + l2 + j] += part_emb_input[c] * structure_emb[l2 + j] * part_emb_input[c] * structure_emb[l2 + j];
                    core_tensor[l1 + l2 + j] += eta_real / core_params_g[l1 + l2 + j] * part_emb_input[c] * structure_emb[l2 + j];
                    if (debug && isnan(core_tensor[l1 + l2 + j])) { 
                        cout << core_tensor[l1 + l2 + j];
                    }
                    
                }
            }
        }
    }
    return 0;
}

void LrTuckerMtl::ForwardProp(BaseInstance* b_inst, int task_id)
{
    ForwardViews(b_inst);
    ForwardOutputs(b_inst, task_id);
}

void LrTuckerMtl::BackProp(BaseInstance* b_inst, real eta_real, int task_id)
{
    BackPropLabel(b_inst, eta_real, task_id);
    BackPropViews(b_inst, eta_real);
}


void LrTuckerMtl::PrintModelInfo() {
    LrTensorTuckerModel::PrintModelInfo();
    cout << "Num of labels in Task1: " << length[0] << endl;
    cout << "Num of labels in Task2: " << length[1] << endl;
}