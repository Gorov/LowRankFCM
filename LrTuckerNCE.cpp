//
//  LrTuckerNCE.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-7.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include "LrTuckerNCE.h"

void LrTuckerNCE::InitModel()
{
    emb_map = (real*) malloc(word_emb_dim * rank1 * sizeof(real));
    printf("FeaEmb: Allocate memory: %d * %d\n", rank1, word_emb_dim);
    for (int a = 0; a < word_emb_dim * rank1; a++) emb_map[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
    //    for (int a = 0; a < rank1 * word_emb_dim; a++) emb_map[a] = (rand() / (real)RAND_MAX - 0.5) / 10;//word_emb_dim;
    if (word_emb_dim == rank1) {
        for (int a = 0; a < rank1; a++) emb_map[a + a * rank1] = 1.0;
    }
    params_g = (real*) malloc(word_emb_dim * rank1 * sizeof(real));
    for (int a = 0; a < word_emb_dim * rank1; a++) params_g[a] = 1.0;
    
    core_tensor = (real*) malloc(rank1 * rank2 * rank3 * sizeof(real));
    for (int a = 0; a < rank1 * rank2 * rank3; a++) core_tensor[a] = 0.0;
    core_params_g = (real*) malloc(rank1 * rank2 * rank3 * sizeof(real));
    for (int a = 0; a < rank1 * rank2 * rank3; a++) core_params_g[a] = 1.0;
    
    word_repr = (real*) malloc(max_sent_len * rank1 * sizeof(real));
    feat_repr = (real*) malloc(max_sent_len * rank2 * sizeof(real));
    //    label_repr = (real*) malloc(rank * sizeof(real));
    word_repr_sum = (real*) malloc(rank1 * sizeof(real));
    feat_repr_sum = (real*) malloc(rank2 * sizeof(real));
    
    emb_input = (real*) malloc(rank3 * sizeof(real));
    part_emb_input = (real*) malloc(rank3 * sizeof(real));
    
    structure_emb = (real*) malloc(rank1 * rank2 * sizeof(real));
    part_struct_emb = (real*) malloc(rank1 * rank2 * sizeof(real));
    
    neg_samples.resize(num_neg + 1);
    grad_s = (real*) malloc((num_neg + 1) * sizeof(real));
}

void LrTuckerNCE::ForwardOutputs(BaseInstance* b_inst)
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
    for (c = 0; c < num_neg + 1; c++) {
        l1 = neg_samples[c] * rank3;
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

long LrTuckerNCE::BackPropLabel(BaseInstance* b_inst, real eta_real) {
    int c, y;
    long long l1, l2;
    
    for (int a = 0; a < rank3; a++) part_emb_input[a] = 0.0;
    for (c = 0; c < num_neg; c++) {
        l1 = neg_samples[c] * rank3;
        if (1 == c) y = 1;
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

void LrTuckerNCE::PrintModelInfo() {
    LrTensorTuckerModel::PrintModelInfo();
    cout << "num_neg: " << num_neg << endl;
}
