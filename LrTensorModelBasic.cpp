//
//  LrTensorModelBasic.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-2.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream> 
#include "LrTensorModelBasic.h"

void LrTensorModelBasic::InitModel()
{
    params_g = (real*) malloc(word_emb_dim * rank * sizeof(real));
    for (int a = 0; a < word_emb_dim * rank; a++) params_g[a] = 1.0;
    
    word_repr = (real*) malloc(max_sent_len * rank * sizeof(real));
    feat_repr = (real*) malloc(max_sent_len * rank * sizeof(real));
    //    label_repr = (real*) malloc(rank * sizeof(real));
    word_repr_sum = (real*) malloc(rank * sizeof(real));
    feat_repr_sum = (real*) malloc(rank * sizeof(real));
    
    emb_input = (real*) malloc(rank * sizeof(real));
    part_emb_input = (real*) malloc(rank * sizeof(real));
    grad_s = (real*) malloc(num_labels * sizeof(real));
}

void LrTensorModelBasic::GetRepresentations(BaseInstance *b_inst, int position) {
    int a;
    long long l1,l2;
    
    l2 = position * rank;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    l1 = p_inst -> word_ids[position] * word_emb_dim;
    for (a = 0; a < word_emb_dim; a++) word_repr[a + l2] += emb_model -> syn0[a + l1];
    
    for (int i = 0; i < p_inst -> fct_nums_fea[position]; i++) {
        l1 = p_inst -> fct_fea_ids[position][i] * rank;
        for (a = 0; a < rank; a++) feat_repr[a + l2] += fea_model -> syn0[a + l1];
    }
}

void LrTensorModelBasic::ForwardViews(BaseInstance *b_inst) {
    //    long long l1;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    int i;
    for (i = 0; i < max_sent_len * rank; i++) word_repr[i] = 0.0;
    for (i = 0; i < max_sent_len * rank; i++) feat_repr[i] = 0.0;
    
    for (int i = 0; i < p_inst->len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            GetRepresentations(b_inst, i);
        }
    }
}

void LrTensorModelBasic::ForwardOutputs(BaseInstance* b_inst)
{
    int a, c, i;
    long long l1, l2;
    real sum;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    for (a = 0; a < rank; a++) emb_input[a] = 0.0;
    
    for (i = 0; i < p_inst -> len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            l2 = i * rank;
            
            for (a = 0; a < rank; a++) emb_input[a] += word_repr[a + l2] * feat_repr[a + l2];
            if (isinf(emb_input[0])) {
                cout << emb_input[0] << endl;
            }
        }
    }
    if (debug) {
    cout << emb_input[0] << endl;
    cout << emb_input[199] << endl;
    }
    
    word2int::iterator iter;
    for (c = 0; c < num_labels; c++) {
        l1 = c * rank;
        sum = 0;
        for (a = 0; a < rank; a++) {
            sum += lab_model -> syn0[a + l1] * emb_input[a];
            if (isnan(sum)) {
                cout << sum << endl;
                exit(-1);
            }
        }
        b_inst -> scores[c] += sum;
    }
}

long LrTensorModelBasic::BackPropLabel(BaseInstance* b_inst, real eta_real) {
    int c, y;
    long long l1;
    for (c = 0; c < num_labels; c++) {
        l1 = c * rank;
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        grad_s[c] = (y - b_inst -> scores[c]);
        for (int a = 0; a < rank; a++) {
            if (!adagrad) {
                lab_model -> syn0[l1 + a] += eta_real * grad_s[c] * emb_input[a];
            }
            else {
                lab_model -> params_g[l1 + a] += grad_s[c] * emb_input[a] * grad_s[c] * emb_input[a];
                lab_model -> syn0[l1 + a] += eta_real / sqrt(lab_model -> params_g[a + l1]) * grad_s[c] * emb_input[a];
            }
        }
        if (debug) cout << lab_model -> syn0[l1] << endl;
    }
    return 0;
}

void LrTensorModelBasic::BackPropViews(BaseInstance* b_inst, real eta_real) {
    int i, a, n, c;
    long long l1, l2, l_lab;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    for (a = 0; a < rank; a++) part_emb_input[a] = 0.0;
    
    for (c = 0; c < num_labels; c++) {
        l_lab = c * rank;
        
        for (a = 0; a < rank; a++) {
            part_emb_input[a] += grad_s[c] * lab_model -> syn0[a + l_lab];
        }
        /* old version
        if (update_word) {
            for (n = 0; n < b_inst -> len; n++) {
                if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                    l1 = p_inst -> word_ids[n] * word_emb_dim;
                    l2 = n * rank;
                    if (!adagrad){
                        for (a = 0; a < word_emb_dim; a++) {
                            emb_model -> syn0[a + l1] += eta_real * grad_s[c] * lab_model -> syn0[a + l_lab] *feat_repr[a + l2];
                        }
                    }
                    else {
                        for (a = 0; a < word_emb_dim; a++) {
                            emb_model -> params_g[a + l1] += grad_s[c] * lab_model -> syn0[a + l_lab] *feat_repr[a + l2] * grad_s[c] * lab_model -> syn0[a + l_lab] *feat_repr[a + l2];
                            emb_model -> syn0[a + l1] += eta_real / sqrt(emb_model -> params_g[a + l1]) * lab_model -> syn0[a + l_lab] *feat_repr[a + l2] * grad_s[c];
                        }
                    }
                }
            }
        }
        
        for (n = 0; n < b_inst -> len; n++) {
            if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                l2 = n * rank;
                for (int k = 0; k < p_inst -> fct_nums_fea[n]; k++) {
                    l1 = p_inst -> fct_fea_ids[n][k] * rank;
                    if (! adagrad) {
                        for (i = 0; i < rank; i++) fea_model -> syn0[l1 + i] += eta_real * lab_model -> syn0[i + l_lab] * grad_s[c] * word_repr[i + l2];
                    }
                    else {
                        for (i = 0; i < rank; i++) {
                            fea_model -> params_g[l1 + i] += lab_model -> syn0[i + l_lab] * grad_s[c] * word_repr[i + l2] * lab_model -> syn0[i + l_lab] * grad_s[c] * word_repr[i + l2];
                            fea_model -> syn0[l1 + i] += eta_real / sqrt(fea_model -> params_g[l1 + i]) * lab_model -> syn0[i + l_lab] * grad_s[c] * word_repr[i + l2];
                        }
                    }
                }
            }
        }
         */
    }
    
    if (debug) {
        cout << part_emb_input[0] << endl;
        cout << part_emb_input[rank - 1] << endl;
    }
    
    if (update_word) {
        for (n = 0; n < b_inst -> len; n++) {
            if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                l1 = p_inst -> word_ids[n] * word_emb_dim;
                l2 = n * rank;
                if (!adagrad){
                    for (a = 0; a < word_emb_dim; a++) {
                        emb_model -> syn0[a + l1] += eta_real * part_emb_input[i] * feat_repr[a + l2];
                    }
                }
                else {
                    for (a = 0; a < word_emb_dim; a++) {
                        emb_model -> params_g[a + l1] += part_emb_input[i] * feat_repr[a + l2] * part_emb_input[i] * feat_repr[a + l2];
                        emb_model -> syn0[a + l1] += eta_real / sqrt(emb_model -> params_g[a + l1]) * part_emb_input[i] * feat_repr[a + l2];
                    }
                }
            }
        }
    }
    for (n = 0; n < b_inst -> len; n++) {
        if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
            l2 = n * rank;
            for (int k = 0; k < p_inst -> fct_nums_fea[n]; k++) {
                l1 = p_inst -> fct_fea_ids[n][k] * rank;
                if (! adagrad) {
                    for (i = 0; i < rank; i++) fea_model -> syn0[l1 + i] += eta_real * part_emb_input[i] * word_repr[i + l2];
                }
                else {
                    for (i = 0; i < rank; i++) {
                        fea_model -> params_g[l1 + i] += part_emb_input[i] * word_repr[i + l2] * part_emb_input[i] * word_repr[i + l2];
                        fea_model -> syn0[l1 + i] += eta_real / sqrt(fea_model -> params_g[l1 + i]) * part_emb_input[i] * word_repr[i + l2];
                    }
                }
            }
        }
    }
}

void LrTensorModelBasic::ForwardProp(BaseInstance* b_inst)
{
    ForwardViews(b_inst);
    ForwardOutputs(b_inst);
}

void LrTensorModelBasic::BackProp(BaseInstance* b_inst, real eta_real)
{
    BackPropLabel(b_inst, eta_real);
    BackPropViews(b_inst, eta_real);
}

void LrTensorModelBasic::PrintModelInfo() {
    cout << word_emb_dim << endl;
    cout << rank << endl;
}
