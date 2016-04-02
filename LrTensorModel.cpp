//
//  LrTensorModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrTensorModel.h"

void LrTensorModel::InitModel()
{
    emb_map = (real*) malloc(word_emb_dim * rank * sizeof(real));
    printf("FeaEmb: Allocate memory: %d * %d\n", rank, word_emb_dim);
//    for (int a = 0; a < word_emb_dim * rank; a++) emb_map[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
    for (int a = 0; a < rank * word_emb_dim; a++) emb_map[a] = (rand() / (real)RAND_MAX - 0.5) / 10;//word_emb_dim;
    if (word_emb_dim == rank) {
//        for (int a = 0; a < rank; a++) emb_map[a + a * rank] = 1.0;
    }
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
    
    grad_emb_map = (real*) malloc(rank * word_emb_dim * sizeof(real));
}

void LrTensorModel::GetRepresentations(BaseInstance *b_inst, int position) {
    int a;
    long long l1,l2,l3;
    l2 = position * rank;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    l1 = p_inst -> word_ids[position] * word_emb_dim;
//    for (int i = 0; i < rank; i++) {
//        l3 = i * word_emb_dim;
//        for (a = 0; a < word_emb_dim; a++) word_repr[i + l2] += emb_model -> syn0[a + l1] * emb_map[l3 + a];
//    }
    
    l3 = p_inst -> word_ids[position] * word_emb_dim;
    for (int i = 0; i < word_emb_dim; i++) {
        l1 = i * rank;
        for (a = 0; a < rank; a++) word_repr[a + l2] += emb_model -> syn0[i + l3] * emb_map[a + l1];
    }
//    for (a = 0; a < rank; a++) word_repr[a + l2] = 1.0;
    
    for (int i = 0; i < p_inst -> fct_nums_fea[position]; i++) {
        l1 = p_inst -> fct_fea_ids[position][i] * rank;
        for (a = 0; a < rank; a++) feat_repr[a + l2] += fea_model -> syn0[a + l1];
    }
}

void LrTensorModel::ForwardViews(BaseInstance *b_inst) {
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

void LrTensorModel::ForwardOutputs(BaseInstance* b_inst)
{
    int a, c, i;
    long long l1, l2;
    real sum;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    for (a = 0; a < rank; a++) emb_input[a] = 0.0;
    
    for (i = 0; i < p_inst -> len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            l2 = i * rank;
//            for (a = 0; a < rank; a++) word_repr_sum[a] += word_repr[a + l2];
//            for (a = 0; a < rank; a++) feat_repr_sum[a] += feat_repr[a + l2];
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
//            sum += lab_model -> syn0[a + l1] * word_repr_sum[a] * feat_repr_sum[a];
            sum += lab_model -> syn0[a + l1] * emb_input[a];
            if (isnan(sum)) {
                cout << sum << endl;
                exit(-1);
            }
        }
        b_inst -> scores[c] += sum;
    }
}

long LrTensorModel::BackPropLabel(BaseInstance* b_inst, real eta_real) {
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
                lab_model -> syn0[l1 + a] += eta_real / sqrt(lab_model -> params_g[l1 + a]) * grad_s[c] * emb_input[a];
            }
        }
        if (debug) cout << lab_model -> syn0[l1] << endl;
    }
    return 0;
}

void LrTensorModel::BackPropViews(BaseInstance* b_inst, real eta_real) {
    int i, j, n, c;
    long long l1, l2, l3, l_lab;
    for (i = 0; i < word_emb_dim * rank; i++) grad_emb_map[i] = 0.0;
    for (i = 0; i < rank; i++) part_emb_input[i] = 0.0;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    if (update_word) {
        for (n = 0; n < b_inst -> len; n++) {
            if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                l1 = p_inst -> word_ids[n] * word_emb_dim;
                l2 = n * rank;
                for (i = 0; i < rank; i++) {
                    l3 = i * word_emb_dim;
                    for (j = 0; j < word_emb_dim; j++) {
                        grad_emb_map[l3 + j] += feat_repr[i + l2] * emb_model -> syn0[j + l1];
                    }
                }
            }
        }
    }
    for (c = 0; c < num_labels; c++) {
        l_lab = c * rank;
        
        for (i = 0; i < rank; i++) {
            part_emb_input[i] += grad_s[c] * lab_model -> syn0[i + l_lab];
        }
        /*
        if (update_word && debug) {
            for (i = 0; i < rank; i++) {
                l3 = i * word_emb_dim;
                for (j = 0; j < word_emb_dim; j++) {
                    if (!adagrad) {
                        emb_map[j + l3] += eta_real * grad_s[c] * lab_model -> syn0[i + l_lab] * grad_emb_map[j + l3];
                    }
                    else {
                        params_g[j + l3] += lab_model -> syn0[i + l_lab] * grad_s[c] * grad_emb_map[j + l3] * lab_model -> syn0[i + l_lab] * grad_s[c] * grad_emb_map[j + l3];
                        emb_map[j + l3] += eta_real / sqrt(params_g[j + l3]) * lab_model -> syn0[i + l_lab] * grad_s[c] * grad_emb_map[j + l3];
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
    
    real tmp;
    if (false) {//if (update_word) {
        for (i = 0; i < rank; i++) {
            l3 = i * word_emb_dim;
            for (j = 0; j < word_emb_dim; j++) {
//                if (j != i) continue;
                if (!adagrad) {
                    emb_map[j + l3] += eta_real * part_emb_input[i] * grad_emb_map[j + l3];
                }
                else {
                    params_g[j + l3] += part_emb_input[i] * grad_emb_map[j + l3] * part_emb_input[i] * grad_emb_map[j + l3];
                    emb_map[j + l3] += eta_real / sqrt(params_g[j + l3]) * part_emb_input[i] * grad_emb_map[j + l3];
                }
            }
        }
    }
    
    if (update_word) {
        for (n = 0; n < b_inst -> len; n++) {
            if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                l2 = n * rank;
                l3 = p_inst -> word_ids[n] * word_emb_dim;
                for (int k = 0; k < word_emb_dim; k++) {
                    l1 = k * rank;
                    for (i = 0; i < rank; i++) {
                        tmp = part_emb_input[i] * feat_repr[i + l2] * emb_model->syn0[l3 + k];
                        params_g[l1 + i] += tmp * tmp;
                        emb_map[l1 + i] += eta_real / sqrt(params_g[l1 + i]) * tmp;
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
                {
                    for (i = 0; i < rank; i++) {
                        fea_model -> params_g[l1 + i] += part_emb_input[i] * word_repr[i + l2] * part_emb_input[i] * word_repr[i + l2];
                        fea_model -> syn0[l1 + i] += eta_real / sqrt(fea_model -> params_g[l1 + i]) * part_emb_input[i] * word_repr[i + l2];
                    }
                }
            }
        }
    }
}

void LrTensorModel::ForwardProp(BaseInstance* b_inst)
{
    ForwardViews(b_inst);
    ForwardOutputs(b_inst);
}

void LrTensorModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    BackPropLabel(b_inst, eta_real);
    BackPropViews(b_inst, eta_real);
}

void LrTensorModel::PrintModelInfo() {
    cout << "dimension of word embedding: " << word_emb_dim << endl;
    cout << "rank: " << rank << endl;
}

