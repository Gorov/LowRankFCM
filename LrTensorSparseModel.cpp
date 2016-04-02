//
//  LrTensorSparseModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-10.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrTensorSparseModel.h"

void LrSparseTensorModel::InitModel()
{
    num_group = rank / group_size;
    if (!sparse_map) {
        LrTensorModel::InitModel();
    }
    else {
        emb_map = (real*) malloc(word_emb_dim * group_size * sizeof(real));
        printf("FeaEmb: Allocate memory: %d * %d\n", group_size, word_emb_dim);
        for (int a = 0; a < group_size * word_emb_dim; a++) emb_map[a] = (rand() / (real)RAND_MAX - 0.5) / 10;//word_emb_dim;
        
        params_g = (real*) malloc(word_emb_dim * group_size * sizeof(real));
        for (int a = 0; a < word_emb_dim * group_size; a++) params_g[a] = 1.0;
        
        word_repr = (real*) malloc(max_sent_len * rank * sizeof(real));
        feat_repr = (real*) malloc(max_sent_len * rank * sizeof(real));
        word_repr_sum = (real*) malloc(rank * sizeof(real));
        feat_repr_sum = (real*) malloc(rank * sizeof(real));
        
        emb_input = (real*) malloc(rank * sizeof(real));
        part_emb_input = (real*) malloc(rank * sizeof(real));
        grad_s = (real*) malloc(num_labels * sizeof(real));
        
        grad_emb_map = (real*) malloc(rank * word_emb_dim * sizeof(real));
    }
}

void LrSparseTensorModel::GetRepresentations(BaseInstance *b_inst, int position) {
    int a;
    long long l1,l2,l3;
    int2int::iterator iter;
    int feaid;
    int start, end;
    
    l2 = position * rank;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    l1 = p_inst -> word_ids[position] * word_emb_dim;
    //    for (int i = 0; i < rank; i++) {
    //        l3 = i * word_emb_dim;
    //        for (a = 0; a < word_emb_dim; a++) word_repr[i + l2] += emb_model -> syn0[a + l1] * emb_map[l3 + a];
    //    }
    
    if (!sparse_map) {
        l3 = p_inst -> word_ids[position] * word_emb_dim;
        for (int i = 0; i < word_emb_dim; i++) {
            l1 = i * rank;
            for (a = 0; a < rank; a++) word_repr[a + l2] += emb_model -> syn0[i + l3] * emb_map[a + l1];
        }
    }
    else {
        l3 = p_inst -> word_ids[position] * word_emb_dim;
        for (int i = 0; i < word_emb_dim; i++) {
            l1 = i * group_size;
            for (a = 0; a < group_size; a++) word_repr[a + l2] += emb_model -> syn0[i + l3] * emb_map[a + l1];
        }
        for (int i = 1; i < num_group; i++) {
            l1 = i * group_size;
            for (a = 0; a < group_size; a++) word_repr[a + l2 + l1] = word_repr[a + l2];
        }
    }
    //    for (a = 0; a < rank; a++) word_repr[a + l2] = 1.0;
    
    for (int i = 0; i < p_inst -> fct_nums_fea[position]; i++) {
        feaid = p_inst -> fct_fea_ids[position][i];
        iter = feat_group -> find(feaid);
        l1 = feaid * rank;
        start = iter -> second * group_size;
        end = start + group_size;
        for (a = start; a < end; a++) feat_repr[a + l2] += fea_model -> syn0[a + l1];
    }
}

void LrSparseTensorModel::BackPropViews(BaseInstance* b_inst, real eta_real) {
    int i, j, n, c;
    long long l1, l2, l3, l4, l_lab;
    
    int2int::iterator iter;
    int feaid;
    int start, end;
    
    for (i = 0; i < word_emb_dim * rank; i++) grad_emb_map[i] = 0.0;
    for (i = 0; i < rank; i++) part_emb_input[i] = 0.0;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    if (update_word) {
        if (!sparse_map) {
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
        else {
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
            for (int i = 1; i < num_group; i++) {
                l1 = i * group_size;
                for (int a = 0; a < group_size; a++) {
                    l2 = a * word_emb_dim;
                    l3 = (a + l1) * word_emb_dim;
                    for (j = 0; j < word_emb_dim; j++) {
                        grad_emb_map[l2 + j] += grad_emb_map[l3 + j];
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
        if (!sparse_map) {
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
        else {
            for (n = 0; n < b_inst -> len; n++) {
                if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                    l2 = n * rank;
                    l3 = p_inst -> word_ids[n] * word_emb_dim;
                    for (int k = 0; k < word_emb_dim; k++) {
                        l1 = k * group_size;
                        for (int g = 0; g < num_group; g++) {
                            l4 = g * group_size;
                            for (i = 0; i < group_size; i++) {
                                tmp = part_emb_input[i + l4] * feat_repr[i + l4 + l2] * emb_model->syn0[l3 + k];
                                params_g[l1 + i] += tmp * tmp;
                                emb_map[l1 + i] += eta_real / sqrt(params_g[l1 + i]) * tmp;
                            }
                        }
                    }
                }
            }
        }
    }
    
    for (n = 0; n < b_inst -> len; n++) {
        if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
            l2 = n * rank;
            for (int k = 0; k < p_inst -> fct_nums_fea[n]; k++) {
                feaid = p_inst -> fct_fea_ids[n][k];
                iter = feat_group -> find(feaid);
                l1 = feaid * rank;
                start = iter -> second * group_size;
                end = start + group_size;

                {
                    for (i = start; i < end; i++) {
                        fea_model -> params_g[l1 + i] += part_emb_input[i] * word_repr[i + l2] * part_emb_input[i] * word_repr[i + l2];
                        fea_model -> syn0[l1 + i] += eta_real / sqrt(fea_model -> params_g[l1 + i]) * part_emb_input[i] * word_repr[i + l2];
                    }
                }
            }
        }
    }
}

void LrSparseTensorModel::PrintModelInfo() {
    cout << "dimension of word embedding: " << word_emb_dim << endl;
    cout << "rank: " << rank << endl;
    cout << "number of group: " << num_group << endl;
    cout << "dimension of feature: " << group_size << endl;
}

