//
//  LrTuckerRankingModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-29.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrTuckerRankingModel.h"

void LrTuckerRankingModel::InitModel()
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
    printf("Core Tensor: Allocate memory: %d * %d * %d \n", rank1, rank2, rank3);
    for (int a = 0; a < rank1 * rank2 * rank3; a++) core_tensor[a] = 0.0;
    core_params_g = (real*) malloc(rank1 * rank2 * rank3 * sizeof(real));
    for (int a = 0; a < rank1 * rank2 * rank3; a++) core_params_g[a] = 1.0;
    
    vec_word_repr.resize(max_list_len);
    vec_feat_repr.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_word_repr[i] = (real*) malloc(max_sent_len * rank1 * sizeof(real));
        vec_feat_repr[i] = (real*) malloc(max_sent_len * rank2 * sizeof(real));
    }
    //    label_repr = (real*) malloc(rank * sizeof(real));
    vec_structure_emb.resize(max_list_len);
    vec_part_struct_emb.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_structure_emb[i] = (real*) malloc(rank1 * rank2 * sizeof(real));
        vec_part_struct_emb[i] = (real*) malloc(rank1 * rank2 * sizeof(real));
    }
    emb_input = (real*) malloc(max_list_len * sizeof(real));
    
    vec_word_repr_sum.resize(max_list_len);
    vec_feat_repr_sum.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_word_repr_sum[i] = (real*) malloc(rank1 * sizeof(real));
        vec_feat_repr_sum[i] = (real*) malloc(rank2 * sizeof(real));
    }
    
    part_emb_input = (real*) malloc(max_list_len * sizeof(real));
}

void LrTuckerRankingModel::GetRepresentations(BaseInstance *b_inst, int pair_id, int position) {
    int a;
    long long l1,l2,l3;
    l2 = position * rank1;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    
    l3 = p_inst -> id_pairs[pair_id][position] * word_emb_dim;
    if (rank1 == word_emb_dim && update_emb == false) {
        for (a = 0; a < rank1; a++) vec_word_repr[pair_id][a + l2] = emb_model -> syn0[a + l3];
    }
    else {
        for (int i = 0; i < word_emb_dim; i++) {
            l1 = i * rank1;
            for (a = 0; a < rank1; a++) vec_word_repr[pair_id][a + l2] += emb_model -> syn0[i + l3] * emb_map[a + l1];
        }    
    }
    
    l2 = position * rank2;
    if (rank2 == num_fea && update_fea_emb == false) {
        for (int i = 0; i < p_inst -> fea_num_pairs[pair_id][position]; i++) {
            int fea_id = p_inst -> fea_vec_pairs[pair_id][position][i];
            vec_feat_repr[pair_id][fea_id + l2] += 1;
        }
    }
    else {
        for (int i = 0; i < p_inst -> fea_num_pairs[pair_id][position]; i++) {
            l1 = p_inst -> fea_vec_pairs[pair_id][position][i] * rank2;
            for (a = 0; a < rank2; a++) vec_feat_repr[pair_id][a + l2] += fea_model -> syn0[a + l1];
        }
    }
}

void LrTuckerRankingModel::ForwardViews(BaseInstance *b_inst, int pair_id) {
    //    long long l1;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    int i;
    for (i = 0; i < max_sent_len * rank1; i++) vec_word_repr[pair_id][i] = 0.0;
    for (i = 0; i < max_sent_len * rank2; i++) vec_feat_repr[pair_id][i] = 0.0;
    
    for (int i = 0; i < 2; i++) {
        if (p_inst -> id_pairs[pair_id][i] != -1 && p_inst -> fea_num_pairs[pair_id][i] != 0) {
            GetRepresentations(b_inst, pair_id, i);
        }
    }
}

void LrTuckerRankingModel::ForwardStructureEmb(BaseInstance* b_inst, int pair_id) {
    int a;
    long l1, l2, l3;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    
    for (a = 0; a < rank1 * rank2; a++) vec_structure_emb[pair_id][a] = 0.0;
    for (int position = 0; position < 2; position++) {
        if (p_inst -> id_pairs[pair_id][position] != -1 && p_inst -> fea_num_pairs[pair_id][position] != 0) {
            l2 = position * rank1;
            l3 = position * rank2;
            for (int i = 0; i < rank2; i++) {
                l1 = i * rank1;
                for (int j = 0; j < rank1; j++) {
                    vec_structure_emb[pair_id][l1 + j] += vec_word_repr[pair_id][j + l2] * vec_feat_repr[pair_id][i + l3];
                    if (debug && isnan(vec_structure_emb[pair_id][l1 + j])) {
                        cout << vec_word_repr[pair_id][j] << endl;
                        cout << vec_feat_repr[pair_id][i] << endl;    
                    }
                }
            }
        }
    }   
}

void LrTuckerRankingModel::ForwardOutputs(BaseInstance* b_inst)
{
    int c;
    long long l2;
    real sum;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    for (int i = 0; i < p_inst -> list_len; i++) {
        ForwardStructureEmb(b_inst, i);
    }
    for (c = 0; c < p_inst -> list_len; c++) emb_input[c] = 0.0;
    
    for (c = 0; c < p_inst -> list_len; c++) {
//        l1 = c * rank1 * rank2;
        sum = 0;
        for (int i = 0; i < rank2; i++) {
            l2 = i * rank1;
            for (int j = 0; j < rank1; j++) {
                sum += vec_structure_emb[c][l2 + j] * core_tensor[l2 + j];
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
    for (c = 0; c < p_inst -> list_len; c++) {
        b_inst -> scores[c] += emb_input[c];
    }
}

long LrTuckerRankingModel::BackPropLabel(BaseInstance* b_inst, real eta_real) {
    int c, y;
    long long l1, l2;
    
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    for (int a = 0; a < p_inst -> list_len; a++) part_emb_input[a] = 0.0;
    for (c = 0; c < p_inst -> list_len; c++) {
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        part_emb_input[c] = (y - b_inst -> scores[c]);
    }
    
    for (int pair_id = 0; pair_id < p_inst -> list_len; pair_id++) {
        for (int a = 0; a < rank2 * rank1; a++) vec_part_struct_emb[pair_id][a] = 0.0;
        for (int i = 0; i < rank2; i++) {
            l2 = i * rank1;
            for (int j = 0; j < rank1; j++) {
                vec_part_struct_emb[pair_id][l2 + j] += part_emb_input[pair_id] * core_tensor[l2 + j];
                if (!adagrad) {
                    core_tensor[l2 + j] += eta_real * part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j];
                }
                else {
                    core_params_g[l2 + j] += part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j] * part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j];
                    core_tensor[l2 + j] += eta_real / sqrt(core_params_g[l2 + j]) * part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j];
                    if (debug && isnan(core_tensor[l1 + l2 + j])) { 
                        cout << core_tensor[l1 + l2 + j];
                    }
                    
                }
            }
        }
    }
    return 0;
}

void LrTuckerRankingModel::BackPropViews(BaseInstance* b_inst, real eta_real, int pair_id) {
    int i, n, a;
    long long l1, l2, l3;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    
    real tmp;
    
    if (update_word) {
        for (n = 0; n < 2; n++) {
            if (p_inst -> id_pairs[pair_id][n] != -1 && p_inst -> fea_num_pairs[pair_id][n] != 0) {
                l2 = n * rank2;
                l3 = p_inst -> id_pairs[pair_id][n] * word_emb_dim;
                
                for (int a = 0; a < rank1; a++) vec_word_repr_sum[pair_id][a] = 0.0;
                for (a = 0; a < rank1; a++) {
                    for (int j = 0; j < rank2; j++) { 
                        vec_word_repr_sum[pair_id][a] += vec_part_struct_emb[pair_id][j * rank1 + a] * vec_feat_repr[pair_id][l2 + j];
                    }
                }
                for (int k = 0; k < word_emb_dim; k++) {
                    l1 = k * rank1;
                    for (i = 0; i < rank1; i++) {
                        tmp = vec_word_repr_sum[pair_id][i] * emb_model->syn0[l3 + k];
                        params_g[l1 + i] += tmp * tmp;
                        emb_map[l1 + i] += eta_real / sqrt(params_g[l1 + i]) * tmp;
                    }
                }
            }
        }
    }
    
    if (!update_word && update_word_emb && rank1 == word_emb_dim) { // fine tuning
        for (n = 0; n < 2; n++) {
            if (p_inst -> id_pairs[pair_id][n] != -1 && p_inst -> fea_num_pairs[pair_id][n] != 0) {
                l2 = n * rank2;
                l3 = p_inst -> id_pairs[pair_id][n] * word_emb_dim;
                
                for (int a = 0; a < rank1; a++) vec_word_repr_sum[pair_id][a] = 0.0;
                for (a = 0; a < rank1; a++) {
                    for (int j = 0; j < rank2; j++) { 
                        vec_word_repr_sum[pair_id][a] += vec_part_struct_emb[pair_id][j * rank1 + a] * vec_feat_repr[pair_id][l2 + j];
                    }
                }
                for (int k = 0; k < word_emb_dim; k++) {
                    tmp = vec_word_repr_sum[pair_id][k];
                    emb_model -> params_g[l3 + k] += tmp * tmp;
                    emb_model -> syn0[l3 + k] += eta_real / sqrt(emb_model -> params_g[l3 + k]) * tmp;
                }
            }
        }
    }
    
    if (update_fea_emb) {
        for (n = 0; n < 2; n++) {
            if (p_inst -> id_pairs[pair_id][n] != -1 && p_inst -> fea_num_pairs[pair_id][n] != 0) {
                l2 = n * rank1;
                for (int k = 0; k < p_inst -> fea_num_pairs[pair_id][n]; k++) {
                    l1 = p_inst -> fea_vec_pairs[pair_id][n][k] * rank2;
                    
                    for (int a = 0; a < rank2; a++) vec_feat_repr_sum[pair_id][a] = 0.0;
                    for (a = 0; a < rank2; a++) {
                        l3 = a * rank1;
                        for (int j = 0; j < rank1; j++) { 
                            vec_feat_repr_sum[pair_id][a] += vec_part_struct_emb[pair_id][l3 + j] * vec_word_repr[pair_id][l2 + j];
                        }
                    }
                    
                    for (a = 0; a < rank2; a++) {
                        fea_model -> params_g[a + l1] += vec_feat_repr_sum[pair_id][a] * vec_feat_repr_sum[pair_id][a];
                        fea_model -> syn0[a + l1] += eta_real / sqrt(fea_model -> params_g[a + l1]) * vec_feat_repr_sum[pair_id][a];
                    }
                }
            }
        }
    }
}

void LrTuckerRankingModel::ForwardProp(BaseInstance* b_inst)
{
    PPAInstance* p_inst = (PPAInstance*)b_inst;
    for (int pair_id = 0; pair_id < p_inst -> list_len; pair_id++) {
        ForwardViews(b_inst, pair_id);
    }
    ForwardOutputs(b_inst);
}

void LrTuckerRankingModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    PPAInstance* p_inst = (PPAInstance*)b_inst;
    BackPropLabel(b_inst, eta_real);
    for (int pair_id = 0; pair_id < p_inst -> list_len; pair_id++) {
        BackPropViews(b_inst, eta_real, pair_id);
    }
    for (int i = 0; i < rank2; i++) {
        int l2 = i * rank1;
        for (int j = 0; j < rank1; j++) {
            core_tensor[l2 + j] -= 0.0 * eta_real / sqrt(core_params_g[l2 + j]) * lambda * core_tensor[l2 + j];
        }
    }
}

void LrTuckerRankingModel::PrintModelInfo() {
    cout << "dimension of word embedding: " << word_emb_dim << endl;
    cout << "rank1: " << rank1 << endl;
    cout << "rank2: " << rank2 << endl;
    cout << "rank3: " << rank3 << endl;
}
