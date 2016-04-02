//
//  LrRankingModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-2-8.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrRankingModel.h"

void LrRankingModel::InitModel()
{
//    emb_map = (real*) malloc(word_emb_dim * rank * sizeof(real));
//    printf("FeaEmb: Allocate memory: %d * %d\n", rank, word_emb_dim);
//    for (int a = 0; a < word_emb_dim * rank; a++) emb_map[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
//    //    for (int a = 0; a < rank1 * word_emb_dim; a++) emb_map[a] = (rand() / (real)RAND_MAX - 0.5) / 10;//word_emb_dim;
//    if (word_emb_dim == rank) {
//        for (int a = 0; a < rank; a++) emb_map[a + a * rank] = 1.0;
//    }
//    params_g = (real*) malloc(word_emb_dim * rank * sizeof(real));
//    for (int a = 0; a < word_emb_dim * rank; a++) params_g[a] = 1.0;
    
    vec_emb_map.resize(2);
    vec_params_g.resize(2);
    for (int i = 0; i < 2; i++) {
        vec_emb_map[i] = (real*) malloc(word_emb_dim * rank * sizeof(real));
        printf("FeaEmb: Allocate memory: %d * %d\n", rank, word_emb_dim);
        for (int a = 0; a < word_emb_dim * rank; a++) vec_emb_map[i][a] = (rand() / (real)RAND_MAX - 0.5) * 10 / rank;// 0.0;
        if (word_emb_dim == rank) {
            for (int a = 0; a < rank; a++) vec_emb_map[i][a + a * rank] = 1.0;
        }
        vec_params_g[i] = (real*) malloc(word_emb_dim * rank * sizeof(real));
        for (int a = 0; a < word_emb_dim * rank; a++) vec_params_g[i][a] = 1.0;
    }
    
    if (tucker) {
        core_tensor = (real*) malloc(rank * rank2 * sizeof(real));
        for (int a = 0; a < rank * rank2; a++) core_tensor[a] = 0.0;
		printf("Core tensor: Allocate memory: %d * %d\n", rank, rank2);
        core_params_g = (real*) malloc(rank * rank2 * sizeof(real));
        for (int a = 0; a < rank * rank2; a++) core_params_g[a] = 1.0;
    }
    else {
        core_tensor = (real*) malloc(rank* sizeof(real));
        for (int a = 0; a < rank; a++) core_tensor[a] = 0.0;
		printf("Core tensor: Allocate memory: %d\n", rank);
        core_params_g = (real*) malloc(rank * sizeof(real));
        for (int a = 0; a < rank; a++) core_params_g[a] = 1.0;
    }
    
    vec_word_repr.resize(max_list_len);
    vec_feat_repr.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_word_repr[i] = (real*) malloc(max_sent_len * rank * sizeof(real));
        vec_feat_repr[i] = (real*) malloc(max_sent_len * rank2 * sizeof(real));
    }
    
    vec_structure_emb.resize(max_list_len);
    vec_part_struct_emb.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_structure_emb[i] = (real*) malloc(rank * rank2 * sizeof(real));
        vec_part_struct_emb[i] = (real*) malloc(rank * rank2 * sizeof(real));
    }
    
    vec_word_repr_sum.resize(max_list_len);
    vec_feat_repr_sum.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_word_repr_sum[i] = (real*) malloc(rank * sizeof(real));
        vec_feat_repr_sum[i] = (real*) malloc(rank2 * sizeof(real));
    }
    
    emb_input = (real*) malloc(max_list_len * sizeof(real));
    part_emb_input = (real*) malloc(max_list_len * sizeof(real));
    
    vec_bigram_repr.resize(max_list_len);
    for (int i = 0; i < max_list_len; i++) {
        vec_bigram_repr[i] = (real*) malloc(rank * sizeof(real));
    }
}

void LrRankingModel::GetRepresentations(BaseInstance *b_inst, int pair_id) {
    int a;
    long long l1,l2,l3;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    for (int position = 0; position < 2; position++) {
        l2 = position * rank;
        
        if (p_inst -> id_pairs[pair_id][position] == -1) {
            for (a = 0; a < rank; a++) vec_word_repr[pair_id][a + l2] = 1.0;
            continue;
        }
        
        l3 = p_inst -> id_pairs[pair_id][position] * word_emb_dim;
        if (rank == word_emb_dim && update_emb == false) {
            for (a = 0; a < rank; a++) vec_word_repr[pair_id][a + l2] = emb_model -> syn0[a + l3];
        }
        else {
            for (int i = 0; i < word_emb_dim; i++) {
                l1 = i * rank;
                for (a = 0; a < rank; a++) vec_word_repr[pair_id][a + l2] += emb_model -> syn0[i + l3] * vec_emb_map[position][a + l1];
            }    
        }
    }
    if (p_inst -> id_pairs[pair_id][0] == -1 && p_inst -> id_pairs[pair_id][1] == -1) for (a = 0; a < rank; a++) vec_bigram_repr[pair_id][a] = 0.0;
    else for (a = 0; a < rank; a++) vec_bigram_repr[pair_id][a] = vec_word_repr[pair_id][a] * vec_word_repr[pair_id][a + rank];
    
    int position = 0;
    if (rank2 == num_fea && update_fea_emb == false) {
        for (int i = 0; i < p_inst -> fea_num_pairs[pair_id][position]; i++) {
            int fea_id = p_inst -> fea_vec_pairs[pair_id][position][i];
            vec_feat_repr[pair_id][fea_id] += 1;
        }
    }
    else {
        for (int i = 0; i < p_inst -> fea_num_pairs[pair_id][position]; i++) {
            l1 = p_inst -> fea_vec_pairs[pair_id][position][i] * rank2;
            for (a = 0; a < rank2; a++) vec_feat_repr[pair_id][a] += fea_model -> syn0[a + l1];
        }
    }
}

void LrRankingModel::ForwardViews(BaseInstance *b_inst, int pair_id) {
    int i;
    for (i = 0; i < 2 * rank; i++) vec_word_repr[pair_id][i] = 0.0;
    if (!tucker) {
        for (i = 0; i < rank; i++) vec_feat_repr[pair_id][i] = 0.0;
    }
    else {
        for (i = 0; i < rank2; i++) vec_feat_repr[pair_id][i] = 0.0;
    }
    
    GetRepresentations(b_inst, pair_id);
}

void LrRankingModel::ForwardStructureEmb(BaseInstance* b_inst, int pair_id) {
    int a;
    long l1;
//    PPAInstance* p_inst = (PPAInstance*) b_inst;
    
    if (tucker) {
        for (a = 0; a < rank * rank2; a++) vec_structure_emb[pair_id][a] = 0.0;
        for (int i = 0; i < rank2; i++) {
            l1 = i * rank;
            for (int j = 0; j < rank; j++) {
                vec_structure_emb[pair_id][l1 + j] += vec_bigram_repr[pair_id][j] * vec_feat_repr[pair_id][i];
            }
        }
    }
    else {
        for (a = 0; a < rank; a++) vec_structure_emb[pair_id][a] = 0.0;
        for (a = 0; a < rank; a++) vec_structure_emb[pair_id][a] = vec_bigram_repr[pair_id][a] * vec_feat_repr[pair_id][a];
    }
}

void LrRankingModel::ForwardOutputs(BaseInstance* b_inst)
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
        sum = 0;
        if (tucker) {
            for (int i = 0; i < rank2; i++) {
                l2 = i * rank;
                for (int j = 0; j < rank; j++) {
                    sum += vec_structure_emb[c][l2 + j] * core_tensor[l2 + j];
                }
            }
        }
        else {
            for (int i = 0; i < rank; i++) sum += (vec_structure_emb[c][i] * core_tensor[i]);
        }
        emb_input[c] = sum;
    }
    
    word2int::iterator iter;
    for (c = 0; c < p_inst -> list_len; c++) {
        b_inst -> scores[c] += emb_input[c];
    }
}

long LrRankingModel::BackPropLabel(BaseInstance* b_inst, real eta_real) {
    int c, y;
    
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    for (int a = 0; a < p_inst -> list_len; a++) part_emb_input[a] = 0.0;
    for (c = 0; c < p_inst -> list_len; c++) {
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        part_emb_input[c] = (y - b_inst -> scores[c]);
    }
    
    for (int pair_id = 0; pair_id < p_inst -> list_len; pair_id++) {
        if (tucker) {
            long long l2;
            for (int a = 0; a < rank2 * rank; a++) vec_part_struct_emb[pair_id][a] = 0.0;
            for (int i = 0; i < rank2; i++) {
                l2 = i * rank;
                for (int j = 0; j < rank; j++) {
                    vec_part_struct_emb[pair_id][l2 + j] += part_emb_input[pair_id] * core_tensor[l2 + j];

                    core_params_g[l2 + j] += part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j] * part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j];
                    core_tensor[l2 + j] += eta_real / sqrt(core_params_g[l2 + j]) * part_emb_input[pair_id] * vec_structure_emb[pair_id][l2 + j];
                }
            }
        }
        else {
            for (int a = 0; a < rank; a++) vec_part_struct_emb[pair_id][a] = 0.0;
            for (int i = 0; i < rank; i++) {
                vec_part_struct_emb[pair_id][i] += part_emb_input[pair_id] * core_tensor[i];

                core_params_g[i] += part_emb_input[pair_id] * vec_structure_emb[pair_id][i] * part_emb_input[pair_id] * vec_structure_emb[pair_id][i];
                core_tensor[i] += eta_real / sqrt(core_params_g[i]) * part_emb_input[pair_id] * vec_structure_emb[pair_id][i];
            }
        }
    }
    return 0;
}

void LrRankingModel::BackPropViews(BaseInstance* b_inst, real eta_real, int pair_id) {
    int i, n, a;
    long long l1, l2, l3;
    PPAInstance* p_inst = (PPAInstance*) b_inst;
    
    real tmp;
    
    if (update_word) {
        for (n = 0; n < 2; n++) {
            if (p_inst -> id_pairs[pair_id][n] == -1) {
                continue;
            }
            
            l3 = p_inst -> id_pairs[pair_id][n] * word_emb_dim;
                
            for (int a = 0; a < rank; a++) vec_word_repr_sum[pair_id][a] = 0.0;
            for (a = 0; a < rank; a++) {
                for (int j = 0; j < rank2; j++) { 
                    vec_word_repr_sum[pair_id][a] += vec_part_struct_emb[pair_id][j * rank + a] * vec_feat_repr[pair_id][j];
                }
            }
            
            l2 = (1 - n) * rank;
            for (a = 0; a < rank; a++) vec_word_repr_sum[pair_id][a] *= vec_word_repr[pair_id][l2 + a];
            
            for (int k = 0; k < word_emb_dim; k++) {
                l1 = k * rank;
                for (i = 0; i < rank; i++) {
                    tmp = vec_word_repr_sum[pair_id][i] * emb_model->syn0[l3 + k];
                    vec_params_g[n][l1 + i] += tmp * tmp;
                    vec_emb_map[n][l1 + i] += eta_real / sqrt(vec_params_g[n][l1 + i]) * tmp;
                }
            }
        }
    }
    
    if (update_fea_emb) {
        for (int k = 0; k < p_inst -> fea_num_pairs[pair_id][n]; k++) {
            l1 = p_inst -> fea_vec_pairs[pair_id][n][k] * rank2;
            
            for (int a = 0; a < rank2; a++) vec_feat_repr_sum[pair_id][a] = 0.0;
            for (a = 0; a < rank2; a++) {
                l3 = a * rank;
                for (int j = 0; j < rank; j++) { 
                    vec_feat_repr_sum[pair_id][a] += vec_part_struct_emb[pair_id][l3 + j] * vec_bigram_repr[pair_id][j];
                }
            }
            
            for (a = 0; a < rank2; a++) {
                fea_model -> params_g[a + l1] += vec_feat_repr_sum[pair_id][a] * vec_feat_repr_sum[pair_id][a];
                fea_model -> syn0[a + l1] += eta_real / sqrt(fea_model -> params_g[a + l1]) * vec_feat_repr_sum[pair_id][a];
            }
        }
    }
}

void LrRankingModel::ForwardProp(BaseInstance* b_inst)
{
    PPAInstance* p_inst = (PPAInstance*)b_inst;
    for (int pair_id = 0; pair_id < p_inst -> list_len; pair_id++) {
        ForwardViews(b_inst, pair_id);
    }
    ForwardOutputs(b_inst);
}

void LrRankingModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    PPAInstance* p_inst = (PPAInstance*)b_inst;
    BackPropLabel(b_inst, eta_real);
    for (int pair_id = 0; pair_id < p_inst -> list_len; pair_id++) {
        BackPropViews(b_inst, eta_real, pair_id);
    }
    for (int n = 0; n < 2; n++) {
        for (int k = 0; k < word_emb_dim; k++) {
            int l1 = k * rank;
            for (int i = 0; i < rank; i++) {
                vec_emb_map[n][l1 + i] -= eta_real / sqrt(vec_params_g[n][l1 + i]) * lambda * vec_emb_map[n][l1 + i];
            }
        }
    }
    for (int i = 0; i < rank2; i++) {
        int l2 = i * rank;
        for (int j = 0; j < rank; j++) {
            core_tensor[l2 + j] -= 0.0 * eta_real / sqrt(core_params_g[l2 + j]) * lambda * core_tensor[l2 + j];
        }
    }
}

void LrRankingModel::PrintModelInfo() {
    cout << "dimension of word embedding: " << word_emb_dim << endl;
    cout << "rank: " << rank << endl;
    cout << "Tucker: " << tucker << endl;
    cout << "rank2: " << rank2 << endl;
}
