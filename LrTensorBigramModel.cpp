//
//  LrTensorBigramModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-24.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "PrepInstance.h"
#include "LrTensorBigramModel.h"

void LrTensorBigramModel::InitModel()
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
    
    core_tensor = (real*) malloc(rank1 * rank1 * rank3 * sizeof(real));
    for (int a = 0; a < rank1 * rank1 * rank3; a++) core_tensor[a] = 0.0;
    printf("Core Tensor: Allocate memory: %d * %d * %d\n", rank1, rank1, rank3);
    core_params_g = (real*) malloc(rank1 * rank1 * rank3 * sizeof(real));
    for (int a = 0; a < rank1 * rank1 * rank3; a++) core_params_g[a] = 1.0;
    
    rank_views.push_back(rank1);
    real* word_repr = (real*) malloc(max_sent_len * rank1 * sizeof(real));
    word_repr_views.push_back(word_repr);
    rank_views.push_back(rank1);
    word_repr = (real*) malloc(max_sent_len * rank1 * sizeof(real));
    word_repr_views.push_back(word_repr);
    real* word_repr_sum = (real*) malloc(rank1 * sizeof(real));
    word_repr_sum_views.push_back(word_repr_sum);
    word_repr_sum = (real*) malloc(rank1 * sizeof(real));
    word_repr_sum_views.push_back(word_repr_sum);
    
    emb_input = (real*) malloc(rank3 * sizeof(real));
    part_emb_input = (real*) malloc(rank3 * sizeof(real));
    
    structure_emb = (real*) malloc(rank1 * rank1 * sizeof(real));
    part_struct_emb = (real*) malloc(rank1 * rank1 * sizeof(real));
    
    grad_s = (real*) malloc(num_labels * sizeof(real));
}

void LrTensorBigramModel::GetRepresentations(BaseInstance *b_inst, int position) {
    int a;
    long long l1,l2,l3;
    l2 = position * rank1;
    PrepBigramInstance* p_inst = (PrepBigramInstance*) b_inst;
    
//    l1 = p_inst -> word_ids[position] * word_emb_dim;
    
    for (int v = 0; v < word_repr_views.size(); v++) {
        if (v == 0)
            l3 = p_inst -> word_pairs[position].first * word_emb_dim;
        else
            l3 = p_inst -> word_pairs[position].second * word_emb_dim;
        for (int i = 0; i < word_emb_dim; i++) {
            l1 = i * rank_views[v];
            for (a = 0; a < rank_views[v]; a++) word_repr_views[v][a + l2] += emb_model -> syn0[i + l3] * emb_map[a + l1];
        }
    }
}

void LrTensorBigramModel::ForwardViews(BaseInstance *b_inst) {
    //    long long l1;
    PrepBigramInstance* p_inst = (PrepBigramInstance*) b_inst;
    int i, v;
    for (v = 0; v < rank_views.size(); v++)
        for (i = 0; i < max_sent_len * rank_views[v]; i++) word_repr_views[v][i] = 0.0;
    
    for (int i = 0; i < p_inst->num_pairs; i++) {
        if (p_inst -> word_pairs[i].first != -1 && p_inst -> word_pairs[i].second != -1) {
            GetRepresentations(b_inst, i);
        }
    }
}

void LrTensorBigramModel::ForwardStructureEmb(BaseInstance* b_inst) {
    int a;
    long l1, l2, l3;
    PrepBigramInstance* p_inst = (PrepBigramInstance*) b_inst;
    for (a = 0; a < rank3; a++) emb_input[a] = 0.0;
    for (a = 0; a < rank_views[0] * rank_views[1]; a++) structure_emb[a] = 0.0;
    for (int position = 0; position < p_inst -> num_pairs; position++) {
        if (p_inst -> word_pairs[position].second == 0) {
            cout << "here" << endl;
        }
        if (p_inst -> word_pairs[position].first != -1 && p_inst -> word_pairs[position].second != -1) {
            l2 = position * rank_views[0];
            l3 = position * rank_views[1];
            for (int i = 0; i < rank_views[1]; i++) {
                l1 = i * rank_views[0];
                for (int j = 0; j < rank_views[0]; j++) {
                    structure_emb[l1 + j] += word_repr_views[0][j + l2] * word_repr_views[1][i + l3];
                    if (debug && isnan(structure_emb[l1 + j])) {
                        cout << word_repr_views[0][j] << endl;
                        cout << word_repr_views[1][i] << endl;    
                    }
                }
            }
        }
    }   
}

void LrTensorBigramModel::ForwardOutputs(BaseInstance* b_inst)
{
    int a, c;
    long long l1, l2;
    real sum;
    
    ForwardStructureEmb(b_inst);
    for (c = 0; c < rank3; c++) emb_input[c] = 0.0;
    
    for (c = 0; c < rank3; c++) {
        l1 = c * rank_views[0] * rank_views[1];
        sum = 0;
        for (int i = 0; i < rank_views[1]; i++) {
            l2 = i * rank_views[0];
            for (int j = 0; j < rank_views[0]; j++) {
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
    for (c = 0; c < num_labels; c++) {
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

long LrTensorBigramModel::BackPropLabel(BaseInstance* b_inst, real eta_real) {
    int c, y;
    long long l1, l2;
    
    for (int a = 0; a < rank3; a++) part_emb_input[a] = 0.0;
    for (c = 0; c < num_labels; c++) {
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
    
    for (int a = 0; a < rank_views[1] * rank_views[0]; a++) part_struct_emb[a] = 0.0;
    for (c = 0; c < rank3; c++) {
        l1 = c * rank_views[0] * rank_views[1];
        for (int i = 0; i < rank_views[1]; i++) {
            l2 = i * rank_views[0];
            for (int j = 0; j < rank_views[0]; j++) {
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

void LrTensorBigramModel::BackPropViews(BaseInstance* b_inst, real eta_real) {
//    int i, j, n, a;
//    long long l1, l2, l3;
//    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    if (debug) {
        //        cout << part_struct_emb[0] << endl;
        //        cout << part_struct_emb[rank1 * rank2 - 1] << endl;
    }
    
//    real tmp;
    
    if (update_word) {
        //not supported yet
    }
}

void LrTensorBigramModel::ForwardProp(BaseInstance* b_inst)
{
    ForwardViews(b_inst);
    ForwardOutputs(b_inst);
}

void LrTensorBigramModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    BackPropLabel(b_inst, eta_real);
    BackPropViews(b_inst, eta_real);
}

void LrTensorBigramModel::PrintModelInfo() {
    cout << "dimension of word embedding: " << word_emb_dim << endl;
    cout << "rank1: " << rank1 << endl;
//    cout << "rank2: " << rank2 << endl;
    cout << "rank3: " << rank3 << endl;
}
