//
//  LrTensorTuckerModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 15-1-10.
//  Copyright (c) 2015年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrTensorTuckerModel.h"

void LrTensorTuckerModel::InitModel()
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
    
    grad_s = (real*) malloc(num_labels * sizeof(real));
}

void LrTensorTuckerModel::GetRepresentations(BaseInstance *b_inst, int position) {
    int a;
    long long l1,l2,l3;
    l2 = position * rank1;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    
    l1 = p_inst -> word_ids[position] * word_emb_dim;
    //    for (int i = 0; i < rank; i++) {
    //        l3 = i * word_emb_dim;
    //        for (a = 0; a < word_emb_dim; a++) word_repr[i + l2] += emb_model -> syn0[a + l1] * emb_map[l3 + a];
    //    }
    
    l3 = p_inst -> word_ids[position] * word_emb_dim;
    for (int i = 0; i < word_emb_dim; i++) {
        l1 = i * rank1;
        for (a = 0; a < rank1; a++) word_repr[a + l2] += emb_model -> syn0[i + l3] * emb_map[a + l1];
    }
    //    for (a = 0; a < rank; a++) word_repr[a + l2] = 1.0;
    
    l2 = position * rank2;
    for (int i = 0; i < p_inst -> fct_nums_fea[position]; i++) {
        l1 = p_inst -> fct_fea_ids[position][i] * rank2;
        for (a = 0; a < rank2; a++) feat_repr[a + l2] += fea_model -> syn0[a + l1];
    }
}

void LrTensorTuckerModel::ForwardViews(BaseInstance *b_inst) {
    //    long long l1;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    int i;
    for (i = 0; i < max_sent_len * rank1; i++) word_repr[i] = 0.0;
    for (i = 0; i < max_sent_len * rank2; i++) feat_repr[i] = 0.0;
    
    for (int i = 0; i < p_inst->len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            GetRepresentations(b_inst, i);
        }
    }
}

void LrTensorTuckerModel::ForwardStructureEmb(BaseInstance* b_inst) {
    int a;
    long l1, l2, l3;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    for (a = 0; a < rank3; a++) emb_input[a] = 0.0;
    for (a = 0; a < rank1 * rank2; a++) structure_emb[a] = 0.0;
    for (int position = 0; position < p_inst -> len; position++) {
        if (p_inst -> word_ids[position] != -1 && p_inst -> fct_nums_fea[position] != 0) {
            l2 = position * rank1;
            l3 = position * rank2;
            for (int i = 0; i < rank2; i++) {
                l1 = i * rank1;
                for (int j = 0; j < rank1; j++) {
                    structure_emb[l1 + j] += word_repr[j + l2] * feat_repr[i + l3];
                    if (debug && isnan(structure_emb[l1 + j])) {
                        cout << word_repr[j] << endl;
                        cout << feat_repr[i] << endl;    
                    }
                }
            }
        }
    }   
}

void LrTensorTuckerModel::ForwardOutputs(BaseInstance* b_inst)
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

long LrTensorTuckerModel::BackPropLabel(BaseInstance* b_inst, real eta_real) {
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

void LrTensorTuckerModel::BackPropViews(BaseInstance* b_inst, real eta_real) {
    int i, n, a;
    long long l1, l2, l3;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;

    if (debug) {
//        cout << part_struct_emb[0] << endl;
//        cout << part_struct_emb[rank1 * rank2 - 1] << endl;
    }
    
    real tmp;
    
    if (update_word) {
        for (n = 0; n < b_inst -> len; n++) {
            if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                l2 = n * rank2;
                l3 = p_inst -> word_ids[n] * word_emb_dim;
                
                for (int a = 0; a < rank1; a++) word_repr_sum[a] = 0.0;
                for (a = 0; a < rank1; a++) {
                    for (int j = 0; j < rank2; j++) { 
                        word_repr_sum[a] += part_struct_emb[j * rank1 + a] * feat_repr[l2 + j];
                    }
                }
                //update emb_map[k,i]
                for (int k = 0; k < word_emb_dim; k++) {
                    l1 = k * rank1;
                    for (i = 0; i < rank1; i++) {
                        tmp = word_repr_sum[i] * emb_model->syn0[l3 + k];
                        params_g[l1 + i] += tmp * tmp;
                        emb_map[l1 + i] += eta_real / sqrt(params_g[l1 + i]) * tmp;
                    }
                }
            }
        }
    }
    
    if (update_fea_emb) {
        for (n = 0; n < b_inst -> len; n++) {
            if (p_inst -> word_ids[n] != -1 && p_inst -> fct_nums_fea[n] != 0) {
                l2 = n * rank1;
                for (int k = 0; k < p_inst -> fct_nums_fea[n]; k++) {
                    l1 = p_inst -> fct_fea_ids[n][k] * rank2;
                    
                    for (int a = 0; a < rank2; a++) feat_repr_sum[a] = 0.0;
                    for (a = 0; a < rank2; a++) {
                        l3 = a * rank1;
                        for (int j = 0; j < rank1; j++) { 
                            feat_repr_sum[a] += part_struct_emb[l3 + j] * word_repr[l2 + j];
                        }
                    }
                    
                    for (a = 0; a < rank2; a++) {
                        fea_model -> params_g[a + l1] += feat_repr_sum[a] * feat_repr_sum[a];
                        fea_model -> syn0[a + l1] += eta_real / sqrt(fea_model -> params_g[a + l1]) * feat_repr_sum[a];
                    }
                }
            }
        }
    }
}

void LrTensorTuckerModel::ForwardProp(BaseInstance* b_inst)
{
    ForwardViews(b_inst);
    ForwardOutputs(b_inst);
}

void LrTensorTuckerModel::BackProp(BaseInstance* b_inst, real eta_real)
{
    BackPropLabel(b_inst, eta_real);
    BackPropViews(b_inst, eta_real);
}

void LrTensorTuckerModel::PrintModelInfo() {
    cout << "dimension of word embedding: " << word_emb_dim << endl;
    cout << "rank1: " << rank1 << endl;
    cout << "rank2: " << rank2 << endl;
    cout << "rank3: " << rank3 << endl;
}
