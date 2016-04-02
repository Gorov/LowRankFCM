//
//  LrFcemModelLabel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrFcemModelLabel.h"

void LrFcemModelLabel::InitModel()
{
    label_emb = (real*) malloc(num_labels * word_emb_dim * feat_emb_dim * sizeof(real));
    for (int a = 0; a < num_labels * word_emb_dim * feat_emb_dim; a++) label_emb[a] = 0.0;// (rand() / (real)RAND_MAX - 0.5) / layer1_size;// 0.0;
    params_g = (real*) malloc(num_labels * word_emb_dim * feat_emb_dim * sizeof(real));
    for (int a = 0; a < num_labels * word_emb_dim * feat_emb_dim; a++) params_g[a] = 1.0;
    
    emb_p = (real*) malloc(word_emb_dim * feat_emb_dim * sizeof(real));
    part_emb_p = (real*) malloc(word_emb_dim * feat_emb_dim * sizeof(real));
    word_emb_sum = (real*) malloc(word_emb_dim * sizeof(real));
    feat_emb_sum = (real*) malloc(feat_emb_dim * sizeof(real));
    word_grad_sum = (real*) malloc(word_emb_dim * sizeof(real));
    feat_grad_sum = (real*) malloc(feat_emb_dim * sizeof(real));
}

void LrFcemModelLabel::OuterProd(vector<int>& words, int num_words, vector<int>& feats, int num_feats) {
    int a;
    long long l1;
    for (int i = 0; i < word_emb_dim; i++) word_emb_sum[i] = 0.0;
    for (int i = 0; i < feat_emb_dim; i++) feat_emb_sum[i] = 0.0;
    
    for (int i = 0; i < num_words; i++) {
        l1 = words[i] * word_emb_dim;
        for (a = 0; a < word_emb_dim; a++) word_emb_sum[a] += emb_model -> syn0[a + l1];
    }
    for (int i = 0; i < num_feats; i++) {
        l1 = feats[i] * feat_emb_dim;
        for (a = 0; a < feat_emb_dim; a++) feat_emb_sum[a] += fea_model -> syn0[a + l1];
    }
    for (int i = 0; i < feat_emb_dim; i++) {
        l1 = i * word_emb_dim;
        for (int j = 0; j < word_emb_dim; j++) {
            emb_p[l1 + j] += word_emb_sum[j] * feat_emb_sum[i];
            if (isnan(emb_p[l1 + j])) {
                cout << word_emb_sum[j] << endl;
                cout << feat_emb_sum[i] << endl;    
            }
        }
    }
}

void LrFcemModelLabel::ForwardOuterProd(BaseInstance* b_inst)
{
    for (int a = 0; a < feat_emb_dim * word_emb_dim; a++) emb_p[a] = 0.0;
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    vector<int> words;
    words.resize(1);
    for (int i = 0; i < p_inst->len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            words[0] = p_inst -> word_ids[i];
            OuterProd(words, 1, p_inst -> fct_fea_ids[i], p_inst -> fct_nums_fea[i]);
        }
    }
}

void LrFcemModelLabel::ForwardOutputs(BaseInstance* b_inst)
{
    int c;
    long long l1, l2;
    real sum;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = 0; c < num_labels; c++) {
        l1 = c * feat_emb_dim * word_emb_dim;
        sum = 0;
        for (int i = 0; i < feat_emb_dim; i++) {
            l2 = i * word_emb_dim;
            for (int j = 0; j < word_emb_dim; j++) {
                sum += emb_p[l2 + j] * label_emb[l1 + l2 + j];
            }
        }
        b_inst -> scores[c] += sum;
    }
}

long LrFcemModelLabel::BackPropOuterProd(BaseInstance* b_inst, real eta_real) {
    int c, y;
    long long l1, l2;
    for (int a = 0; a < feat_emb_dim * word_emb_dim; a++) part_emb_p[a] = 0.0;
    for (c = 0; c < num_labels; c++) {
        l1 = c * feat_emb_dim * word_emb_dim;
        if (b_inst -> label_id == c) y = 1;
        else y = 0;
        for (int i = 0; i < feat_emb_dim; i++) {
            l2 = i * word_emb_dim;
            for (int j = 0; j < word_emb_dim; j++) {
                part_emb_p[l2 + j] += (y - b_inst -> scores[c]) * label_emb[l1 + l2 + j];
                if (!adagrad) {
                    label_emb[l1 + l2 + j] += eta_real * (y - b_inst -> scores[c]) * emb_p[l2 + j];
                }
                else {
                    params_g[l1 + l2 + j] += (y - b_inst -> scores[c]) * emb_p[l2 + j] * (y - b_inst -> scores[c]) * emb_p[l2 + j];
                    label_emb[l1 + l2 + j] += eta_real / params_g[l1 + l2 + j] * (y - b_inst -> scores[c]) * emb_p[l2 + j];
                    //                    if (isnan(label_emb[l1 + l2 + j]))
                    //                    cout << label_emb[l1 + l2 + j] << endl;
                }
            }
        }
    }
    //    for (int i = 0; i < feat_emb_dim; i++) {
    //        l2 = i * word_emb_dim;
    //        cout << i << ":" << label_emb[l2] << endl;
    //    }
    return 0;
}

void LrFcemModelLabel::BackPropViews(vector<int>& words, int num_words, vector<int>& feats, int num_feats, real eta_real) {
    int a;
    long long l1, l2;
    for (int i = 0; i < word_emb_dim; i++) word_emb_sum[i] = 0.0;
    for (int i = 0; i < feat_emb_dim; i++) feat_emb_sum[i] = 0.0;
    
    //    for (int i = 0; i < num_feats; i++) {
    //        l1 = feats[i] * feat_emb_dim;
    //        for (a = 0; a < feat_emb_dim; a++) feat_emb_sum[a] += fea_model -> syn0[a + l1];
    //    }
    
    if (update_fea_emb) {
        for (int i = 0; i < num_words; i++) {
            l1 = words[i] * word_emb_dim;
            for (a = 0; a < word_emb_dim; a++) word_emb_sum[a] += emb_model -> syn0[a + l1];
        }
        if (!adagrad) {
            for (int i = 0; i < num_feats; i++) {
                l1 = feats[i] * feat_emb_dim;
                for (a = 0; a < feat_emb_dim; a++) {
                    l2 = a * word_emb_dim;
                    for (int j = 0; j < word_emb_dim; j++) { 
                        fea_model -> syn0[l1 + a] += eta_real * part_emb_p[l2 + j] * word_emb_sum[j];
                    }
                }
            }
        }
        else {
            for (int i = 0; i < num_feats; i++) {
                l1 = feats[i] * feat_emb_dim;
                for (int a = 0; a < feat_emb_dim; a++) feat_grad_sum[a] = 0.0;
                for (a = 0; a < feat_emb_dim; a++) {
                    l2 = a * word_emb_dim;
                    for (int j = 0; j < word_emb_dim; j++) { 
                        feat_grad_sum[a] += part_emb_p[l2 + j] * word_emb_sum[j];
                    }
                }
                for (a = 0; a < feat_emb_dim; a++) {
                    fea_model -> params_g[a + l1] += feat_grad_sum[a] * feat_grad_sum[a];
                    fea_model -> syn0[a + l1] += eta_real / sqrt(fea_model -> params_g[a + l1]) * feat_grad_sum[a];
                    if (isnan(fea_model -> syn0[a + l1])) {
                        cout << num_feats << ":" << endl;
                        for (int k = 0; k < feat_emb_dim; k++) {
                            cout << fea_model -> params_g[a + l1] << endl;
                            cout << feat_grad_sum[k] << endl;
                        }
                        cout << endl;
                        exit(-1);
                    }
                }
            }
        }
    }
    
    if (update_emb) {
        //todo
    }
}

long LrFcemModelLabel::BackPropPhrase(BaseInstance* b_inst, real eta_real) {
    LrFcemInstance* p_inst = (LrFcemInstance*) b_inst;
    vector<int> words;
    words.resize(1);
    for (int i = 0; i < p_inst->len; i++) {
        if (p_inst -> word_ids[i] != -1 && p_inst -> fct_nums_fea[i] != 0) {
            words[0] = p_inst -> word_ids[i];
            BackPropViews(words, 1, p_inst -> fct_fea_ids[i], p_inst -> fct_nums_fea[i], eta_real);
        }
    }
    return 0;
}
//
//void LrFcemModel::BackPropFea(BaseInstance* b_inst, int word_id, int c, int y, real eta_real) {
//    long a;
//    long l1 = c * layer1_size;
//    long l2 = word_id * layer1_size;
//    if (!adagrad) for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real * (y - b_inst->scores[c]) * emb_model->syn0[a + l2];
//    else {
//        for (a = 0; a < layer1_size; a++) params_g[a + l1] += (y - b_inst->scores[c]) * emb_model->syn0[a + l2] * (y - b_inst->scores[c]) * emb_model->syn0[a + l2];
//        for (a = 0; a < layer1_size; a++) label_emb[a + l1] += eta_real / sqrt(params_g[a + l1]) * ( (y - b_inst->scores[c]) * emb_model->syn0[a + l2] );
//    }
//}

void LrFcemModelLabel::ForwardProp(BaseInstance* b_inst)
{
    ForwardOuterProd(b_inst);
    ForwardOutputs(b_inst);
}

void LrFcemModelLabel::BackProp(BaseInstance* b_inst, real eta_real)
{
    //    int a;
    //    long long l1, l2;
    
    BackPropOuterProd(b_inst, eta_real);
    BackPropPhrase(b_inst, eta_real);
    
    //    if (update_emb) {
    //        for (int i = 0; i < inst -> count; i++) {
    //            if (inst -> word_ids[i] >= 0) {
    //                l1 = inst -> word_ids[i] * layer1_size;
    //                l2 = layer1_size * i;
    //                if (!adagrad) for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta_real * part_emb_p[a + l2];
    //                else {
    //                    for (a = 0; a < layer1_size; a++) emb_model->params_g[a + l1] += part_emb_p[a + l2] * part_emb_p[a + l2];
    //                    for (a = 0; a < layer1_size; a++) emb_model->syn0[a + l1] += eta_real / sqrt(emb_model->params_g[a + l1]) * part_emb_p[a + l2];
    //                }
    //            }
    //        }
    //    }
}

void LrFcemModelLabel::PrintModelInfo() {
    cout << word_emb_dim;
    cout << feat_emb_dim;
}
