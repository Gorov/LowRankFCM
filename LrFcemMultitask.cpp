//
//  LrFcemMultitask.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-29.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "LrFcemMultitask.h"

void LrFcemMultitask::ForwardOutputs(BaseInstance* b_inst, int task_id)
{
    int c;
    long long l1, l2;
    real sum;
    //real norm1 = 1.0, norm2 = 1.0;
    word2int::iterator iter;
    for (c = start[task_id]; c < start[task_id] + length[task_id]; c++) {
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

long LrFcemMultitask::BackPropOuterProd(BaseInstance* b_inst, real eta_real, int task_id) {
    int c, y;
    long long l1, l2;
    for (int a = 0; a < feat_emb_dim * word_emb_dim; a++) part_emb_p[a] = 0.0;
    for (c = start[task_id]; c < start[task_id] + length[task_id]; c++) {
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
                    if (isnan(label_emb[l1 + l2 + j]))
                        cout << label_emb[l1 + l2 + j] << endl;
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

void LrFcemMultitask::ForwardProp(BaseInstance* b_inst, int task_id)
{
    ForwardOuterProd(b_inst);
    ForwardOutputs(b_inst, task_id);
}

void LrFcemMultitask::BackProp(BaseInstance* b_inst, real eta_real, int task_id)
{
    BackPropOuterProd(b_inst, eta_real, task_id);
    BackPropPhrase(b_inst, eta_real);
}

