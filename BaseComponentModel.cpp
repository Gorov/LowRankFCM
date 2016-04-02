//
//  BaseComponentModel.cpp
//  RE_FCT
//
//  Created by gflfof gflfof on 14-8-30.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <iomanip> 
#include "BaseComponentModel.h"

void BaseComponentModel::Init() {
    eta = eta0 = 0.01;
    iter = 1;
    
    word_emb_dim = emb_model -> dim;
    feat_emb_dim = fea_model -> dim;
    update_emb = false;
}
