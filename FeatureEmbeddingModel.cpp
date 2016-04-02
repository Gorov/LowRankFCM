//
//  FeatureEmbeddingModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>

#include <sstream>
#include <algorithm>
#include "FeatureEmbeddingModel.h"

const long long max_w = 500;

int FeatureEmbeddingModel::AddFeature(string feat) {
    int id;
    feat2int::iterator iter = vocabdict.find(feat);
    if (iter != vocabdict.end()) {
        return iter -> second;
    }
    id = (int)vocabdict.size();
    vocabdict[feat] = id;
    cout << feat << "\t" << id << endl;
    
    return id;
}

int FeatureEmbeddingModel::SearchFeature(string feat) {
    feat2int::iterator iter = vocabdict.find(feat);
    if (iter == vocabdict.end()) {
        return -1;
    }
    else return iter -> second;
}

int FeatureEmbeddingModel::InitEmb(int dim)
{
    long long words, a, b, l1;
    
    words = vocabdict.size();
    vocab_size = words;
    
    this -> dim = dim;
    syn0 = (real *)malloc(words * dim * sizeof(real));
    params_g = (real *)malloc(words * dim * sizeof(real));
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * dim * sizeof(float) / 1048576);
        return -1;
    }
    printf("FeaEmb: Allocate memory: %lld * %d\n", words, dim);
    
    for (b = 0; b < words; b++) {
        l1 = b * dim;
        for (a = 0; a < dim; a++) syn0[a + l1] = (rand() / (real)RAND_MAX - 0.5) / dim;
        for (a = 0; a < dim; a++) syn0[a + l1] = 0.0;
        syn0[b + l1] = 1.0;
        for (a = 0; a < dim; a++) params_g[a + l1] = 1.0;
    }
//    for (int a = 0; a < dim; a++) syn0[a] = 1.0;
    for (int i = 0; i < basefeat_list.size(); i++) {
        l1 = basefeat_list[i] * dim;
//        for (int a = 0; a < dim; a++) syn0[a + l1] = 1.0;
//        syn0[i + l1] = 1.0;
    }
    return 0;
}

int FeatureEmbeddingModel::InitEmb()
{
    long long words, a, b, l1;
    
    words = vocabdict.size();
    vocab_size = words;
    
    this -> dim = (int)words;
    syn0 = (real *)malloc(words * words * sizeof(real));
    params_g = (real *)malloc(words * words * sizeof(real));
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * words * sizeof(float) / 1048576);
        return -1;
    }
    printf("FeaEmb: Allocate memory: %lld * %d\n", words, dim);
    
    for (b = 0; b < words; b++) {
        l1 = b * dim;
        for (a = 0; a < dim; a++) syn0[a + l1] = 0.0;
        syn0[b + l1] = 1.0;
        for (a = 0; a < dim; a++) params_g[a + l1] = 1.0;
    }
    return 0;
}

int FeatureEmbeddingModel::InitEmb(char* freqfile, int dim)
{
    long long a, b;
    char line_buf[1000];
    feat2int::iterator iter;
    
    ifstream ifs(freqfile);
    this -> dim = dim;
    this -> vocab_size = vocabdict.size();
    printf("Allocate memory: %ld * %d\n", vocab_size, dim);
    syn0 = (real *)malloc(vocab_size * dim * sizeof(real));
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %ld MB\n", vocab_size * dim * sizeof(float) / 1048576);
        return -1;
    }
    
    for (b = 0; b < vocab_size; b++) {
        ifs.getline(line_buf, 1000, '\n');
        istringstream iss(line_buf);
        for (a = 0; a < dim; a++) iss >> syn0[a + dim * b];
        for (a = 0; a < dim; a++) params_g[a + dim * b] = 1.0;
    }
    ifs.close();
    return 0;
}

int FeatureEmbeddingModel::LoadEmb(char* modelname)
{
    long long words, size, a, b;
    float len;
    char ch;
    FILE *f = fopen(modelname, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    
    dim = (int)size;
    
    words += 42;
    syn0 = (real *)malloc(words * size * sizeof(real));
    params_g = (real *)malloc(words * size * sizeof(real));
    for (a = 0; a < words * size; a++) params_g[a] = 1.0;
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
        return -1;
    }
    
    char tmpword[max_w * 2];
    for (b = 0; b < words - 42; b++) {
        fscanf(f, "%s%c", tmpword, &ch);
        if (feof(f)) {
            break;
        }
        //cout << tmpword << endl;
        feat2int::iterator iter = vocabdict.find(string(tmpword));
        if (iter == vocabdict.end()) {
            vocabdict[string(tmpword)] = (int)b;
        }
        vocablist.push_back(tmpword);
        if ((int)vocablist.size() != b + 1) {
            cout << "here" << endl;
        }
        float tmp;
        for (a = 0; a < size; a++) {
            fread(&tmp, sizeof(float), 1, f);
            syn0[a + b * size] = tmp;
        }
        //for (a = 0; a < size; a++) fread(&syn1neg[a + b * size], sizeof(real), 1, f);
        len = 0.0;
        for (a = 0; a < size; a++) len += syn0[a + b * size] * syn0[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) syn0[a + b * size] /= len;
        for (a = 0; a < size; a++) params_g[a + b * size] = 1.0;
    }
    fclose(f);
    return 0;
}

void FeatureEmbeddingModel::SaveEmb(char* modelfile) {
    int b;
    long long tmp;
    FILE* fileout = fopen(modelfile, "wb");
    fprintf(fileout, "%ld %lld\n", vocab_size, tmp);
    dim = (int) tmp;
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s ", vocablist[i].c_str());
        for (b = 0; b < dim; b++) fwrite(&syn0[i * dim + b], sizeof(real), 1, fileout);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

