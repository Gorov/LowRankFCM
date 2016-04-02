//
//  LabelEmbeddingModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-15.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>

#include <sstream>
#include <algorithm>
#include "LabelEmbeddingModel.h"

const long long max_w = 500;

int LabelEmbeddingModel::AddLabel(string feat) {
    int id;
    label2int::iterator iter = vocabdict.find(feat);
    if (iter != vocabdict.end()) {
        return iter -> second;
    }
    id = (int)vocabdict.size();
    vocabdict[feat] = id;
    cout << feat << "\t" << id << endl;
    return id;
}

int LabelEmbeddingModel::SearchLabel(string feat) {
    label2int::iterator iter = vocabdict.find(feat);
    if (iter == vocabdict.end()) {
        return -1;
    }
    else return iter -> second;
}

int LabelEmbeddingModel::InitEmb(int dim)
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
//        for (a = 0; a < dim; a++) syn0[a + l1] = (rand() / (real)RAND_MAX - 0.5) * 0.1 / dim;
        for (a = 0; a < dim; a++) syn0[a + l1] = 0.0;
        syn0[b + l1] = 1.0;
        for (a = 0; a < dim; a++) params_g[a + l1] = 1.0;
    }
    return 0;
}

int LabelEmbeddingModel::InitEmb()
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

int LabelEmbeddingModel::InitEmb(char* freqfile, int dim)
{
    long long words, a, b;
    char line_buf[1000];
    string word;
    
    words = 0;
    label2int::iterator iter;
    ifstream ifs(freqfile);
    ifs.getline(line_buf, 1000, '\n');
    while (strcmp(line_buf, "") != 0) {
        words ++;
        ifs.getline(line_buf, 1000, '\n');    
    }
    ifs.close();
    
    this -> dim = dim;
    syn0 = (real *)malloc(words * dim * sizeof(real));
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * dim * sizeof(float) / 1048576);
        return -1;
    }
    
    ifs.open(freqfile);
    for (b = 0; b < words; b++) {
        ifs.getline(line_buf, 1000, '\n');
        istringstream iss(line_buf);
        iss >> word;
        vocabdict[word] = (int)b;
        vocablist.push_back(word);
        for (a = 0; a < dim; a++) syn0[a + dim * b] = (rand() / (real)RAND_MAX - 0.5) / dim;
        for (a = 0; a < dim; a++) params_g[a + dim * b] = 1.0;
    }
    ifs.close();
    return 0;
}

int LabelEmbeddingModel::LoadEmb(char* modelname)
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
        label2int::iterator iter = vocabdict.find(string(tmpword));
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

void LabelEmbeddingModel::SaveEmb(char* modelfile) {
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
