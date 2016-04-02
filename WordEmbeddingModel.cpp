//
//  WordEmbeddingModel.cpp
//  LR_FCEM_proj
//
//  Created by gflfof gflfof on 14-11-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include <sstream>
#include <algorithm>
#include "WordEmbeddingModel.h"

const long long max_w = 500;

bool pairCompare(pair<int, string> a, pair<int, string> b);

bool pairCompare(pair<int, string> a, pair<int, string> b) {
    return a.first > b.first;
}

int WordEmbeddingModel::InitEmb(char* freqfile, int dim)
{
    long long words, a, b;
    char line_buf[1000];
    string word;
    
    words = 0;
    word2int::iterator iter;
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

int WordEmbeddingModel::LoadEmb(char* modelname)
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
        word2int::iterator iter = vocabdict.find(string(tmpword));
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
        //for (a = 0; a < size; a++) syn0[a + b * size] /= len;
        for (a = 0; a < size; a++) params_g[a + size * b] = 1.0;
    }
    for (b = 0; b < 42; b++) {
        string key = "SST:" + sstags[b];
        word2int::iterator iter = vocabdict.find(key);
        if (iter == vocabdict.end()) {
            vocabdict[key] = (int)(words - 42 + b);
        }
        vocablist.push_back(key);
        //word2int::iterator iter2 = vocabdict.find(sstags[b]);
        //if (iter2 == vocabdict.end()) {
        for (a = 0; a < size; a++) syn0[a + (words - 42 + b) * size] = (rand() / (real)RAND_MAX - 0.5) / dim;
        for (a = 0; a < size; a++) params_g[a + (words - 42 + b) * size] = 1.0;
        //}
        //else {
        //    for (a = 0; a < size; a++) syn0[a + (words - 42 + b) * size] = syn0[a + iter2 -> second * size];
        //}
    }
    fclose(f);
    return 0;
}

int WordEmbeddingModel::LoadEmbUnnorm(char* modelname)
{
    long long words, size, a, b;
    char ch;
    FILE *f = fopen(modelname, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    dim = (int)size;
    //syn1neg = (float *)malloc(words * size * sizeof(float));
    syn0 = (real *)malloc(words * size * sizeof(real));
    
    if (syn0 == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
        return -1;
    }
    
    char tmpword[max_w * 2];
    for (b = 0; b < words; b++) {
        fscanf(f, "%s%c", tmpword, &ch);
        if (feof(f)) {
            break;
        }
        //cout << tmpword << endl;
        word2int::iterator iter = vocabdict.find(string(tmpword));
        if (iter == vocabdict.end()) {
            vocabdict[string(tmpword)] = (int)b;
        }
        vocablist.push_back(tmpword);
        float tmp;
        for (a = 0; a < size; a++) {
            fread(&tmp, sizeof(float), 1, f);
            syn0[a + b * size] = tmp;
        }
    }
    fclose(f);
    return 0;
}

void WordEmbeddingModel::SaveEmb(char* modelfile) {
    int b;
    long long tmp;
    FILE* fileout = fopen(modelfile, "wb");
    fprintf(fileout, "%ld %lld\n", vocab_size, tmp);
    dim = (int) tmp;
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s ", vocablist[i].c_str());
        //fwrite(&syn1neg[i * layer1_size], sizeof(real), layer1_size, fileout);
        for (b = 0; b < dim; b++) fwrite(&syn0[i * dim + b], sizeof(real), 1, fileout);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

void WordEmbeddingModel::SaveEmbTxt(char* modelfile, real alpha) {
    int b;
    FILE* fileout = fopen(modelfile, "w");
    for (int i = 0; i < vocab_size; i++) {
        fprintf(fileout, "%s", vocablist[i].c_str());
        for (b = 0; b < dim; b++) fprintf(fileout, "\t%f", alpha * syn0[i * dim + b]);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}
