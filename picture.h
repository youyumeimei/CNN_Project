//
// Created by 黄颖盈 on 2021/1/1.
//

#ifndef CNN_FINAL_PICTURE_H
#define CNN_FINAL_PICTURE_H

#include <iostream>
#include "matrix.h"
#include "face_binary_cls.cpp"

using namespace std;

class Picture {
    int size;
    int channel;
    float * data;
    atomic_int * count;
public:
    Picture();
    Picture(string path);
    Picture(int size,int channel,const float * data);
    ~Picture();
    void convolution(conv_param & convParam);
    void maxPool(int k_size,int strides);
    void Flatten(const fc_param & fcParam);
    void FullConnected();

    int getSize() const{return size;}
    float * getData(){return data;}
};


#endif //CNN_FINAL_PICTURE_H
