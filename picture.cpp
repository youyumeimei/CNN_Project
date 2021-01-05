//
// Created by 黄颖盈 on 2021/1/1.
//

#include "picture.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include "cblas.h"
using namespace cv;
Picture::Picture() {
    size=0;
    channel=0;
    data= nullptr;
    this->count=new atomic_int ;
    *count=1;
}
Picture::Picture(string path) {
    cout<<"The picture's name is "<<path<<endl;
    Mat image = imread(path);
    Mat BGR[3];
    resize(image,image,Size(128,128));
    split(image,BGR);
    this->size=image.rows;
    this->channel=image.channels();
    this->data=new float [size*size*channel];
    int index=0;
    for (int i = 2; i >= 0; i--) {
        for (int j = 0; j < BGR[i].total(); ++j) {
//            string temp = to_string((float)(*(BGR[i].data+j))/255.0f);
            *(data+index)=(float)(*(BGR[i].data+j))/255.0f;
//                    stof(temp.substr(0,9));
//            cout<<*(data+index)<<",";
            index++;
        }
    }
    this->count=new atomic_int ;
    *count=1;
}
Picture::Picture(int size,int channel,const float * data) {
    this->size=size;
    this->channel=channel;
    this->data=new float [size*size*channel]();
    for (int i = 0; i < size*size*channel; ++i) {
        *(this->data+i)=*(data+i);
    }
    this->count=new atomic_int ;
    *this->count=1;
}

Picture::~Picture() {
    if (*count==1){
        delete [] data;
        delete count;
    }else{
        (*count)--;
    }
}
void Picture::convolution(conv_param & convParam) {
    //将weight转化成矩阵；每个元素后面增加一个bias
    Matrix mat_weight=Matrix(convParam.kernel_size*convParam.kernel_size*convParam.in_channels+1,convParam.out_channels);
    mat_weight.trans_weight(convParam.p_weight,convParam.p_bias);
    int new_size=(size-convParam.kernel_size+2*convParam.pad)/convParam.stride+1;
    //将输入的像素转化成矩阵；每个后面增加一个元素1
    Matrix mat_data=Matrix(new_size*new_size,convParam.kernel_size*convParam.kernel_size*convParam.in_channels+1);
    mat_data.trans_data(size,convParam.pad,convParam.stride,channel,convParam.kernel_size,data);
//    Matrix result(convParam.out_channels,new_size*new_size);
    Matrix result=mat_data*mat_weight;
//    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, mat_weight.getColumn(),
//                mat_data.getRow(), mat_weight.getRow(), 1.0, mat_weight.getData(), mat_weight.getColumn(),
//                mat_data.getData(), mat_data.getColumn(), 0.0, result.getData(), mat_data.getRow());

    if ((*count)==1){
        delete [] data;
        delete count;
    } else{
        (*count)--;
    }
    this->data=result.getData();
    this->size=new_size;
    this->channel=convParam.out_channels;
    this->count=result.getCount() ;
    (*count)++;
}
void Picture::maxPool(int k_size, int strides) {
    int outSize;
    if (size%strides==0) outSize=size/strides;
    else  outSize=size/strides+1;

    float * temp=new float [outSize*outSize*channel]();
    int first;
    int index=0;
    float max=0;
    for (int i = 0; i < channel; ++i) {
        for (int j = 0; j <outSize ; j++) {
            for (int k = 0; k < outSize; k++) {
                first = i * size * size + j * size * k_size + k * k_size;
                for (int l = 0; l < k_size; ++l) {
                    for (int m = 0; m < k_size; ++m) {
                        if (j * k_size + l < size && k * k_size + m < size) {
                            if (max < *(data + first + l * size + m))
                                max = *(data + first + l * size + m);
                        }
                    }
                }
                *(temp + index) = max;
                index++;
                max = 0;
            }
        }
    }

    if ((*count)==1){
        delete [] data;
        delete count;
    } else{
        (*count)--;
    }
    this->data=temp;
    this->size=outSize;
    this->count=new atomic_int ;
    *count=1;
}
void Picture::Flatten(const fc_param & fcParam){
    Matrix matrixA= Matrix(fcParam.out_features,fcParam.in_features,fcParam.p_weight);
    Matrix matrixB= Matrix(fcParam.in_features,1,data);
    Matrix matrixC= Matrix(fcParam.out_features,1,fcParam.p_bias);
    if ((*count)==1){
        delete [] data;
        delete count;
    } else{
        (*count)--;
    }
    Matrix result = (matrixA*matrixB+matrixC);
    this->data=result.getData();
    this->size=fcParam.out_features;
    this->count=result.getCount();
    (*count)++;
}
void Picture::FullConnected() {
    double sum=0;
    for (int i = 0; i < size; ++i) {
        sum+=exp(*(data+i));
    }
        cout<<"bg score: "<<exp(*(data))/sum<<endl;
        cout<<"face score: "<<exp(*(data+1))/sum<<endl;
    if(exp(*(data+size-1))/sum>0.90){
        cout<<"你可能是人！"<<endl;
    }else{
        cout<<"你不是人！！！"<<endl;
    }
}
