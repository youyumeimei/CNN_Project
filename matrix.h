//
// Created by 黄颖盈 on 2021/1/1.
//

#ifndef CNN_FINAL_MATRIX_H
#define CNN_FINAL_MATRIX_H
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>

using namespace std;
class Matrix{
private:
    int row;
    int column;
    float * data;
    atomic_int * count;
public:
    Matrix();
    Matrix(int row,int column, float * data);
    Matrix(int row,int column);
    Matrix(const Matrix& matrix);
    ~Matrix();
    int getColumn() const {return column;}
    int getRow() const {return row;}
    float * getData() const {return data;}
    atomic_int * getCount() const {return count;}

    Matrix trans_weight(const float *weight,const float * bias);
    Matrix trans_data(int size,int padding,int stride,int channel,int k_size,const float * data);

    Matrix operator+(const Matrix & matrix);
    Matrix operator*(Matrix& matrix) const;

    friend ostream & operator<<(ostream& os,const Matrix& matrix);

};


#endif //CNN_FINAL_MATRIX_H
