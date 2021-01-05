//
// Created by 黄颖盈 on 2021/1/1.
//

#include "matrix.h"
#include <immintrin.h>

float dot_product1(const float *p1, const float * p2, size_t n,int m);
float dot_product2(const float *p1, const float * p2, size_t n,int m);
float dot_product3(const float *p1, const float * p2, size_t n,int m);

Matrix::Matrix(){
    row=0;
    column=0;
    data=nullptr;
    this->count=new atomic_int ;
    *count=1;
}
Matrix::Matrix(int row,int column, float * data) {
    this->row = row;
    this->column = column;
    this->data=new float[row*column];
    for (int i = 0; i < row*column; ++i) {
        *(this->data+i)=*(data+i);
    }
    this->count=new atomic_int ;
    *this->count=1;
}
Matrix::Matrix(int row,int column) {
    this->row = row;
    this->column = column;
    this->data=new float[row*column]();
    this->count=new atomic_int ;
    *this->count=1;
}
Matrix::Matrix(const Matrix& matrix){
    this->row=matrix.row;
    this->column=matrix.column;
    this->data=matrix.data;
    this->count=matrix.count;
    (*this->count) +=1;
}

Matrix::~Matrix(){
    if (*count==1){
        delete [] data;
        delete count;
    }else{
        (*count) -=1;
    }
}

Matrix Matrix::trans_weight(const float *weight,const float * bias) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            if (i!=row-1){ *(data+i*column+j)=*(weight+j*(row-1)+i);}
            else {*(data+i*column+j)=*(bias+j);}
        }
    }
    return *this;
}
Matrix Matrix::trans_data(int size,int padding,int stride,int channel,int k_size,const float * data) {
    int index=0;
    for (int m = 0; m +k_size<= size+padding*2; m+=stride) {
        for (int n = 0; n+k_size <= size+padding*2; n+=stride) {
            for (int i = 0; i < channel; ++i) {
                for (int j = 0; j < k_size; j++) {
                    for (int k = 0; k < k_size; k++) {
                        if (m+j >= padding && m+j < size + padding && n+k >= padding && n+k < size + padding) {
                            *(this->data + index) = *(data + i *size*size+ (m+j - padding) * size + n+k - padding);
                            index++;
                        } else{
                            index++;
                        }
                    }
                }
            }
            *(this->data+index)=1;
            index++;
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &matrix) {
    float *Data = new float[matrix.row * matrix.column];

    for (int i = 0; i < row * column; ++i) {
        *(Data + i) = *(data + i) + *(matrix.data + i);
    }
    return Matrix(row, column, Data);
}

Matrix Matrix::operator*(Matrix& matrix) const{
    float *Data =  new float[row * column]();
    int index=0;
    if (column == matrix.row) {
        for (int j = 0; j < matrix.column; ++j) {
            for (int i = 0; i < row; ++i) {
                float result=dot_product1(data + i * column, matrix.data + j, column,matrix.column);
                if (result>0){
                    *(Data + index) = result;
                } else{
                    *(Data + index) = 0;
                }
                index++;
            }
        }
    } else{
        cout<<"Your two matrices don't match"<<endl;
        return Matrix();
    }
    return Matrix(row,matrix.column,Data);
}

ostream & operator<<(ostream& os,const Matrix& matrix){
    os.setf(ios_base::fixed, ios_base::floatfield);
    os.precision(3);

    for (int i = 0; i < matrix.row; ++i) {
        for (int j = 0; j < matrix.column; ++j) {
            os<<*(matrix.data+j+i*matrix.column)<<" ";
        }
        os<<endl;
    }
    return os;
}

float dot_product1(const float *p1, const float * p2, size_t n,int m)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += (p1[i] * p2[i*m]);
    return sum;
}

//可以不用是8的倍数
float dot_product2(const float *p1, const float * p2, size_t n,int m)
{
    float sum = 0.0f;
    for (size_t i = 0; i+8 < n; i+=8)
    {
        sum += (p1[i] * p2[i*m]);
        sum += (p1[i+1] * p2[(i+1)*m]);
        sum += (p1[i+2] * p2[(i+2)*m]);
        sum += (p1[i+3] * p2[(i+3)*m]);
        sum += (p1[i+4] * p2[(i+4)*m]);
        sum += (p1[i+5] * p2[(i+5)*m]);
        sum += (p1[i+6] * p2[(i+6)*m]);
        sum += (p1[i+7] * p2[(i+7)*m]);
    }
    for (int i = n-n%8; i <n ; ++i) {
        sum+=(p1[i])*(p2[i*m]);
    }
    return sum;
}

//float dot_product3(const float *p1, const float * p2, size_t n) {
//    if(n % 8 != 0)
//    {
//        std::cerr << "The size n must be a multiple of 8." <<std::endl;
//        return 0.0f;
//    }
//
//    float sum[8] = {0};
//    __m256 a, b;
//    __m256 c = _mm256_setzero_ps();
//
//    float * array=new float [8];
//
//    for (size_t i = 0; i < n; i+=8)
//    {
//        for (int j = 0; j < 8; ++j) {
//            array[j]=*(p2+i*n+j*n);
//        }
//
//        a = _mm256_load_ps(p1 + i);
//        b = _mm256_load_ps(array);
//        c =  _mm256_add_ps(c, _mm256_mul_ps(a, b));
//    }
//
//    _mm256_store_ps(sum, c);
//    return (sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7]);
//}