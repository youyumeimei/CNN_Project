#include <iostream>
#include "picture.h"
#include "picture.cpp"
#include "matrix.cpp"


int main() {
    Picture picture1=Picture("../bg.jpg");
    picture1.convolution(conv_params[0]);
    auto start1 = std::chrono::steady_clock::now();
    picture1.maxPool(2,2);
    picture1.convolution(conv_params[1]);
    picture1.maxPool(2,2);
    picture1.convolution(conv_params[2]);
    picture1.Flatten(fc_params[0]);
    picture1.FullConnected();
    auto end1 = std::chrono::steady_clock::now();
    cout
            << "CNN calculations took "
            << chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() << "µs ≈ "
            << chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms ≈ "
            << chrono::duration_cast<std::chrono::seconds>(end1 - start1).count() << "s.\n\n";



//    Picture picture2=Picture("../face.jpg");
//    auto start2 = std::chrono::steady_clock::now();
//    picture2.convolution(conv_params[0]);
//    picture2.maxPool(2,2);
//    picture2.convolution(conv_params[1]);
//    picture2.maxPool(2,2);
//    picture2.convolution(conv_params[2]);
//    picture2.Flatten(fc_params[0]);
//    picture2.FullConnected();
//    auto end2 = std::chrono::steady_clock::now();
//    cout
//            << "CNN calculations took "
//            << chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << "µs ≈ "
//            << chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms ≈ "
//            << chrono::duration_cast<std::chrono::seconds>(end2 - start2).count() << "s.\n";

//    float f0 [16]={12,20,30,0,
//                   8,12,2,0,
//                   34,70,37,4,
//                   112,100,25,12};
//    float f1 [25]={12,20,30,0,1,
//                   8,12,2,0,0,
//                   34,70,37,4,5,
//                   112,100,25,12,0,
//                   10,5,9,3,6};
//    float f2 [75]={12,20,30,0,34,
//                   1,8,12,2,55,
//                   0,0,34,70,44,
//                   37,4,5,112,22,
//                   1,2,3,4,5,
//
//                   100,25,12,0,2
//                   ,10,5,9,3,4,
//                   6,7,8,9,10,
//                   11,15,13,14,18,
//                   1,8,10,2,19,
//
//                   22,21,45,17,1,
//                   78,65,23,43,2,
//                   99,52,21,56,3,
//                   54,34,74,33,4,
//                   18,198,19,56,10};
//    float f3 [25]={12,20,30,0,1,8,12,2,0,0,34,70,37,4,5,112,100,25,12,0,10,5,9,3,6};
//
//    Picture picture0=Picture(4,1,f0);//size=4,channel=1
//    Picture picture1=Picture(5,1,f1);//size=5,channel=1
//    Picture picture2=Picture(5,3,f2);//size=5,channel=3
//    Picture picture3=Picture(5,1,f3);//size=5,channel=1


//    测试MaxPooling
//    卷积核的size=2，步长为2
//    picture0.maxPool(2,2);
//    picture1.maxPool(2,2);
//    picture2.maxPool(2,2);
//    cout<<"the maxPooling of picture0 is: "<<endl;
//    for (int i = 0; i < picture0.getSize()*picture0.getSize(); ++i) {
//        cout<<*(picture0.getData()+i)<<" ";
//    }
//    cout<<endl;
//    cout<<"the maxPooling of picture1 is: "<<endl;
//    for (int i = 0; i < picture1.getSize()*picture1.getSize(); ++i) {
//        cout<<*(picture1.getData()+i)<<" ";
//    }
//    cout<<endl;
//    cout<<"the maxPooling of picture2 is: "<<endl;
//    for (int i = 0; i < 3; ++i) {
//        for (int j = 0; j < picture2.getSize(); ++j) {
//            for (int k = 0; k < picture2.getSize(); ++k) {
//                cout<<*(picture2.getData()+i*picture2.getSize()*picture2.getSize()+j*picture2.getSize()+k)<<" ";
//            }
//        }cout<<endl;
//    }

////    测试Flattern
//    float weight[50]={1,0,1,0,1,
//                      0,1,0,1,0,
//                      1,0,1,0,1,
//                      0,1,0,1,0,
//                      1,1,1,1,1,
//
//                      0,0,0,0,0,
//                      1,1,1,1,1,
//                      0,0,0,0,0,
//                      0,1,0,1,0,
//                      1,0,1,0,1};
//    float bias[2]={1,2};
//    fc_param test_fcParam = fc_param{25,2,weight,bias};

//    picture3.Flatten(test_fcParam);
//    cout<<"the flattern is: "<<endl;
//    for (int i = 0; i < picture3.getSize(); ++i) {
//        cout<<*(picture3.getData()+i)<<" ";
//    }
//    cout<<endl;

//    转矩阵
//    float p_data[48]={1,2,3,4,
//                      5,6,7,8,
//                      9,10,11,12,
//                      13,14,15,16,
//
//                      1.1,2.1,3.1,4.1,
//                      5.1,6.1,7.1,8.1,
//                      9.1,10.1,11.1,12.1,
//                      13.1,14.1,15.1,16.1,
//
//                      1.2,2.2,3.2,4.2,
//                      5.2,6.2,7.2,8.2,
//                      9.2,10.2,11.2,12.2,
//                      13.2,14.2,15.2,16.2};
////    (4-2+1*2)/2+1=4/2+1=3
//    Matrix m_data=Matrix(3*3,2*2*3+1);
//    m_data.trans_data(4,1,2,3,2,p_data);
//    cout<<m_data;

//    float p_weight[12]={1,2,3,4,
//                      5,6,7,8,
//                      9,10,11,12};
//    float p_bias[3]={1.1,3.3,2.2};
//    Matrix m_weight=Matrix(2*2+1,3);
//    m_weight.trans_weight(p_weight,p_bias);
//    cout<<m_weight;



    return 0;
}
