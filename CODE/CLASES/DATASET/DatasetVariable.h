/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.h to edit this template
 */

/* 
 * File:   DatasetVariable.h
 * Author: piero
 *
 * Created on 29 de noviembre de 2023, 14:05
 */

#ifndef DATASETVARIABLE_H
#define DATASETVARIABLE_H

#include "Dataset.h"

template <typename X, typename Y>
class DatasetVariable: public Dataset<X, Y> {
public:
    DatasetVariable(); //Set everythin in null
    DatasetVariable(int); //Set everything with a size
    DatasetVariable(const X *, const Y *, int); //Copy the data from an array
    DatasetVariable(const DatasetVariable&);
    virtual ~DatasetVariable();
    void scale();
    void split(double);
    int getSize();
private:
    X *x;
    Y *y;
    X *x_predicted;
    Y *y_predicted;
    X maxX;
    Y maxY;
    int n;
    int n_predicted;
    /**************************************************************************/
    double forwardIntern(double (*)(double, double), int, double **, double **, 
                         double *, int, int, double);
    double getError(int);
    double getData(int, int);
    int getSizeTest();
    double forwardTest(double (*)(double, double), int, double **, double **, 
                       double *, int, int, double);
protected:
    void scale(int);
/**************************FRIENDS*****************************************/
template <typename A, typename B>
friend class Dataset;
template <typename A, typename B>
friend class NeuralNetwork;
};
/******************************************************************************/
template <typename X, typename Y>
DatasetVariable<X, Y>::DatasetVariable(){
    x = nullptr;
    y = nullptr;
    x_predicted = nullptr;
    y_predicted = nullptr;
    n = 0;
    n_predicted = 0;
}
template <typename X, typename Y>
DatasetVariable<X, Y>::DatasetVariable(int n){
    x_predicted = nullptr;
    y_predicted = nullptr;
    x = new X[n]{};
    y = new Y[n]{};
    this->n = n;
    n_predicted = 0;
}
template <typename X, typename Y> //addX returns a copy of the data
DatasetVariable<X, Y>::DatasetVariable(const X *arr1, const Y *arr2, int n){
    x_predicted = nullptr;
    y_predicted = nullptr;
    x = new X[n]{};
    y = new Y[n]{};
    this->n = n;
    for(int i=0; i < n; i++){
        x[i] = arr1[i];
        y[i] = arr2[i];
    }
    n_predicted = 0;
}
template <typename X, typename Y>
DatasetVariable<X, Y>::DatasetVariable(const DatasetVariable& orig){
    n = orig.n;
    n_predicted = orig.n_predicted;
    maxX = orig.maxX;
    maxY = orig.maxY;
    if(orig.x == nullptr) x = nullptr;
    else{
        x = new X[orig.n]{};
        for(int i=0; i < n; i++) x[i] = orig.x[i];
    }
    if(orig.y == nullptr) y = nullptr;
    else{
        y = new Y[orig.n]{};
        for(int i=0; i < n; i++) y[i] = orig.y[i];
    }
    if(orig.x_predicted == nullptr) x_predicted = nullptr;
    else{
        x_predicted = new X[orig.n_predicted]{};
        for(int i=0; i < n_predicted; i++) x_predicted[i] = orig.x_predicted[i];
    }
    if(orig.y_predicted == nullptr) y_predicted = nullptr;
    else{
        y_predicted = new Y[orig.n_predicted]{};
        for(int i=0; i < n_predicted; i++) y_predicted[i] = orig.y_predicted[i];
    }
}
template <typename X, typename Y>
DatasetVariable<X, Y>::~DatasetVariable(){
    if(x != nullptr) delete x;
    if(y != nullptr) delete y;
    if(x_predicted != nullptr) delete x_predicted;
    if(y_predicted != nullptr) delete y_predicted;
}
/******************************************************************************/
template<typename X, typename Y>
void DatasetVariable<X, Y>::scale(){
    if(x == nullptr) return;
    X maxX, datoX;
    Y maxY, datoY;
    maxX = x[0];
    maxY = y[0];
    for(int i=1; i < n; i++){
        datoX = x[i];
        datoY = y[i];
        if(datoX > maxX) maxX = datoX;
        if(datoY > maxY) maxY = datoY;
    }
    this->maxX = maxX;
    this->maxY = maxY;
    for(int i=0; i < n; i++){
        x[i] = x[i] / maxX;
        y[i] = y[i] / maxY;
    }
}
template<typename X, typename Y>
void DatasetVariable<X, Y>::split(double percent){
    if(percent > 1) return;
    int n_back=n*percent;
    x_predicted = new X[n_back]{};
    y_predicted = new Y[n_back]{};
    n_predicted = n_back;
    X *auxX;
    Y *auxY;
    auxX = new X[n - n_back]{};
    auxY = new Y[n - n_back]{};
    for(int i=0; i < n; i++){
        if(i < (n - n_back)) auxX[i] = x[i];
        else x_predicted[i-(n - n_back)] = x[i];
        if(i < (n - n_back)) auxY[i] = y[i];
        else y_predicted[i-(n - n_back)] = y[i];
    }
    n -= n_back;
    delete x;
    delete y;
    x = auxX;
    y = auxY;
}
template <typename X, typename Y>
int DatasetVariable<X, Y>::getSize(){
    return n;
}
/******************************************************************************/
template <typename X, typename Y>
double DatasetVariable<X, Y>::forwardIntern(double (*activation)(double, double), 
                                            int index, double **weights, 
                                            double **neurons, 
                                            double *weightsOutput, int hiddenSize,
                                            int inputSize, double alpha){
    double output=0;
    for(int i=0; i < hiddenSize; i++){
        for(int j=0; j < inputSize; j++){
            neurons[index][i] = weights[j][i] * x[index];
        }
        neurons[index][i] = activation(neurons[index][i], alpha);
    }
    for(int i=0; i < hiddenSize; i++) 
        output += neurons[index][i] * weightsOutput[i];
    return activation(output, alpha);
}
template <typename X, typename Y>
double DatasetVariable<X, Y>::getError(int index){
    return y[index];
}
template <typename X, typename Y>
double DatasetVariable<X, Y>::getData(int indexI, int indexJ){
    return x[indexI];
}
template <typename X, typename Y>
int DatasetVariable<X, Y>::getSizeTest(){
    return n_predicted;
}
template <typename X, typename Y>
double DatasetVariable<X, Y>::forwardTest(double (*activation)(double, double), int index,
                                          double **weights, double **neurons, 
                                          double *weightsOutput, int hiddenSize,
                                          int inputSize, double alpha){
    double output=0;
    for(int i=0; i < hiddenSize; i++){
        for(int j=0; j < inputSize; j++){
            neurons[index][i] = weights[j][i] * x_predicted[index];
        }
        neurons[index][i] = activation(neurons[index][i], alpha);
    }
    for(int i=0; i < hiddenSize; i++) 
        output += neurons[index][i] * weightsOutput[i];
    return activation(output, alpha);
}
/******************************************************************************/
template <typename X, typename Y>
void DatasetVariable<X, Y>::scale(int indiceX){ }
#endif /* DATASETVARIABLE_H */

