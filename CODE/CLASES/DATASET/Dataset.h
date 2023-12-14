/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.h to edit this template
 */

/* 
 * File:   Dataset.h
 * Author: piero
 *
 * Created on 27 de noviembre de 2023, 19:19
 */

#ifndef DATASET_H
#define DATASET_H

//X = datatype of input, Y = datatype of output not an array
template <typename X, typename Y>
class Dataset {
public:
    /**************************************************************************/
    virtual ~Dataset();
    //Escala en valores pequeños
    virtual void scale(int) = 0 ; 
    virtual void scale() = 0;
    //Separa en dos arreglos, el porcentaje es para el tamaño de las pruebas
    virtual void split(double) = 0; 
    virtual double forwardIntern(double (*)(double, double), int, double **, 
                                 double **, double *, int, int, double) = 0;
    virtual double getError(int) = 0;
    virtual int getSize() = 0;
    virtual double getData(int, int) = 0;
    virtual int getSizeTest() = 0;
    virtual double forwardTest(double (*)(double, double), int, double **, 
                               double **, double *, int, int, double) = 0;
    /**************************************************************************/
    template <typename A, typename B>
    friend class NeuralNetwork;
    template <typename A, typename B>
    friend class DatasetArray;
    template <typename A, typename B>
    friend class DatasetVariable;
};

template <typename X, typename Y>
Dataset<X, Y>::~Dataset(){
    
}
#endif /* DATASET_H */

