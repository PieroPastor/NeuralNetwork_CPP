/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.h to edit this template
 */

/* 
 * File:   NeuralNetwork.h
 * Author: piero
 *
 * Created on 28 de noviembre de 2023, 19:39
 */

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cmath>
#include <cstring>
#include <random>
#include <chrono>
#include "Dataset.h"
#include "DatasetArray.h"
#include "DatasetVariable.h"

//X = datatype of input, Y = datatype of output not an array
template <typename A, typename B>
class NeuralNetwork {
public:
    NeuralNetwork();
    NeuralNetwork(int, int, int);
    NeuralNetwork(const DatasetArray<A, B>&, int, int, int);
    NeuralNetwork(const DatasetVariable<A, B>&, int, int, int);
    virtual ~NeuralNetwork();
    /**************************************************************************/
    void setDatasetArray(const DatasetArray<A, B>&);
    void setDatasetVariable(const DatasetVariable<A, B>&);
    void updateNeurons(); //Reset the neurons with the sizes selected before
    //Nombre de la función de activación, el método de optimización, 
    //número de iteraciones, imprime avances de entrenamiento, alpha para algunas funciones
    void training(const char *, const char *, int, bool, double alpha=0.01);
private:
    Dataset<A, B> *data;
    int inputSize;
    int hiddenSize;
    int outputSize;
    double **weights;
    double *weightsOutput;
    double **neurons;
    double *output;
    double *error;
    /***************************CONSTANTS VARIABLES****************************/
    static constexpr double E = 2.71828; 
    /***************************AUXILIARY FUNCTIONS****************************/
    void setWeights();
    double loss();
    void forward(const char *, double); //Select a function from the library
    void forward(double (*)(double, double), double); //Select a function created
    void backwardGradientDescent(const char *, double);
    void backwardGradientDescent(double (*)(double, double), double);
    void proof();
    void test(const char*, double);
    void test(double (*)(double, double), double);
    void saveWeights();
    /**************************ACTIVATION FUNCTIONS****************************/
    static double sigmoid(double, double);
    static double sigmoidPrime(double, double);
    static double identity(double, double);
    static double identityPrime(double, double);
    static double tanh(double, double);
    static double tanhPrime(double, double);
    static double relu(double, double);
    static double reluPrime(double, double);
    static double lrelu(double, double);
    static double lreluPrime(double, double);
/**************************************************************************/
template <typename X, typename Y>
friend class DatasetArray;
template <typename X, typename Y>
friend class DatasetVariable;
template <typename X, typename Y>
friend class Dataset;
};
/************************************BUILDERS**********************************/
template <typename A, typename B>
NeuralNetwork<A, B>::NeuralNetwork(){
    inputSize = 0;
    hiddenSize = 0;
    outputSize = 0;
    weights = nullptr;
    weightsOutput = nullptr;
    neurons = nullptr;
    output = nullptr;
    error = nullptr;
}
template <typename A, typename B>
NeuralNetwork<A, B>::NeuralNetwork(int a, int b, int c){
    data = nullptr;
    inputSize = a;
    hiddenSize = b;
    outputSize = c;
    weights = new double*[a]{};
    weightsOutput = new double[b]{};
    if(data != nullptr){
        neurons = new double*[data->getSize()]{};
        output = new double[data->getSize()]{};
        error = new double[data->getSize()]{};
        for(int i=0; i < data->getSize(); i++) neurons[i] = new double[b]{};
    }
    setWeights();
}
template <typename A, typename B>
NeuralNetwork<A, B>::NeuralNetwork(const DatasetArray<A, B>& orig, int a, 
                                      int b, int c){
    setDatasetArray(orig);
    inputSize = a;
    hiddenSize = b;
    outputSize = c;
    weights = new double*[a];
    weightsOutput = new double[b];
    if(data != nullptr){
        neurons = new double*[data->getSize()]{};
        output = new double[data->getSize()]{};
        error = new double[data->getSize()]{};
        for(int i=0; i < data->getSize(); i++) neurons[i] = new double[b]{};
    }
    setWeights();
}
template <typename A, typename B>
NeuralNetwork<A, B>::NeuralNetwork(const DatasetVariable<A, B>& orig, int a, 
                                      int b, int c){
    setDatasetVariable(orig);
    inputSize = a;
    hiddenSize = b;
    outputSize = c;
    weights = new double*[a];
    weightsOutput = new double[b];
    neurons = new double*[data->getSize()]{};
    output = new double[data->getSize()]{};
    for(int i=0; i < data->getSize(); i++) neurons[i] = new double[b]{};
    error = new double[data->getSize()]{};
    setWeights();
}
template <typename A, typename B>
NeuralNetwork<A, B>::~NeuralNetwork(){
    if(data != nullptr) delete data;
    if(weights != nullptr) delete [] weights;
    if(weightsOutput != nullptr) delete weightsOutput;
    if(neurons != nullptr) delete [] neurons;
    if(output != nullptr) delete output;
    if(error != nullptr) delete error;
}
/******************************************************************************/
template <typename A, typename B>
void NeuralNetwork<A, B>::setDatasetArray(const DatasetArray<A, B>& orig){
    data = new DatasetArray<A, B>(orig);
}
template <typename A, typename B>
void NeuralNetwork<A, B>::setDatasetVariable(const DatasetVariable<A, B>& orig){
    data = new DatasetVariable<A, B>(orig);
}
template <typename A, typename B>
void NeuralNetwork<A, B>::updateNeurons(){
    if(data != nullptr){
        neurons = new double*[data->getSize()]{};
        output = new double[data->getSize()]{};
        error = new double[data->getSize()]{};
        for(int i=0; i < data->getSize(); i++) 
            neurons[i] = new double[hiddenSize]{};
    }
}
template <typename A, typename B>
void NeuralNetwork<A, B>::training(const char *nom, const char *nom2, 
                                   int epochs, bool verbose, double alpha){
    for(int i=0; i < epochs; i++){
        forward(nom, alpha);
        if(verbose){
            cout << "Train " << i+1 << ": ";
            proof();
            cout << endl;
            cout << "Loss: " << loss() << endl;
        }
        if(strcmp(nom2, "gradient_descent") == 0) backwardGradientDescent(nom, alpha);
    }
    test(nom, alpha);
    saveWeights();
}
/**************************** AUXILIARY FUNCTIONS******************************/
template <typename A, typename B>
void NeuralNetwork<A, B>::setWeights(){
    double randWeights[]={0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    int index;
    for(int i=0; i < inputSize; i++) {
        weights[i] = new double[hiddenSize]{};
        for(int j=0; j < hiddenSize; j++){
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            /*Crear un generador de números aleatorios utilizando la semilla 
            del reloj del sistema*/
            std::mt19937 generator(seed);
            // Definir un rango para los números aleatorios
            std::uniform_int_distribution<int> distribution(0, 9);
            index = distribution(generator);
            weights[i][j] = randWeights[index];
        }
    }
    for(int i=0; i < hiddenSize; i++){
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        /*Crear un generador de números aleatorios utilizando la semilla del 
        reloj del sistema*/
        std::mt19937 generator(seed);
        // Definir un rango para los números aleatorios
        std::uniform_int_distribution<int> distribution(0, 9);
        index = distribution(generator);
        weightsOutput[i] = randWeights[index];
    }
}
/**********************************LEARNING************************************/
template <typename A, typename B>
void NeuralNetwork<A, B>::forward(const char *nom, double alpha){
    if(strcmp(nom, "sigmoid_function") == 0) forward(this->sigmoid, alpha);
    else if(strcmp(nom, "identity") == 0) forward(this->identity, alpha);
    else if(strcmp(nom, "tanh") == 0) forward(this->tanh, alpha);
    else if(strcmp(nom, "relu") == 0) forward(this->relu, alpha);
}
template <typename A, typename B>    
void NeuralNetwork<A, B>::forward(double (*activation)(double, double), double alpha){
    for(int i=0; i < data->getSize(); i++){
        output[i] = data->forwardIntern(activation, i, this->weights, 
                                        this->neurons, this->weightsOutput, 
                                        this->hiddenSize, this->inputSize, alpha);
        error[i] = output[i];
    }
}
template <typename A, typename B>
void NeuralNetwork<A, B>::backwardGradientDescent(const char *nom, double alpha){
    if(strcmp(nom, "sigmoid_function") == 0) backwardGradientDescent(this->sigmoidPrime, alpha);
    else if(strcmp(nom, "identity") == 0) backwardGradientDescent(this->identityPrime, alpha);
    else if(strcmp(nom, "tanh") == 0) backwardGradientDescent(this->tanh, alpha);
    else if(strcmp(nom, "relu") == 0) backwardGradientDescent(this->relu, alpha);
}
template <typename A, typename B>
void NeuralNetwork<A, B>::backwardGradientDescent(double(*activation)(double, double), double alpha){
    double errorOutput[data->getSize()];
    double **errorWeights;
    errorWeights = new double*[data->getSize()];
    for(int i=0; i < data->getSize(); i++) 
        errorWeights[i] = new double[hiddenSize]{};
    for(int i=0; i < data->getSize(); i++){
        errorOutput[i] = data->getError(i) - error[i];
        errorOutput[i] = errorOutput[i] * activation(error[i], alpha);
        //Transpuesta de la matris WEIGHTSOUTPUT
        for(int j=0; j < hiddenSize; j++) 
            errorWeights[i][j] = errorOutput[i] * weightsOutput[j]; 
    }
    /*Multiplico cada valor en los errores de peso por la derivada de la funcion 
    de activacion*/
    for(int i=0; i < data->getSize(); i++){
        for(int j=0; j < hiddenSize; j++){
            errorWeights[i][j] = errorWeights[i][j] * activation(neurons[i][j], alpha);
        }
    }
    for(int i=0; i < inputSize; i++){
        for(int j=0; j < hiddenSize; j++){
            /*Se deben de multiplica data (getSizeXinput)* 
            errorWeights(getSizeXhiddenSize), al trasponer data si permite 
            multiplicar*/
            for(int k=0; k < data->getSize(); k++){
                weights[i][j] += data->getData(k, i) * errorWeights[k][j];
            }
        }
    }
    for(int i=0; i < hiddenSize; i++){
        for(int j=0; j < data->getSize(); j++){
            /*Se deben de multiplica neurons (getSizeXhiddenSize)* 
            errorOutput(getSizeX1), al trasponer data si permite multiplicar*/
            weightsOutput[i] += neurons[j][i] * errorOutput[j];
        }
    }
    delete [] errorWeights;
}
template <typename A, typename B>
void NeuralNetwork<A, B>::proof(){
    cout << "[";
    for(int i=0; i < data->getSize(); i++){
        cout << fixed << setprecision(5) << output[i];
        if(i != data->getSize()-1) cout << ", "; 
    }
    cout << "]";
}
template <typename A, typename B>
double NeuralNetwork<A, B>::loss(){
    double perdida=0;
    for(int i=0; i < data->getSize(); i++) 
        perdida += (0.5) * pow(error[i] - data->getError(i), 2);
    return perdida;
}
template <typename A, typename B>
void NeuralNetwork<A, B>::test(const char *nom, double alpha){
    if(strcmp(nom, "sigmoid_function") == 0) test(this->sigmoid, alpha);
    else if(strcmp(nom, "identity") == 0) test(this->identity, alpha);
    else if(strcmp(nom, "tanh") == 0) test(this->tanh, alpha);
    else if(strcmp(nom, "relu") == 0) test(this->relu, alpha);
}
template <typename A, typename B>
void NeuralNetwork<A, B>::test(double (*activation)(double, double), double alpha){
    for(int i=0; i < data->getSizeTest(); i++){
        cout << "Data " << i+1 << ": "; 
        cout << data->forwardTest(activation, i, this->weights, this->neurons,
                                  this->weightsOutput, this->hiddenSize, 
                                  this->inputSize, alpha) << endl;
    }
}
template <typename A, typename B>
void NeuralNetwork<A, B>::saveWeights(){
    ofstream arch("weights.txt", ios::out);
    arch << "NEURONS WEIGHTS: " << endl;
    for(int i=0; i < inputSize; i++){
        arch << "[";
        for(int j=0; j < hiddenSize; j++){
            arch << weights[i][j];
            if(j != hiddenSize-1) arch << ", ";
            else arch << "]";
        }
        arch << endl;
    }
    arch << "OUTPUT WEIGHTS: " << endl << "[";
    for(int i=0; i < hiddenSize; i++){
        arch << weightsOutput[i];
        if(i != hiddenSize-1) arch << ", ";
    }
    arch << "]";
}
/*************************** ACTIVATION FUNCTIONS******************************/
/*En las derivadas se asume que ingresará el valor de output de la función normal,
 * por eso se usa directamente la "x", ya que, la derivada requiere la función 
 * normal*/
template <typename A, typename B>
double NeuralNetwork<A, B>::sigmoid(double x, double alpha){
    return 1/(1+pow(E, x * -1));
}
template <typename A, typename B>
double NeuralNetwork<A, B>::sigmoidPrime(double x, double alpha){
    return x * (1 - x);
}
template <typename A, typename B>
double NeuralNetwork<A, B>::identity(double x, double alpha){
    return x;
}
template <typename A, typename B>
double NeuralNetwork<A, B>::identityPrime(double x, double alpha){
    return 1;
}
template <typename A, typename B>
double NeuralNetwork<A, B>::tanh(double x, double alpha){
    return 2/(1 + pow(E, -2 * x)) - 1;
}
template <typename A, typename B>
double NeuralNetwork<A, B>::tanhPrime(double x, double alpha){
    return 1 - (x * x); 
}
template <typename A, typename B>
double NeuralNetwork<A, B>::relu(double x, double alpha){
    return (x<0)?0:x;
}
template <typename A, typename B>
double NeuralNetwork<A, B>::reluPrime(double x, double alpha){
    return (x==0)?0:1;
}
template <typename A, typename B>
double NeuralNetwork<A, B>::lrelu(double x, double alpha){
    return (x*alpha > x)?(x*alpha):x;
}
template <typename A, typename B>
double NeuralNetwork<A, B>::lreluPrime(double x, double alpha){
    //Como será input el output de la función principal se debe conseguir el
    //input anterior que se colocó en la función principal
    if(x/alpha < x) return(x/alpha>=0)?(x/alpha):alpha;
    else (x>=0)?x:alpha;
}
#endif /* NEURALNETWORK_H */

