/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/main.cc to edit this template
 */

/* 
 * File:   main.cpp
 * Author: piero
 *
 * Created on 27 de noviembre de 2023, 19:17
 */

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#include "Dataset.h"
#include "DatasetVariable.h"
#include "NeuralNetwork.h"

int main(int argc, char** argv) {
    double **a, *b;
    a = new double*[4];
    b = new double[4]{0.92, 0.86, 0.89, 1};
    a[0] = new double[2]{2, 9};
    a[1] = new double[2]{1, 5};
    a[2] = new double[2]{3, 6};
    a[3] = new double[2]{4, 8};
    DatasetArray<double, double> data(a, b, 4, 2); //Carga los arreglos en Dataset
    NeuralNetwork<double, double> nn(2, 3, 1); //Crea la red neuronal
    data.scale(0); //Escala todo en base al mayor dato para que esté menor a 1, nos sirve con la función sigmoide
    data.split(0.3); //Separa un porcentaje para pruebas
    nn.setDatasetArray(data); //Carga el Dataset a la red neuronal
    nn.updateNeurons(); //Actualiza la red neuronal con la información del Dataset
    nn.training("sigmoid_function", "gradient_descent", 1000, true); //Entrena
    
    return 0;
}

