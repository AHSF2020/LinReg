/*
 * machine_learning.cpp
 *
 * Created: 2024-09-18 09:02:12
 *  Author: AdelH
 */ 

#include "vector.h"

using namespace container;

namespace ml
{

/********************************************************************************
 * @brief Linear regression model implementation.
 ********************************************************************************/
class LinReg
{
public:

    // Default constructor deleted.
    LinReg() = delete; 
    
    /********************************************************************************
     * @brief Creates a new linear regression model.
     * 
     * @param LinRegInput Reference to vector holding input values for training.
     * @param LinRegOutput Reference to vector holding output values for training.
     ********************************************************************************/
    LinReg(const Vector<double>& LinRegInput, const Vector<double>& LinRegOutput);

    /********************************************************************************
     * @brief Provides input values for training.
     * 
     * @return Reference to vector holding input values for training.
     ********************************************************************************/
    const Vector<double>& input() const;

    /********************************************************************************
     * @brief Provides output values for training.
     * 
     * @return Reference to vector holding output values for training.
     ********************************************************************************/
    const Vector<double>& output() const;

    /********************************************************************************
     * @brief Trains the model.
     * 
     * @param epochCount The number of epochs to train the model.
     * @param learningRate The adjustment rate of parameters (default = 1 %).
     * 
     * @return True if training was performed, false if an error occurred (i.e. 
     *         the epoch count = 0, learning_rate <= 0.0, or there are no training
     *         sets).
     ********************************************************************************/
    bool train(const size_t epochCount, const double learningRate = 0.01); 
    
    /********************************************************************************
     * @brief Performs prediction based on given input.
     * 
     * @param input The input for which to predict.
     * 
     * @return The predicted value as a double.
     ********************************************************************************/
    double predict(const double input) const;

private:

    void optimize(const double input, const double reference, const double learningRate);
    void randomizer();

    Vector<double> myInput;
    Vector<double> myOutput;
    Vector<double>myOrder;
    double myBias;
    double myWeight;
}; 
} // namespace ml