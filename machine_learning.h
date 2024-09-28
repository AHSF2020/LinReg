/*
 * machine_learning.cpp
 *
 * Created: 2024-09-18 09:02:12
 *  Author: AdelH
 */ 

/*
 * Kommentarer från Erik:
 *
 * Överlag ser det mycket bra ut. Det enda jag har att anmärka på är:
 *     - Använd engelska kommentarer.
 *     - Kommentera helst enligt Doxygen, se LinReg::LinReg samt LinReg::predict nedan.
 * 
 */

#include "vector.h"

using namespace container;

namespace ml
{

class LinReg
{
public:
    LinReg() = delete;
    
    /********************************************************************************
     * @brief Creates a new linear regression model.
     * 
     * @param LinRegInput Reference to vector holding input values for training.
     * @param LinRegOutput Reference to vector holding output values for training.
     ********************************************************************************/
    LinReg(const Vector<double>& LinRegInput, const Vector<double>& LinRegOutput);

    const Vector<double>& input() const;
    const Vector<double>& output() const;

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
    Vector<double> myInput;
    Vector<double> myOutput;
    Vector<double>myOrder;
    double myBias;
    double myWeight;

    void optimize(const double input, const double reference, const double learningRate);
    void randomizer();
}; // end class LinReg

 } // namespace ml