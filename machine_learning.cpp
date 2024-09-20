/*
 * machine_learning.cpp
 *
 * Created: 2024-09-18 09:22:04
 *  Author: AdelH
 */ 

#include <stdlib.h>
#include <time.h>

#include "machine_learning.h"
#include "vector.h"

namespace 
{
void initRandomGenerator()
{
    static bool generatorInitialized{false};

    if (!generatorInitialized)
    {
        srand(time(nullptr));
        generatorInitialized = true;
    }
}

} // namespace

namespace ml
{
    // konstruktor
    LinReg::LinReg(const Vector<double>& input, const Vector<double>& output)
        : myInput{input}
        , myOutput{output}
        , myOrder{}
        , myBias{0.5}
        , myWeight{0.5}
    {
        // fixar medlemsmariablerna som inte är satt i konstrutorn här
        myOrder.resize(myInput.size());

        for (auto i{0U}; i < myOrder.size(); ++i)
        {
            myOrder[i] = i;
        }
    } // end of construktor
    
    // ---------------getter------------------------------------
    const Vector<double>& LinReg::input() const
    {
        return myInput;  
    }
    
    const Vector<double>& LinReg::output() const
    {
        return myOutput;
    } // -------------end of getter---------------------  


    // --------------funktioner för metoder--------------------
    double LinReg::predict(const double input) const
    {
        const double yp = myWeight * input + myBias;
        return yp;
    }


    bool LinReg::train(const size_t epochCount, const double learningRate)
    {
         if (epochCount == 0U || learningRate <= 0) { return false;}

         initRandomGenerator();
         
         for (auto epoch{0U}; epoch < epochCount; ++epoch)
         {
            randomizer();

            for (const auto& i : myOrder)
            {
                optimize(myInput[i], myOutput[i], learningRate);
            }
         }
         return true;
    } // end class bool train

    void LinReg::optimize(const double input, const double reference, const double learningRate)
    {
        if (input == 0) {myBias = reference;}
        else
        {
            const auto prediction{myWeight * input + myBias};
            const auto error{reference - prediction};
            myBias   += error * learningRate;
            myWeight += error * learningRate * input;
        }
    } // end of class optimizer

    void LinReg::randomizer()
    {
    // Kör for-loop, där i = 0 upp till myOrder.size().
        for (auto i{0U}; i < myOrder.size(); ++i)
        {
            const auto temp{myOrder[i]};
            const auto r{rand() % myOrder.size()}; // Ger random mellan 0 - myOrder.size() - 1.
           myOrder[i] = myOrder[r];
           myOrder[r] = temp;
        } // end of class randomizer
    }


  } // Namespace ml
