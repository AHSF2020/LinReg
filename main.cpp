/********************************************************************************
 * @brief Embedded temperature prediction system with a linear regression model.
 ********************************************************************************/
#include "adc.h"
#include "gpio.h"
#include "serial.h"
#include "timer.h"
#include "watchdog.h"
#include "machine_learning.h"

#include "vector.h"

using namespace container;
using namespace driver;

namespace 
{

constexpr uint8_t tempSensor{2U}; // Temp sensor PIN.

/********************************************************************************
 * @brief Devices used in the embedded system.
 *
 * @param led              LED connected to pin 13, used as an error indicator
 *                         and/or temperature prediction indicator.
 * @param predictionButton Button connected to pin 9, used to toggle the LED.
 * @param debounceTimer    Timer used to reduced the effect of contact bounces when
 *                         pressing the button.
 * @param predictionTimer  Timer used to predict the temperature every 60 seconds.
 ********************************************************************************/
GPIO led{8, GPIO::Direction::Output}; 
GPIO predictionButton{13, GPIO::Direction::InputPullup};
Timer debounceTimer{Timer::Circuit::Timer0, 300};
Timer predictionTimer{Timer::Circuit::Timer1, 60000};

/********************************************************************************
 * @brief Linear regression model and it's training data.
 *
 * @param input  Vector holding input values for training.
 * @param output Vector holding output values for training.
 * @param linReg Linear regression model instance.
 ********************************************************************************/
Vector<double> input {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
Vector<double> output {-50, -40, -30, -20, -10, 0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
ml::LinReg linReg{input, output};

/********************************************************************************
 * @brief Predicts the temperature with the linear regression model and prints
 *        the predicted value in the serial terminal.
 ********************************************************************************/
void tempData(void)
{
    const auto inputVoltage{adc::getDutyCycle(tempSensor)* 5};
    const auto temperature{static_cast<int>(linReg.predict(inputVoltage) + 0.5)};
    serial::printf("%d Celsius\n", temperature);
}

/********************************************************************************
 * @brief Callback routine called when predictionButton is pressed or released.
 *        Temperature prediction is performed every time the button is pressed.
 *        Pin change interrupts are disabled for 300 ms on the button's I/O port
 *        to reduce the effects of contact bounces.
 ********************************************************************************/
void buttonCallback(void) 
{
    predictionButton.disableInterruptsOnIoPort();
    debounceTimer.start();

    if (predictionButton.read()) 
    {
        tempData();
        predictionTimer.restart();
    }
}

/********************************************************************************
 * @brief Enables pin change interrupts on the button's I/O port 300 ms after
 *        press or release to reduce the effects of contact bounces.
 ********************************************************************************/
void debounceTimerCallback(void) 
{
    debounceTimer.stop();
    predictionButton.enableInterruptsOnIoPort();
}

/********************************************************************************
 * @brief Predicts the temperature and toggles the LED when the prediction timer 
 *        elapses, which is every 60 seconds (or 60 seconds after the last
 *        button push).
 ********************************************************************************/
void predictionTimerCallback(void) 
{
     led.toggle(); 
     tempData();
}

/********************************************************************************
 * @brief Trains linear regression model and initializes hardware.
 ********************************************************************************/
inline void setup(void) 
{
    if (!linReg.train(1000, 0.1)) 
    { 
        led.set(); 
        return;
    }

    serial::init();
    serial::printf("Finished training!\n");

    predictionButton.addCallback(buttonCallback);
    debounceTimer.addCallback(debounceTimerCallback);
    predictionTimer.addCallback(predictionTimerCallback);

    predictionButton.enableInterrupt();
    predictionTimer.start();
    watchdog::init(watchdog::Timeout::Timeout8192ms);
    watchdog::enableSystemReset();
}

} // namespace

/********************************************************************************
 * @brief Perform a setup of the system, then running the program as long as
 *        voltage is supplied. The hardware is interrupt controlled, hence the
 *        while loop is almost empty. If the program gets stuck anywhere, the
 *        watchdog timer won't be reset in time and the program will then restart.
 ********************************************************************************/
int main(void)
{
    setup();
   
    while (1)
    {
        watchdog::reset();
    }

};
