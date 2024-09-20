/********************************************************************************
 * @brief Demonstration of GPIO device drivers in C++.
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

constexpr uint8_t tempSensor{2U};

/********************************************************************************
 * @brief Devices used in the embedded system.
 *
 * @param errorLed    LED connected to pin 13, used as an error indicator.
 * @param predictionButton Button connected to pin 9, used to toggle the LED.
 * @param debounceTimer  Timer used to reduced the effect of contact bounces when
 *                pressing the button.
 * @param predictionTimer  Timer used to toggle the LED every 100 ms when enabled.
 ********************************************************************************/
GPIO errorLed{8, GPIO::Direction::Output};
GPIO predictionButton{13, GPIO::Direction::InputPullup};
Timer debounceTimer{Timer::Circuit::Timer0, 300};
Timer predictionTimer{Timer::Circuit::Timer1, 60000};

Vector<double> input {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5};
Vector<double> output {-50, -40, -30, -20, -10, 0.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
ml::LinReg LinReg{input, output};

// Så här beräkningar du inspänningen (det som du låter din modell prediktera på):
// Uin = duty cycle * 5;
// Låt din modell prediktera med Uin som input, den bör gissa på motsvarande temperaturen om den har tränats rätt.

/********************************************************************************
 * @brief Callback routine called when predictionButton is pressed or released.
 *        Every time predictionButton is pressed, predictionTimer is toggled, which indirectly
 *        toggles the LED (since predictionTimer is responsible for toggling errorLed).
 *        Pin change interrupts are disabled for 300 ms on the button's I/O port
 *        to reduce the effects of contact bounces.
 ********************************************************************************/

 void tempData(void)
 {
     // Beräkna Uin genom att kalla på adc::dutyCycle, multiplicera med 5.
     const auto inputVoltage{adc::getDutyCycle(tempSensor)* 5};

     // Hör ska vi prediktera temperaturen.
     const int temperature{static_cast<int>(LinReg.predict(inputVoltage) + 0.5)};
     serial::printf("%d Celcius\n", temperature);

 }
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
 * @brief Toggles errorLed when predictionTimer elapses, which is every 100 ms when enabled.
 ********************************************************************************/
void predictionTimerCallback(void) 
{
     errorLed.toggle(); // Ta bort denna rad, det var bara för att visa.
     // Prediktera tenperaturen här (precis som ovan),
     tempData();
}

/********************************************************************************
 * @brief Sets callback routines, enabled pin change interrupt on predictionButton and
 *        enables the watchdog timer in system reset mode.
 ********************************************************************************/
inline void setup(void) 
{
    if (!LinReg.train(1000 /*antal epoch*/, 0.1 /*LR*/)) 
    { 
        errorLed.set(); 
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


