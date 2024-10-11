### Dokumentation för nya tillagda funktioner i `machine_learning.h` och ändringar i `main.cpp`

- **predict**: Den här funktionen används för att analysera mönster och hjälpa datorn att lära sig genom att göra förutsägelser baserade på data.
  
- **train**: Detta är en träningsmodell som, baserad på utfallet av `predict`, bestämmer hur många epoker som modellen ska genomgå. Modellen går igenom en inlärningsprocess med hjälp av en lärkurva.
  
- **tempdata**: En variabel i `main` som används för att beräkna, förutsäga och skriva ut temperaturvärden.

### Inlämning

1. **Vad lärde ni er av projektet?**
   Vi fick en djupare förståelse för hur man implementerar maskininlärningsmodeller i ett befintligt system, och hur man utnyttjar funktioner som `predict` och `train` för att bygga modeller.

2. **Vad var lätt/svårt?**
   Det var relativt enkelt att följa den redan etablerade strukturen i projektet och arbeta med befintliga komponenter. Dock var det utmanande att bygga en linjär regressionsmodell, särskilt eftersom jag fortfarande känner mig osäker på hur alla delar ska sättas samman.

3. **Vad hade ni velat ha lärt er mer innan projektet?**
   Jag hade gärna velat ha mer erfarenhet av att jobba med STM32-plattformen istället för Arduino, då detta hade passat bättre för projektets krav.

4. **Övriga kommentarer?**
   Projektets upplägg var mycket uppskattat. Det var givande att arbeta med ett system där grundstrukturen redan fanns, vilket speglar hur det ofta ser ut i arbetslivet. Jag känner också att jag nu är mer bekväm med att använda klasser i C++ och förstår bättre hur man kan navigera sig genom befintliga kodbaser.

   Det var också utmanande att bygga LinReg-modellen, men genom att gå tillbaka till tidigare arbeten kunde jag identifiera vad som saknades i mitt program. Jag känner att jag har gjort stora framsteg, och jag är nöjd med resultatet.
