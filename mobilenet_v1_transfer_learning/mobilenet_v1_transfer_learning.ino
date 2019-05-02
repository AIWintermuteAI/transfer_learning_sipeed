#include <Sipeed_OV2640.h>
#include <Sipeed_ST7789.h>
#include "MBNet_1000.h"
#include "Maix_KPU.h"

SPIClass spi_(SPI0); // MUST be SPI0 for Maix series on board LCD
Sipeed_ST7789 lcd(320, 240, spi_);
Sipeed_OV2640 camera(128, 128, PIXFORMAT_RGB565);
KPUClass KPU;
MBNet_1000 mbnet(KPU, lcd, camera);

const char* kmodel_name = "model";


void setup()
{
    Serial.begin(115200);
    while (!Serial) {
        ; // wait for serial port to connect. Needed for native USB port only
    }
    
    Serial.println("init mobile net, load kmodel from SD card, it may takes a long time");
    if( mbnet.begin(kmodel_name) != 0)
    {
        Serial.println("mobile net init fail");
        while(1);
    }

}

void loop()
{
    if(mbnet.detect() != 0)
    {
        Serial.println("detect object fail");
        return;
    }
    mbnet.show();
}
