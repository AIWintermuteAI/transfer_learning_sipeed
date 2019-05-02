/**
 * @file MBNet_1000.cpp
 * @brief Detect object type
 * @author Neucrack@sipeed.com
 */


#include "MBNet_1000.h"
#include "names.h"
#include "stdlib.h"
#include "errno.h"


MBNet_1000::MBNet_1000(KPUClass& kpu, Sipeed_ST7789& lcd, Sipeed_OV2640& camera)
:_kpu(kpu),_lcd(lcd), _camera(camera),
_model(nullptr), _count(0), _result(nullptr)
{
    _names = mbnet_label_name;
    memset(_statistics, 0, sizeof(_statistics));
}

MBNet_1000::~MBNet_1000()
{
    if(_model)
        free(_model);
}


int MBNet_1000::begin(const char* kmodel_name)
{
    File myFile;
    if(!_camera.begin())
        return -1;
    if(!_lcd.begin(15000000, COLOR_RED))
        return -2;
    _camera.run(true);
    _lcd.setTextSize(2);
    _lcd.setTextColor(COLOR_WHITE);

    if (!SD.begin()) 
    {
        _lcd.setCursor(100,100);
        _lcd.print("No SD Card");
        return -3;
    }

    myFile = SD.open(kmodel_name);
    if (!myFile)
        return -4;
    uint32_t fSize = myFile.size();
    _lcd.setCursor(100,100);
    _lcd.print("Loading ... ");
    _model = (uint8_t*)malloc(fSize);
    if(!_model)
    {
        _lcd.setCursor(100,140);
        _lcd.print("Memmory not enough ... ");
        return ENOMEM;
    }
    long ret = myFile.read(_model, fSize);
    myFile.close();
    if(ret != fSize)
    {
        free(_model);
        _model = nullptr;
        return -5;
    }

    if(_kpu.begin(_model) != 0)
    {
        free(_model);
        _model = nullptr;
        return -6;
    }
    return 0;
}

int MBNet_1000::detect()
{
    uint8_t* img;
    uint8_t* img888;

    img = _camera.snapshot();
    if(img == nullptr || img==0)
        return -1;
    img888 = _camera.getRGB888();
    if(_kpu.forward(img888) != 0)
    {
        return -2;
    }
    while( !_kpu.isForwardOk() );
    if( _kpu.getResult((uint8_t**)&_result, &_count) != 0)
    {
        return -3;
    }
    return 0;
}

void MBNet_1000::show()
{

    const char* name;
    uint8_t i, j;
    uint16_t* img;
float prob;
uint16_t _index;
    _count /= sizeof(float);

    label_init();
    _index = label_sort();

    for ( i = 0; i < 5; i++)
	{
		label_get(i, &prob, &name);
	}
    img = _camera.getRGB565();
    _lcd.fillRect(224,0, _lcd.width()-224, _lcd.height(), COLOR_RED);
    _lcd.drawImage(0, 0, _camera.width(), _camera.height(), img);
    _lcd.setTextSize(2);
    _lcd.setTextColor(COLOR_WHITE);
    _lcd.setCursor(0,0);
    _lcd.print(_names[_index]);
}


void MBNet_1000::label_init( )
{
	int i;
	for(i = 0; i < _count; i++)
		_index[i] = i;
}

uint16_t MBNet_1000::label_sort()
{
    uint16_t _index = 0;
    float _prob = 0;
    int i;
    for(i=0; i<_count; i++)
    {
      if(_result[i] > _prob) { _prob = _result[i]; _index = i;     }
    }
            return _index;
}


void MBNet_1000::label_get(uint16_t index, float* prob, const char** name)
{
	*prob = _result[index];
	*name = _names[index];
}
