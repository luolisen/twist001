#include "x42s_protocol.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
int main(void)
{
    X42S_Frame f, good;
    X42S_MotorPosition p = {123U,0U};
    uint8_t axis;
    for (axis=1U;axis<=6U;++axis) {
        assert(X42S_BuildPositionRead(axis,&f));
        assert(f.id==((uint32_t)axis<<8) && f.extended==1 && f.remote==0);
        assert(f.len==2 && f.data[0]==0x36 && f.data[1]==0x6B);
    }
    assert(!X42S_BuildPositionRead(0,&f));
    assert(!X42S_BuildPositionRead(7,&f));
    assert(!X42S_BuildPositionRead(1,NULL));
    good=(X42S_Frame){0x100,1,0,7,{0x36,0,0,0,0x07,0xB3,0x6B,0}};
    assert(X42S_ParsePosition(1,&good,&p) && p.magnitude_tenths_deg==1971 && !p.negative);
    good.data[1]=1;assert(X42S_ParsePosition(1,&good,&p) && p.negative);
    memset(good.data+2,0xff,4);assert(X42S_ParsePosition(1,&good,&p) && p.magnitude_tenths_deg==UINT32_MAX);
    f=good;f.extended=0;assert(!X42S_ParsePosition(1,&f,&p));
    f=good;f.remote=1;assert(!X42S_ParsePosition(1,&f,&p));
    f=good;f.id=0x101;assert(!X42S_ParsePosition(1,&f,&p));
    f=good;f.data[1]=2;assert(!X42S_ParsePosition(1,&f,&p));
    f=good;f.data[0]=0x37;assert(!X42S_ParsePosition(1,&f,&p));
    f=good;f.data[6]=0x00;assert(!X42S_ParsePosition(1,&f,&p));
    for (axis=0;axis<=8;++axis) { f=good;f.len=axis;assert(X42S_ParsePosition(1,&f,&p)==(axis==7)); }
    assert(!X42S_ParsePosition(2,&good,&p));
    assert(!X42S_ParsePosition(0,&good,&p));
    assert(!X42S_ParsePosition(1,NULL,&p));
    assert(!X42S_ParsePosition(1,&good,NULL));
    p=(X42S_MotorPosition){123,0};f=good;f.len=1;
    assert(!X42S_ParsePosition(1,&f,&p) && p.magnitude_tenths_deg==123 && p.negative==0);
    puts("protocol: all checks passed");
    return 0;
}
