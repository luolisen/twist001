#ifndef X42S_PROTOCOL_H
#define X42S_PROTOCOL_H
#include <stdint.h>
#define X42S_AXIS_COUNT 6U
#define X42S_READ_POSITION 0x36U
#define X42S_FRAME_END 0x6BU
typedef struct {
    uint32_t id;
    uint8_t extended;
    uint8_t remote;
    uint8_t len;
    uint8_t data[8];
} X42S_Frame;
typedef struct {
    uint32_t magnitude_tenths_deg;
    uint8_t negative;
} X42S_MotorPosition;
/* Only a position-read request can be built by this API. */
int X42S_BuildPositionRead(uint8_t address, X42S_Frame *frame);
int X42S_ParsePosition(uint8_t address, const X42S_Frame *frame,
                      X42S_MotorPosition *position);
#endif
