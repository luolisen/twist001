#include "x42s_protocol.h"
#include <stddef.h>
#include <string.h>
int X42S_BuildPositionRead(uint8_t address, X42S_Frame *frame)
{
    if (frame == NULL || address < 1U || address > X42S_AXIS_COUNT) { return 0; }
    memset(frame, 0, sizeof(*frame));
    frame->id = (uint32_t)address << 8;
    frame->extended = 1U;
    frame->len = 2U;
    frame->data[0] = X42S_READ_POSITION;
    frame->data[1] = X42S_FRAME_END;
    return 1;
}
int X42S_ParsePosition(uint8_t address, const X42S_Frame *frame,
                      X42S_MotorPosition *position)
{
    X42S_MotorPosition parsed;
    if (frame == NULL || position == NULL || address < 1U || address > X42S_AXIS_COUNT) { return 0; }
    if (frame->id != ((uint32_t)address << 8) || frame->extended != 1U ||
        frame->remote != 0U || frame->len != 7U ||
        frame->data[0] != X42S_READ_POSITION || frame->data[1] > 1U ||
        frame->data[6] != X42S_FRAME_END) { return 0; }
    parsed.negative = frame->data[1];
    parsed.magnitude_tenths_deg = ((uint32_t)frame->data[2] << 24) |
                                ((uint32_t)frame->data[3] << 16) |
                                ((uint32_t)frame->data[4] << 8) | frame->data[5];
    *position = parsed;
    return 1;
}
