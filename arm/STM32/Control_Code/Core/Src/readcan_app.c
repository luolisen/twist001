#include "readcan_app.h"
#include "gripper_link.h"
#include "gripper_can.h"
#include "x42s_protocol.h"
#include "can.h"
#include "usart.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#define REPLY_TIMEOUT_MS 250U
#define TX_TIMEOUT_MS 50U
#define LINE_CAPACITY 128U
#define FEEDBACK_FRESH_MS 1000U
/* Main-loop ownership: no CAN/UART ISR writes these records. */
typedef struct {
    X42S_MotorPosition position;
    uint32_t received_ms;
    uint8_t valid;
} AxisFeedback;
static AxisFeedback feedback[X42S_AXIS_COUNT];
static uint8_t teaching, playing;
static uint32_t teach_sequence, play_heartbeat;
static uint8_t position_scaled[6];
static uint8_t play_rate=1;
static uint16_t play_speed_override[6];
static const uint16_t play_speed_tenths_rpm[6]={50,50,50,16,16,1};
static const uint16_t play_step_tenths_deg[6]={150,150,150,50,50,5};
static void text(const char *s)
{
    (void)HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)strlen(s), 100U);
}
static void status(const char *kind, uint8_t address)
{
    char line[100];
    (void)snprintf(line, sizeof(line), "%s J%u CAN_ERROR=0x%08lX ESR=0x%08lX\r\n",
                   kind, address, (unsigned long)HAL_CAN_GetError(&hcan2),
                   (unsigned long)hcan2.Instance->ESR);
    text(line);
}
static int receive(X42S_Frame *frame)
{
    CAN_RxHeaderTypeDef header;
    if (HAL_CAN_GetRxFifoFillLevel(&hcan2, CAN_RX_FIFO0) == 0U) { return 0; }
    if (HAL_CAN_GetRxMessage(&hcan2, CAN_RX_FIFO0, &header, frame->data) != HAL_OK) { return 0; }
    frame->extended = (header.IDE == CAN_ID_EXT) ? 1U : 0U;
    frame->remote = (header.RTR == CAN_RTR_REMOTE) ? 1U : 0U;
    frame->id = frame->extended ? header.ExtId : header.StdId;
    frame->len = (uint8_t)header.DLC;
    return 1;
}
static void raw(const char *direction, const X42S_Frame *frame)
{
    char line[120];
    unsigned int i;
    int n = snprintf(line, sizeof(line), "%s t=%lu ID=0x%08lX %s %s DLC=%u DATA=",
                     direction, (unsigned long)HAL_GetTick(), (unsigned long)frame->id,
                     frame->extended ? "EXT" : "STD", frame->remote ? "RTR" : "DATA", frame->len);
    for (i = 0; i < frame->len && i < 8U && n > 0 && n < (int)sizeof(line) - 6; ++i) {
        n += snprintf(line + n, sizeof(line) - (size_t)n, "%02X ", frame->data[i]);
    }
    if (n > 0 && n < (int)sizeof(line) - 3) { (void)snprintf(line+n, sizeof(line)-(size_t)n, "\r\n"); }
    text(line);
}
static void print_position(uint8_t address, const AxisFeedback *axis)
{
    char line[160];
    uint32_t age = HAL_GetTick() - axis->received_ms;
    const X42S_MotorPosition *p = &axis->position;
    (void)snprintf(line, sizeof(line),
        "POS J%u motor_deg=%s%lu.%lu rx_ms=%lu age_ms=%lu valid=%u fresh=%u model=UNHOMED\r\n",
        address, p->negative ? "-" : "", (unsigned long)(p->magnitude_tenths_deg/10U),
        (unsigned long)(p->magnitude_tenths_deg%10U), (unsigned long)axis->received_ms,
        (unsigned long)age, axis->valid, axis->valid && age <= FEEDBACK_FRESH_MS ? 1U : 0U);
    text(line);
}
static void read_position_mode(uint8_t address, int quiet)
{
    X42S_Frame request, frame;
    CAN_TxHeaderTypeDef header = {0};
    uint32_t mailbox, start, count;
    AxisFeedback *axis = &feedback[address-1U];
    axis->valid = 0U;
    /* Drop older queued frames. Continuous unrelated traffic is bounded. */
    for (count = 0U; count < 32U && receive(&frame); ++count) { }
    if (HAL_CAN_GetRxFifoFillLevel(&hcan2, CAN_RX_FIFO0) != 0U) {
        status("RX_BUSY", address); return;
    }
    if (!X42S_BuildPositionRead(address, &request)) { return; }
    header.IDE = CAN_ID_EXT;
    header.ExtId = request.id;
    header.RTR = CAN_RTR_DATA;
    header.DLC = request.len;
    header.TransmitGlobalTime = DISABLE;
    start = HAL_GetTick();
    if (HAL_CAN_AddTxMessage(&hcan2, &header, request.data, &mailbox) != HAL_OK) {
        status("TX_ERROR", address); return;
    }
    if(!quiet) raw("TX", &request); /* Means queued, not proof of an ACK or motor reply. */
    while (HAL_CAN_IsTxMessagePending(&hcan2, mailbox) != 0U) {
        if (HAL_GetTick() - start >= TX_TIMEOUT_MS) {
            (void)HAL_CAN_AbortTxRequest(&hcan2, mailbox);
            status("TX_TIMEOUT", address); return;
        }
    }
    start = HAL_GetTick();
    while (HAL_GetTick() - start < REPLY_TIMEOUT_MS) {
        if (!receive(&frame)) { continue; }
        uint32_t received_ms = HAL_GetTick();
        if(!quiet) raw("RX", &frame);
        if (X42S_ParsePosition(address, &frame, &axis->position)) {
            axis->received_ms = received_ms;
            axis->valid = 1U;
            if(!quiet) print_position(address, axis);
            return;
        }
    }
    status("REPLY_TIMEOUT", address);
}
static void read_encoder(uint8_t address)
{
    X42S_Frame request, frame;
    CAN_TxHeaderTypeDef header = {0};
    uint32_t mailbox, start, count;
    char result[120];
    /* Drop older queued frames. Continuous unrelated traffic is bounded. */
    for (count = 0U; count < 32U && receive(&frame); ++count) { }
    if (HAL_CAN_GetRxFifoFillLevel(&hcan2, CAN_RX_FIFO0) != 0U) {
        status("RX_BUSY", address); return;
    }
    if (!X42S_BuildPositionRead(address, &request)) { return; }
    request.data[0] = 0x31U; /* Read-only single-turn absolute encoder. */
    header.IDE = CAN_ID_EXT;
    header.ExtId = request.id;
    header.RTR = CAN_RTR_DATA;
    header.DLC = request.len;
    header.TransmitGlobalTime = DISABLE;
    start = HAL_GetTick();
    if (HAL_CAN_AddTxMessage(&hcan2, &header, request.data, &mailbox) != HAL_OK) {
        status("TX_ERROR", address); return;
    }
    raw("TX", &request); /* Means queued, not proof of an ACK or motor reply. */
    while (HAL_CAN_IsTxMessagePending(&hcan2, mailbox) != 0U) {
        if (HAL_GetTick() - start >= TX_TIMEOUT_MS) {
            (void)HAL_CAN_AbortTxRequest(&hcan2, mailbox);
            status("TX_TIMEOUT", address); return;
        }
    }
    start = HAL_GetTick();
    while (HAL_GetTick() - start < REPLY_TIMEOUT_MS) {
        if (!receive(&frame)) { continue; }
        uint32_t received_ms = HAL_GetTick();
        raw("RX", &frame);
        if (frame.id == ((uint32_t)address << 8) && frame.extended == 1U &&
            frame.remote == 0U && frame.len == 4U && frame.data[0] == 0x31U &&
            frame.data[3] == 0x6BU) {
            unsigned int value = ((unsigned int)frame.data[1] << 8) | frame.data[2];
            snprintf(result, sizeof(result), "ENC J%u raw=%u rx_ms=%lu single_turn=1 model=UNHOMED\r\n",
                     address, value, (unsigned long)received_ms);
            text(result);
            return;
        }
    }
    status("REPLY_TIMEOUT", address);
}
static void listen(void)
{
    X42S_Frame frame;
    uint32_t start = HAL_GetTick(), count = 0U;
    while (HAL_GetTick() - start < 1000U && count < 100U) {
        if (receive(&frame)) { raw("RX", &frame); ++count; }
    }
    text("LISTEN_DONE\r\n");
}
static void read_position(uint8_t axis) {read_position_mode(axis,0);}
static int motor_exchange(uint8_t axis,const uint8_t *data,uint8_t len,uint8_t code,uint8_t *value)
{
    X42S_Frame frame;CAN_TxHeaderTypeDef h={0};uint32_t mailbox,start,count;
    for(count=0;count<32U && receive(&frame);++count) { }
    if(HAL_CAN_GetRxFifoFillLevel(&hcan2,CAN_RX_FIFO0)) return 0;
    h.IDE=CAN_ID_EXT;h.ExtId=(uint32_t)axis<<8;h.RTR=CAN_RTR_DATA;h.DLC=len;
    if(HAL_CAN_AddTxMessage(&hcan2,&h,(uint8_t *)data,&mailbox)!=HAL_OK) return 0;
    start=HAL_GetTick();
    while(HAL_CAN_IsTxMessagePending(&hcan2,mailbox)) {
        if(HAL_GetTick()-start>=TX_TIMEOUT_MS) {(void)HAL_CAN_AbortTxRequest(&hcan2,mailbox);return 0;}
    }
    start=HAL_GetTick();
    while(HAL_GetTick()-start<REPLY_TIMEOUT_MS) {
        if(receive(&frame) && frame.id==((uint32_t)axis<<8) && frame.extended && !frame.remote) {
            if(code==0x1A) raw("OPTIONS_RX",&frame);
            if(code==0x33 || code==0x34 || code==0x41) {raw("DIAG_RX",&frame);return 1;}
            if(code==0x1A && frame.len==4U && frame.data[0]==code && frame.data[3]==0x6B
               && ((frame.data[1]^frame.data[2])&0x86U)==0U) {
                /* Two option bytes observed on V2.0B. Accept relevant flags only
                   when both agree; never guess type/closed-loop/angle scale. */
                *value=frame.data[1]&0x86U;return 1;
            }
            if(frame.len==3U && frame.data[0]==code && frame.data[2]==0x6B) {*value=frame.data[1];return 1;}
        }
    }
    return 0;
}
static int motor_states(void)
{
    const uint8_t q[]={0x3A,0x6B};uint8_t i,f;char row[80];int disabled=1;
    for(i=1;i<=6;++i) {
        if(!motor_exchange(i,q,2,0x3A,&f)) {snprintf(row,sizeof(row),"MOTOR_STATE J%u valid=0\r\n",i);disabled=0;}
        else {snprintf(row,sizeof(row),"MOTOR_STATE J%u valid=1 enabled=%u flags=0x%02X\r\n",i,f&1U,f);if(f&1U)disabled=0;}
        text(row);
    }
    return disabled;
}
static void teach_begin(void)
{
    const uint8_t off[]={0xF3,0xAB,0,0,0x6B};uint8_t i,reply;int ack=1;
    if(playing) {text("TEACH_REJECT playback active\r\n");return;}
    teaching=0;
    for(i=1;i<=6;++i) {read_position(i);if(!feedback[i-1].valid) {text("TEACH_ERROR position precheck failed; no release sent\r\n");return;}}
    for(i=1;i<=6;++i) if(!motor_exchange(i,off,5,0xF3,&reply)||reply!=2) ack=0;
    if(!motor_states()) {text("TEACH_ERROR release unconfirmed; retain support\r\n");return;}
    teaching=1;teach_sequence=0;
    text(ack?"TEACH_STARTED released=6 confirmed=STATE_AND_ACK model=UNHOMED\r\n":"TEACH_STARTED released=6 confirmed=STATE model=UNHOMED\r\n");
}
static void teach_sample(void)
{
    char row[360];uint8_t i,mask=0;uint32_t start=HAL_GetTick(),end;int n;
    if(!teaching) {text("REJECT teaching not started\r\n");return;}
    for(i=1;i<=6;++i) {read_position_mode(i,1);if(feedback[i-1].valid) mask|=(uint8_t)(1U<<(i-1));}
    end=HAL_GetTick();n=snprintf(row,sizeof(row),"TEACH_FRAME,%lu,%lu,%lu,%u",(unsigned long)teach_sequence++,(unsigned long)start,(unsigned long)end,mask);
    for(i=0;i<6;++i) {
        if(feedback[i].valid)n+=snprintf(row+n,sizeof(row)-(size_t)n,",%s%lu",feedback[i].position.negative?"-":"",(unsigned long)feedback[i].position.magnitude_tenths_deg);
        else n+=snprintf(row+n,sizeof(row)-(size_t)n,",NA");
    }
    for(i=0;i<6;++i) {
        if(feedback[i].valid)n+=snprintf(row+n,sizeof(row)-(size_t)n,",%lu",(unsigned long)feedback[i].received_ms);
        else n+=snprintf(row+n,sizeof(row)-(size_t)n,",NA");
    }
    snprintf(row+n,sizeof(row)-(size_t)n,"\r\n");text(row);
}
/* Manual 4.2 and 5.3.8: repeated code byte in each CAN packet. */
static int play_send(uint8_t axis,const uint8_t *body,size_t length)
{
    CAN_TxHeaderTypeDef h={0};uint8_t packet[8];size_t offset=1,count;uint32_t index=0,mailbox,start;
    if(axis>6||length<2||body[length-1]!=0x6B)return 0;
    do {
        count=length-offset;if(count>7)count=7;
        packet[0]=body[0];memcpy(packet+1,body+offset,count);
        h.IDE=CAN_ID_EXT;h.ExtId=((uint32_t)axis<<8)|index++;h.RTR=CAN_RTR_DATA;h.DLC=(uint32_t)count+1;
        if(HAL_CAN_AddTxMessage(&hcan2,&h,packet,&mailbox)!=HAL_OK)return 0;
        start=HAL_GetTick();while(HAL_CAN_IsTxMessagePending(&hcan2,mailbox)) {
            if(HAL_GetTick()-start>=TX_TIMEOUT_MS) {(void)HAL_CAN_AbortTxRequest(&hcan2,mailbox);return 0;}
        }
        offset+=count;
        if(offset<length) HAL_Delay(2);
    }while(offset<length);
    return 1;
}
static void play_stop(void)
{
    const uint8_t stop[]={0xFE,0x98,0,0x6B};uint8_t i;int ok=1;playing=0;play_rate=1;memset(play_speed_override,0,sizeof(play_speed_override));
    for(i=1;i<=6;++i)if(!play_send(i,stop,4))ok=0;
    text(ok?"PLAY_STOPPED tx_complete=1 enabled_unchanged=1\r\n":"PLAY_STOP_UNCONFIRMED retain support\r\n");
}
static int play_position(uint8_t axis,int32_t value,uint8_t sync)
{
    uint32_t m=value<0?(uint32_t)(-(int64_t)value):(uint32_t)value;uint16_t speed=play_speed_override[axis-1]?play_speed_override[axis-1]:play_speed_tenths_rpm[axis-1]*play_rate;
    uint8_t b[11]={0xFB,value<0?1U:0U,(uint8_t)(speed>>8),(uint8_t)speed,0,0,0,0,1,sync,0x6B};
    if(position_scaled[axis-1]) {if(m>UINT32_MAX/10U)return 0;m*=10;}
    b[4]=(uint8_t)(m>>24);b[5]=(uint8_t)(m>>16);b[6]=(uint8_t)(m>>8);b[7]=(uint8_t)m;
    return play_send(axis,b,11);
}
static int current_signed(uint8_t axis,int32_t *v)
{
    AxisFeedback *f=&feedback[axis-1];read_position_mode(axis,1);
    if(!f->valid||f->position.magnitude_tenths_deg>INT32_MAX)return 0;
    *v=f->position.negative?-(int32_t)f->position.magnitude_tenths_deg:(int32_t)f->position.magnitude_tenths_deg;return 1;
}
static void play_arm(void)
{
    const uint8_t opt[]={0x1A,0x6B},state[]={0x3A,0x6B},enable[]={0xF3,0xAB,1,0,0x6B};
    uint8_t i,f,reply;int32_t pos[6],after;
    if(teaching||playing) {text("PLAY_REJECT finish teaching first\r\n");return;}
    for(i=1;i<=6;++i) {
        if(!motor_exchange(i,opt,2,0x1A,&f)||(f&2U)||!(f&4U)) {text("PLAY_REJECT X closed-loop options unconfirmed\r\n");return;}
        position_scaled[i-1]=(f&0x80U)!=0;
        if(!motor_exchange(i,state,2,0x3A,&f)||(f&0x0CU)||!current_signed(i,&pos[i-1])) {text("PLAY_REJECT fault or position unconfirmed\r\n");return;}
    }
    play_stop();
    for(i=1;i<=6;++i) {
        (void)motor_exchange(i,enable,5,0xF3,&reply);
        if(!motor_exchange(i,state,2,0x3A,&f)||!(f&1U)||!play_position(i,pos[i-1],0)
           ||!current_signed(i,&after)||(int64_t)after-pos[i-1]>20||(int64_t)pos[i-1]-after>20) {
            play_stop();text("PLAY_REJECT enable or hold changed position\r\n");return;
        }
    }
    playing=1;play_heartbeat=HAL_GetTick();text("PLAY_READY speed_joint_deg_s_max=1 lease_ms=3000\r\n");
}
static void play_step(const char *line,int quiet)
{
    const uint8_t q[]={0x3A,0x6B},go[]={0xFF,0x66,0x6B};long parsed[6];int32_t target[6],actual[6],current;uint8_t i,f;char tail;
    if(!playing) {text("PLAY_REJECT not armed\r\n");return;}
    if(sscanf(line,"%ld %ld %ld %ld %ld %ld %c",&parsed[0],&parsed[1],&parsed[2],&parsed[3],&parsed[4],&parsed[5],&tail)!=6) {
        play_stop();text("PLAY_REJECT malformed target\r\n");return;
    }
    for(i=1;i<=6;++i) {
        if(parsed[i-1]<INT32_MIN||parsed[i-1]>INT32_MAX) {play_stop();text("PLAY_REJECT integer range\r\n");return;}
        target[i-1]=(int32_t)parsed[i-1];
        if(!motor_exchange(i,q,2,0x3A,&f)||!(f&1U)||(f&0x0CU)||!current_signed(i,&current)
           ||(int64_t)target[i-1]-current>play_step_tenths_deg[i-1]*play_rate
           ||(int64_t)current-target[i-1]>play_step_tenths_deg[i-1]*play_rate) {
            play_stop();text("PLAY_REJECT state or bounded-step check failed\r\n");return;
        }
        actual[i-1]=current;
    }
    for(i=1;i<=6;++i) {
        X42S_Frame frame;uint32_t start;int accepted=0;
        if(!play_position(i,target[i-1],1)){play_stop();text("PLAY_REJECT CAN target TX failed\r\n");return;}
        start=HAL_GetTick();while(HAL_GetTick()-start<30U) if(receive(&frame)) {
            if(!quiet)raw("PLAY_REPLY",&frame);
            if(frame.id==((uint32_t)i<<8)&&frame.extended&&frame.len==3U&&frame.data[0]==0xFB&&frame.data[2]==0x6B) {
                if(frame.data[1]==0x02){accepted=1;break;}
                if(frame.data[1]==0xE2||frame.data[1]==0xEE)break;
            }
        }
        if(!accepted){play_stop();text("PLAY_REJECT target acceptance unconfirmed\r\n");return;}
    }
    if(!play_send(0,go,3)) {play_stop();return;}
    {X42S_Frame frame;uint32_t start=HAL_GetTick();int rejected=0;
        while(HAL_GetTick()-start<(quiet?3U:20U)) if(receive(&frame)) {
            if(!quiet)raw("PLAY_REPLY",&frame);
            if(frame.extended&&frame.len==3U&&(frame.data[0]==0xFB||frame.data[0]==0xFF)
               &&(frame.data[1]==0xE2||frame.data[1]==0xEE)) rejected=1;
        }
        if(rejected){play_stop();text("PLAY_REJECT motor command rejected\r\n");return;}
    }
    play_heartbeat=HAL_GetTick();
    if(quiet){char row[180];snprintf(row,sizeof(row),"PLAY_STREAM_QUEUED,%lu,%ld,%ld,%ld,%ld,%ld,%ld\r\n",(unsigned long)play_heartbeat,(long)actual[0],(long)actual[1],(long)actual[2],(long)actual[3],(long)actual[4],(long)actual[5]);text(row);}
    else text("PLAY_STEP_QUEUED verify encoder arrival\r\n");
}
static void play_state(void)
{
    int32_t position[6]={0};uint8_t i,mask=0;char row[160];int n;
    for(i=1;i<=6;++i)if(current_signed(i,&position[i-1]))mask|=(uint8_t)(1U<<(i-1));
    n=snprintf(row,sizeof(row),"PLAY_POSITION,%u",mask);
    for(i=0;i<6;++i){if(mask&(1U<<i))n+=snprintf(row+n,sizeof(row)-(size_t)n,",%ld",(long)position[i]);else n+=snprintf(row+n,sizeof(row)-(size_t)n,",NA");}
    snprintf(row+n,sizeof(row)-(size_t)n,"\r\n");text(row);
}
static void play_check(void)
{
    const uint8_t q[]={0x1A,0x6B};uint8_t i,f;char row[110];int ok=1;
    for(i=1;i<=6;++i){
        if(!motor_exchange(i,q,2,0x1A,&f)){snprintf(row,sizeof(row),"PLAY_OPTIONS J%u valid=0\r\n",i);ok=0;}
        else {snprintf(row,sizeof(row),"PLAY_OPTIONS J%u valid=1 flags=0x%02X type=%s closed=%u scale10=%u\r\n",i,f,(f&2)?"Emm":"X",(f&4)!=0,(f&0x80)!=0);if((f&2)||!(f&4))ok=0;}
        text(row);
    }
    text(ok?"PLAY_CHECK_OK\r\n":"PLAY_CHECK_FAILED\r\n");
}
static void command(const char *line)
{
    uint8_t i;
    if(!strcmp(line,"ID?")||!strcmp(line,"HELP")) {
        text("ID x42s-final-v3.1-absread MCU=F407VG CAN2=500K motion=BOUNDED_REPLAY model=UNHOMED grip=CAN_NODE7 UART3_STATUS_ONLY\r\n");
        text("COMMANDS: ENC 1..6 | READ 1..6 | SCAN | STATE | GRIP? | TEACH BEGIN/START/SAMPLE/STOP | MOTOR STATE | PLAY ARM/STEP/PING/STOP\r\n");
    } else if(!strcmp(line,"PLAY CHECK"))play_check();
    else if(!strcmp(line,"PLAY STATE"))play_state();
    else if(!strcmp(line,"PLAY ARM"))play_arm();
    else if(!strcmp(line,"PLAY RATE 1")||!strcmp(line,"PLAY RATE 7")||!strcmp(line,"PLAY RATE 10")) {
        if(!playing){text("PLAY_REJECT not armed\r\n");return;}
        play_rate=!strcmp(line,"PLAY RATE 10")?10U:(uint8_t)(line[10]-'0');
        text(play_rate==10?"PLAY_RATE_OK multiplier=10\r\n":(play_rate==7?"PLAY_RATE_OK multiplier=7\r\n":"PLAY_RATE_OK multiplier=1\r\n"));
    }
    else if(!strncmp(line,"PLAY SPEED ",11)) {
        unsigned long v[6];char tail;const uint16_t limits[]={1750,1500,1750,500,500,40};
        if(!playing){text("PLAY_REJECT not armed\r\n");return;}
        if(sscanf(line+11,"%lu %lu %lu %lu %lu %lu %c",&v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&tail)!=6){play_stop();text("PLAY_REJECT speed format\r\n");return;}
        for(i=0;i<6;++i)if(!v[i]||v[i]>limits[i]){play_stop();text("PLAY_REJECT speed range\r\n");return;}
        for(i=0;i<6;++i)play_speed_override[i]=(uint16_t)v[i];
        text("PLAY_SPEED_OK\r\n");
    }
    else if(!strcmp(line,"PLAY STOP"))play_stop();
    else if(!strcmp(line,"PLAY PING")) {
        if(playing) {play_heartbeat=HAL_GetTick();text("PLAY_ALIVE\r\n");}else text("PLAY_REJECT not armed\r\n");
    } else if(!strncmp(line,"PLAY STEP ",10))play_step(line+10,0);
    else if(!strcmp(line,"PLAY DIAG")) {
        const uint8_t codes[]={0x33,0x34,0x41};uint8_t k,v,q[2];
        for(k=0;k<3;++k){q[0]=codes[k];q[1]=0x6B;(void)motor_exchange(6,q,2,codes[k],&v);}
        text("PLAY_DIAG_DONE\r\n");
    }
    else if(!strncmp(line,"PLAY STREAM ",12))play_step(line+12,1);
    else if(strlen(line)==5 && !strncmp(line,"ENC ",4) && line[4]>='1' && line[4]<='6') {
        if(playing || teaching) text("ENC_REJECT active stream\r\n");
        else read_encoder((uint8_t)(line[4]-'0'));
    }
    else if(!strcmp(line,"MOTOR STATE"))(void)motor_states();
    else if(!strcmp(line,"TEACH BEGIN"))teach_begin();
    else if(!strcmp(line,"TEACH START")) {
        if(playing){text("TEACH_REJECT playback active\r\n");return;}
        teaching=1;teach_sequence=0;text("TEACH_STARTED unit=motor_tenths_deg model=UNHOMED\r\n");
    } else if(!strcmp(line,"TEACH SAMPLE"))teach_sample();
    else if(!strcmp(line,"TEACH STOP")) {teaching=0;text("TEACH_STOPPED\r\n");}
    else if(!strcmp(line,"GRIP CAN?")) {
        Gripper_QueryCAN();
    }
    else if(!strncmp(line,"GRIP ",5)) {
        if(!Gripper_CANCommand(line+5))text("GRIP_REJECT unsupported command\r\n");
    }
    else if(!strcmp(line,"GRIP?"))Gripper_QueryStatus();
    else if(strlen(line)==6&&!strncmp(line,"READ ",5)&&line[5]>='1'&&line[5]<='6')read_position((uint8_t)(line[5]-'0'));
    else if(!strcmp(line,"SCAN")) {for(i=1;i<=6;++i)read_position(i);text("SCAN_DONE\r\n");}
    else if(!strcmp(line,"STATE")){for(i=1;i<=6;++i)print_position(i,&feedback[i-1]);}
    else if(!strcmp(line,"LISTEN"))listen();
    else text("REJECT unsupported command\r\n");
}
void ReadCAN_Run(void)
{
    char line[LINE_CAPACITY];uint32_t last_byte_ms=0;uint8_t byte,discard=0;size_t used=0;
    text("BOOT x42s-final-v3.1-absread READY no startup query, release or motion\r\n");
    for(;;) {
        if(playing&&HAL_GetTick()-play_heartbeat>3000U){play_stop();text("PLAY_WATCHDOG\r\n");}
        if((used||discard)&&HAL_GetTick()-last_byte_ms>1000U){used=0;discard=0;text("REJECT incomplete line\r\n");}
        if(HAL_UART_Receive(&huart1,&byte,1,0)!=HAL_OK)continue;
        last_byte_ms=HAL_GetTick();if(byte=='\r')continue;
        if(byte=='\n'){if(!discard&&used){line[used]=0;command(line);}used=0;discard=0;}
        else if(!discard){if(used>=sizeof(line)-1||byte<32||byte>126){used=0;discard=1;text("REJECT invalid line\r\n");}else line[used++]=(char)byte;}
    }
}
