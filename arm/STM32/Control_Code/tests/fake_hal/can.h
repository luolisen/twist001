#ifndef TEST_CAN_H
#define TEST_CAN_H
#include <stdint.h>
#define CAN_RX_FIFO0 0U
#define CAN_RX_FIFO1 1U
#define CAN_ID_STD 0U
#define CAN_ID_EXT 4U
#define CAN_RTR_DATA 0U
#define CAN_RTR_REMOTE 2U
#define DISABLE 0U
#define HAL_OK 0U
#define HAL_TIMEOUT 3U
typedef struct {uint32_t ESR;} CAN_Regs;
typedef struct {CAN_Regs *Instance;} CAN_HandleTypeDef;
typedef struct {uint32_t StdId,ExtId,IDE,RTR,DLC,TransmitGlobalTime;} CAN_TxHeaderTypeDef;
typedef CAN_TxHeaderTypeDef CAN_RxHeaderTypeDef;
extern CAN_HandleTypeDef hcan2;
uint32_t HAL_GetTick(void);
void HAL_Delay(uint32_t ms);
uint32_t HAL_CAN_GetError(CAN_HandleTypeDef *);
uint32_t HAL_CAN_GetRxFifoFillLevel(CAN_HandleTypeDef *,uint32_t);
int HAL_CAN_GetRxMessage(CAN_HandleTypeDef *,uint32_t,CAN_RxHeaderTypeDef *,uint8_t *);
int HAL_CAN_AddTxMessage(CAN_HandleTypeDef *,CAN_TxHeaderTypeDef *,uint8_t *,uint32_t *);
uint32_t HAL_CAN_IsTxMessagePending(CAN_HandleTypeDef *,uint32_t);
int HAL_CAN_AbortTxRequest(CAN_HandleTypeDef *,uint32_t);
#endif
