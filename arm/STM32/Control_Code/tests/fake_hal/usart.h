#include "can.h"
typedef struct {int unused;} UART_HandleTypeDef;
extern UART_HandleTypeDef huart1;
extern UART_HandleTypeDef huart3;
int HAL_UART_Transmit(UART_HandleTypeDef *,uint8_t *,uint16_t,uint32_t);
int HAL_UART_Receive(UART_HandleTypeDef *,uint8_t *,uint16_t,uint32_t);
