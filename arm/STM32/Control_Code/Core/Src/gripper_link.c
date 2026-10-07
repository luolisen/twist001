#include "gripper_link.h"
#include "usart.h"
#include <string.h>
#include <stdio.h>
#define GRIP_LINE_SIZE 1024U
#define GRIP_TIMEOUT_MS 500U
static char line[GRIP_LINE_SIZE];
static void host(const char *s) {
    (void)HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)strlen(s), 200U);
}
void Gripper_QueryStatus(void) {
    uint8_t byte;
    unsigned int count;
    size_t used = 0U;
    int discard = 0;
    uint32_t start;
    unsigned int received = 0U;
    char preview[65] = {0};
    char diagnostic[160];
    /* Bound the stale-data drain; refuse a continuously busy link. */
    for (count = 0; count < 2048U; ++count) {
        if (HAL_UART_Receive(&huart3, &byte, 1U, 0U) != HAL_OK) break;
    }
    if (count == 2048U) { host("GRIP_RX_BUSY\r\n"); return; }
    if (HAL_UART_Transmit(&huart3, (uint8_t *)"S\n", 2U, 50U) != HAL_OK) {
        host("GRIP_TX_ERROR\r\n"); return;
    }
    start = HAL_GetTick();
    while (HAL_GetTick() - start < GRIP_TIMEOUT_MS) {
        if (HAL_UART_Receive(&huart3, &byte, 1U, 1U) != HAL_OK) continue;
        if (received < sizeof(preview)-1U) preview[received] = (byte >= 32U && byte <= 126U) ? (char)byte : '.';
        ++received;
        if (byte == '\r') continue;
        if (byte == '\n') {
            line[used] = '\0';
            if (!discard && strncmp(line, "version=v2.2.3-DENGFOC-STM32-RO ", 32U) == 0
                && strstr(line, " gap_mm=") && strstr(line, " enabled=")
                && strstr(line, " i2c_error=")) {
                host("GRIP_STATUS "); host(line); host("\r\n"); return;
            }
            used = 0U; discard = 0;
        } else if (!discard) {
            if (byte < 32U || byte > 126U || used >= sizeof(line)-1U) discard = 1;
            else line[used++] = (char)byte;
        }
    }
    (void)snprintf(diagnostic, sizeof(diagnostic), "GRIP_DIAG rx_bytes=%u preview=%s\r\n", received, preview);
    host(diagnostic);
    host("GRIP_TIMEOUT no valid status; feedback invalid\r\n");
}
