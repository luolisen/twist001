cmake_minimum_required(VERSION 3.20)
project(ReadCAN LANGUAGES C ASM)
set(CMAKE_C_STANDARD 99)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(readcan.elf
  Core/Src/gripper_link.c
  Core/Src/gripper_can.c
  Core/Src/main.c
  Core/Src/can.c
  Core/Src/usart.c
  Core/Src/stm32f4xx_it.c
  Core/Src/stm32f4xx_hal_msp.c
  Core/Src/x42s_protocol.c
  Core/Src/readcan_app.c
  Core/Src/system_stm32f4xx.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_can.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_rcc.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_rcc_ex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash_ex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash_ramfunc.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_gpio.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_dma_ex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_dma.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_pwr.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_pwr_ex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_cortex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_exti.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_i2c.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_i2c_ex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_tim.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_tim_ex.c
  Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_uart.c
  Drivers/CMSIS/Device/ST/STM32F4xx/Source/Templates/gcc/startup_stm32f407xx.s
)
target_include_directories(readcan.elf PRIVATE
  Core/Inc Drivers/STM32F4xx_HAL_Driver/Inc Drivers/STM32F4xx_HAL_Driver/Inc/Legacy
  Drivers/CMSIS/Include Drivers/CMSIS/Device/ST/STM32F4xx/Include
)
target_compile_definitions(readcan.elf PRIVATE USE_HAL_DRIVER STM32F407xx)
target_compile_options(readcan.elf PRIVATE
  -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
  -Og -g3 -ffunction-sections -fdata-sections
  $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra>
)
set(LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/STM32F407VG_FLASH.ld")
set_target_properties(readcan.elf PROPERTIES LINK_DEPENDS "${LINKER_SCRIPT}")
target_link_options(readcan.elf PRIVATE
  -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
  -T${LINKER_SCRIPT} --specs=nano.specs --specs=nosys.specs
  -Wl,--gc-sections -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/readcan.map
  -Wl,--print-memory-usage
)
add_custom_command(TARGET readcan.elf POST_BUILD
  COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:readcan.elf> ${CMAKE_CURRENT_BINARY_DIR}/readcan.hex
  COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:readcan.elf> ${CMAKE_CURRENT_BINARY_DIR}/readcan.bin
  COMMAND ${CMAKE_SIZE} $<TARGET_FILE:readcan.elf>
)
