# 2026-10-08 仓库更新验证

本次更新 STM32、ESP32 与 REV6 最新双相机仿真。safe、code、三模型训练归档及历史 PCB 文件未修改。

| 项目 | 本次结果 | 边界 |
| --- | --- | --- |
| STM32F407VG 重新构建 | 成功，FLASH 24,940 B、RAM 6,464 B | GCC 15.3 工具链；保留 HAL 未使用参数及 nosys 链接警告 |
| ESP32 重新构建 | 成功，应用 1,710,674 B，全局 RAM 72,524 B | esp32 core 3.3.11，huge_app 3 MB；临时随机测试密钥，构建产物不发布 |
| 固件六组回归 | 全部通过，ASan/UBSan | 协议、应用、夹爪链路、CAN、权限/租约与轨迹；无硬件访问 |
| REV6 基础及任务场景 | 加载、两路渲染、各 100 步数值烟测通过 | MuJoCo 3.13.0；不等于任务成功率或碰撞验收 |
| 相机配置 | 前置 48°，腕部 45.2°；640×360 全视野补边至 256/512 方图 | 实拍人工对齐候选，尚未完成光学校准 |
| 模型来源 | 与来源 XML 语义一致，仅序列化与资源路径变更 | ROS MuJoCo 使用同一资源；历史 URDF 单独保留 |
| ROS 2 | 静态资源与单项 MJCF 契约检查通过 | 未执行 ROS 2 构建和运行 |
| 烧录及实机 | 未执行 | 不改变设备上的现有固件 |

源码来源是 FinalIntegration v3.0。`source_manifest.json` 的 `sha256` 为原源文件 hash，`published_sha256` 为仓库文件 hash；唯一固件文本差异是 `can.c` 的行尾空白规范化。源工程的旧 README/manifest 保留为历史材料，以本记录为当前验证依据。

STM32 构建方法见 `arm/STM32/README.md`，ESP32 配置及构建见 `arm/ESP32/README.md`。macOS 联合回归：

```sh
bash arm/firmware_tools/test_firmware.sh
```

注意量程：真实夹爪 0–74.63 mm，模型 0–80 mm；适配、标定和真实闭环需独立完成。仓库未包含原设备的无线密钥、私有配置或构建二进制。
