# XR 离线动作库生成指南

这套工具链用于把 `xr_teleoperate` 录制出来的 `frames.jsonl` 直接转换成当前项目可消费的动作库 CSV。

当前版本会优先读取这些字段：

- `left_robot_world_pose`
- `right_robot_world_pose`

如果新字段不存在，才回退到旧字段：

- `left_wrist_pose`
- `right_wrist_pose`

## 工具组成

- [`tools/xr_bridge_export.py`](tools/xr_bridge_export.py)
  - XR bridge JSONL 校验和示例生成
- [`tools/xr_to_action_csv.py`](tools/xr_to_action_csv.py)
  - 离线转换器
- [`config/xr_retarget.yaml`](config/xr_retarget.yaml)
  - retarget 默认配置

## 快速校验 JSONL

```bash
python tools/xr_bridge_export.py validate /path/to/frames.jsonl
```

## 离线转换命令

```bash
python tools/xr_to_action_csv.py \
  --input /path/to/frames.jsonl \
  --config config/xr_retarget.yaml \
  --output /path/to/xr_demo.csv
```

只导出左臂：

```bash
python tools/xr_to_action_csv.py \
  --input /path/to/frames.jsonl \
  --config config/xr_retarget.yaml \
  --arms left \
  --output /path/to/xr_left_arm.csv
```

当前 `xr_teleoperate` 默认配置面向新录制的 `left_robot_world_pose/right_robot_world_pose` 数据：

- 不再对输入位置额外施加旧的 `-90°` 平面旋转
- 默认只使用录制开头 `20` 帧做标定
- 左右臂分别估计各自的 `world -> robot base` 平移

这要求录制开始后的前一小段保持中性姿态，便于工具用作标定窗口。

## 当前默认 URDF

当前配置默认指向你工作区里的 QingYun URDF：

- `../../zqz1_ws/urdf/QingYun_Robot_Description/qingyun_z1_A_rev_1_0_description/urdf/qingyun_z1_A_rev_1_0.urdf`

如果以后路径变了，可以直接在命令行里覆盖：

```bash
python tools/xr_to_action_csv.py \
  --input /path/to/frames.jsonl \
  --config config/xr_retarget.yaml \
  --urdf /abs/path/to/robot.urdf \
  --output /path/to/xr_demo.csv
```

## 输出格式

输出 CSV 格式：

```text
frame,can_iface,motor_id,position_rad,elapsed_ms,speed_rad_s,accel_rad_s2
```

## 建议

新录制的数据优先使用 `left_robot_world_pose/right_robot_world_pose`。

因为它们是：

- 机器人坐标轴
- 世界系原点
- 不减头
- 不加腰部偏移

更适合做离线 retarget 和动作库生成。
