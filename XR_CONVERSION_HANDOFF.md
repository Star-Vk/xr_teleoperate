# XR 数据转换交接

这份文档给下一位 Codex 使用，目标是只处理 `xr_teleoperate` 里的 XR 数据离线转换，不再混入本次对话里其他调试背景。

## 当前结论

模型文件，urdf文件所在目录：~/workspace/qyz1/zqz1_ws/urdf/QingYun_Robot_Description/qingyun_z1_A_rev_1_0_description

- 转换工具已经迁移到当前项目 `xr_teleoperate`
- 新链路优先读取：
  - `tele_data.left_robot_world_pose`
  - `tele_data.right_robot_world_pose`
- 旧字段仍保留回退兼容：
  - `tele_data.left_wrist_pose`
  - `tele_data.right_wrist_pose`
- `xr_button_test_001/episode_0001` 已经成功跑通“读取 JSONL -> 转成 CSV”整条链路
- 但当前输出的 CSV 明显不对，关节角幅值非常小，且大量 IK 失败
- 这个“小角度 / 结果不对”是当前遗留问题，下一位 Codex 需要继续处理

## 关键文件

转换入口：

- [tools/xr_to_action_csv.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_to_action_csv.py)

桥接 JSONL 解析：

- [tools/xr_bridge_export.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_bridge_export.py)
- [tools/xr_retarget/bridge.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/bridge.py)

retarget / IK 主逻辑：

- [tools/xr_retarget/retarget.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/retarget.py)
- [tools/xr_retarget/urdf_kinematics.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/urdf_kinematics.py)
- [tools/xr_retarget/csv_export.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/csv_export.py)
- [tools/xr_retarget/math_utils.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/math_utils.py)

默认配置：

- [config/xr_retarget.yaml](/home/starvk/workspace/qyz1/xr_teleoperate/config/xr_retarget.yaml)

当前说明文档：

- [XR_RETARGET_GUIDE.md](/home/starvk/workspace/qyz1/xr_teleoperate/XR_RETARGET_GUIDE.md)

## 输入数据格式

当前工具面向 `xr_teleoperate` 录制出来的 `frames.jsonl`。

每行至少需要：

```json
{
  "timestamp_ms": 1774423307479,
  "tele_data": {
    "left_robot_world_pose": [[... 4x4 ...]],
    "right_robot_world_pose": [[... 4x4 ...]]
  }
}
```

推荐同时存在这些字段，但当前转换并不强依赖它们：

- `tele_data.head_pose_valid`
- `tele_data.left_arm_pose_valid`
- `tele_data.right_arm_pose_valid`

桥接解析逻辑当前优先级：

1. `left_robot_world_pose` / `right_robot_world_pose`
2. `left_wrist_pose` / `right_wrist_pose`

其中 `left_robot_world_pose/right_robot_world_pose` 是当前推荐主输入，因为它们是：

- 机器人坐标轴
- world 原点
- 不减 head
- 不加 waist 偏移

更适合离线 retarget。

## CSV 输出格式

输出 CSV 格式固定为：

```text
frame,can_iface,motor_id,position_rad,elapsed_ms,speed_rad_s,accel_rad_s2
```

当前电机顺序：

- 左臂 `can2`: `3, 4, 5, 6`
- 右臂 `can3`: `13, 14, 15, 16`

因此每个导出动作帧应对应 8 行。

## 当前已经做过的改动

### 1. 工具从旧项目迁到本项目

旧项目参考来源是兄弟目录 `../zqz1_ws` 下的旧工具链，已迁到当前仓库。

### 2. 解析逻辑已适配新字段

[tools/xr_retarget/bridge.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/bridge.py) 已修改为优先读取：

- `left_robot_world_pose`
- `right_robot_world_pose`

### 3. 默认标定策略已改

[config/xr_retarget.yaml](/home/starvk/workspace/qyz1/xr_teleoperate/config/xr_retarget.yaml) 当前默认配置：

- `mode: per_arm_average`
- `rotation_rpy_deg: [0, 0, 0]`
- `frame_count: 20`

含义：

- 不再套旧的 `-90°` 旋转
- 默认只拿录制开头 `20` 帧做标定
- 左右臂分别估计自己的平移

### 4. retarget 代码已支持 per-arm translation

[tools/xr_retarget/retarget.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_retarget/retarget.py) 里已经加入：

- `CalibrationConfig.default_frame_count`
- `RetargetSummary.left_translation_xyz`
- `RetargetSummary.right_translation_xyz`
- `per_arm_average` 标定模式
- 每只手单独 `arm_translations`

### 5. CLI 默认会自动截取标定窗口

[tools/xr_to_action_csv.py](/home/starvk/workspace/qyz1/xr_teleoperate/tools/xr_to_action_csv.py) 里已实现：

- 如果用户不手动传 `--calib-start-frame/--calib-end-frame`
- 就自动使用配置中的前 `frame_count` 帧作为标定窗口

## 已验证的数据

可用于新链路验证的数据：

- [frames.jsonl](/home/starvk/workspace/qyz1/xr_teleoperate/teleop/utils/data/xr_button_test_001/episode_0001/frames.jsonl)

这个 episode 已确认：

- 总帧数 `529`
- `left_robot_world_pose` 存在
- `right_robot_world_pose` 存在
- `head_pose_valid / left_arm_pose_valid / right_arm_pose_valid` 存在

旧数据：

- [frames.jsonl](/home/starvk/workspace/qyz1/xr_teleoperate/teleop/utils/data/pico_capture_001/episode_0007/frames.jsonl)

这类旧 episode 大概率没有 `left_robot_world_pose/right_robot_world_pose`，不能直接当新链路主验证样本。

## 已执行的转换命令

```bash
cd /home/starvk/workspace/qyz1/xr_teleoperate
python3 tools/xr_to_action_csv.py \
  --input teleop/utils/data/xr_button_test_001/episode_0001/frames.jsonl \
  --config config/xr_retarget.yaml \
  --output output/xr_button_test_001_episode_0001.csv
```

当前输出文件：

- [xr_button_test_001_episode_0001.csv](/home/starvk/workspace/qyz1/xr_teleoperate/output/xr_button_test_001_episode_0001.csv)

## 当前转换结果

最近一次实际转换输出摘要：

- 输入帧数：`529`
- 导出动作帧：`3`
- CSV 总行数：`24`
- 标定窗口：`0..19`
- `left_failures=473`
- `right_failures=473`

转换日志里打印过：

- `failure_samples=left=[56,57,58,59,60,61,62,63,64,65]`
- `failure_samples=right=[56,57,58,59,60,61,62,63,64,65]`

说明大约从第 `56` 帧开始，IK 大量失败。

## 当前遗留问题

### 1. CSV 明显不对

当前导出的 CSV 角度值非常小，例如：

- `-0.027278`
- `0.025136`

这明显不像一段正常的大幅手柄动作应得到的动作库结果。

### 2. IK 大量失败

虽然链路已经不是“529/529 全失败”了，但仍然存在：

- 前几十帧还能部分求解
- 后续大部分帧超出当前模型工作空间或标定不正确
- 导出器最后只保留了极少数动作帧

### 3. 当前 CSV 只能证明链路通了，不能证明结果正确

所以必须明确：

- 当前项目里的“数据读取 / 转换命令 / CSV 导出”已经接通
- 但“转换出来的动作角是否正确”还没有解决

## 下一位 Codex 建议优先做的事

1. 先不要再迁文件了，重点只看现有 `xr_teleoperate/tools/xr_retarget/*`
2. 用 [frames.jsonl](/home/starvk/workspace/qyz1/xr_teleoperate/teleop/utils/data/xr_button_test_001/episode_0001/frames.jsonl) 继续做最小复现
3. 检查 `left_robot_world_pose/right_robot_world_pose` 到机器人基座系的映射是否仍然差一个固定外参
4. 检查当前 `tip_link`、`tool_offset_xyz`、关节方向、home pose 是否和真实机器人一致
5. 重点确认“角度太小”到底是：
   - 标定平移仍不对
   - 输入 pose 尺度/坐标系不对
   - IK 目标被夹到可达域边界
   - 导出压缩策略把变化吃掉了
6. 在修复前，不要把当前 CSV 当作可执行动作库

## 给下一位 Codex 的一句话总结

当前 `xr_teleoperate` 已经有独立的 XR 离线转换工具，能把新录制的 `left_robot_world_pose/right_robot_world_pose` 读进来并导出动作 CSV；但最近一次实际转换出来的关节角幅值明显过小，而且存在大量 IK 失败，所以“链路已通，结果未正确”是当前状态。
