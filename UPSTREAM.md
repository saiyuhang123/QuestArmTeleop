# Upstream and attribution

This repository is maintained by **syh** as a downstream adaptation of
[agilexrobotics/QuestArmTeleop](https://github.com/agilexrobotics/QuestArmTeleop).

The downstream work includes integration for a six-axis CANopen arm, guarded
joint-command output, joint-state bridging, fake-hardware launch paths, tests,
and related documentation. New downstream changes are maintained by syh
(`2906212119@qq.com`).

The original project history and attribution are retained. Copyright notices
and license headers embedded in individual upstream or third-party source files
must not be removed. The upstream repository does not currently contain a
repository-level `LICENSE` file, so this repository does not claim to replace or
broaden the rights granted by the original copyright holders. Obtain permission
or licensing clarification from the relevant copyright holder before public or
commercial redistribution when required.

Git remotes are intended to be arranged as follows:

- `origin`: `https://github.com/saiyuhang123/QuestArmTeleop.git`
- `upstream`: `https://github.com/agilexrobotics/QuestArmTeleop.git`

Upstream updates can be reviewed with `git fetch upstream` before they are
selectively merged into this maintained version.

# 上游来源与归属

本仓库由 **syh** 作为下游版本维护，原始项目为
[agilexrobotics/QuestArmTeleop](https://github.com/agilexrobotics/QuestArmTeleop)。

当前维护版新增内容主要包括：六轴 CANopen 机械臂适配、安全关节指令输出、
关节状态桥接、假硬件启动链路、测试及配套文档。新增的下游改动由 syh
（`2906212119@qq.com`）维护。

本仓库保留原项目的 Git 历史及归属信息。原项目代码和第三方文件中已有的版权
及许可证声明不得删除。由于上游仓库当前没有仓库级 `LICENSE` 文件，本仓库不
声明替代或扩大原版权持有人授予的权利；如需公开或商业再分发，应根据实际用途
向相关版权持有人确认授权。
