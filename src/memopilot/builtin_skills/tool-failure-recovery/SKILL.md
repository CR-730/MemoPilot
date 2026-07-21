---
name: tool-failure-recovery
description: 在工具失败或结果不确定时，基于真实观察决定重试、停止或向用户说明
background_allowed: false
always: true
---
工具报错时，把错误结果作为本轮证据继续判断；仅在调用本身可安全重试时重试。
如果外部副作用是否发生无法确认，不要盲目重放，应明确说明不确定性并停止自动执行。
