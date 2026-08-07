---
layout: post
title: 最常用的 Linux 命令的 Windows 对应命令
date: 2026-04-02 17:10:00
description: 记录最常用的 Linux 命令的 Windows 对应命令。
tags: [Windows, PowerShell, 命令行]
categories: [开发工具]
source_type: repost
giscus_comments: false
related_posts: false
pretty_table: true

toc:
  beginning: true
---

平时在 Linux 和 Windows 之间切换时，最头疼的通常不是“有没有工具”，而是“同一件事在另一边该怎么敲命令”。  
这篇做一个常用场景速查，先给出 Windows `cmd` 对照，再补充更原生的 PowerShell 写法。

## 一、常用命令速查表

| Linux                                                                                | Windows (cmd)                                                                                                         | Windows (PowerShell)                                                                                                  | 用途                                |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `lsof -i \| grep https`                                                              | `netstat -n -a \| findstr "https"`                                                                                    | `netstat -n -a \| Select-String "https"`                                                                              | 过滤网络连接信息                    |
| `ssh root@remote "tcpdump -s 0 -U -n -w - -i eth0 not port 22" \| wireshark -k -i -` | `ssh root@remote "tcpdump -s 0 -U -n -w - -i eth0 not port 22" \| "C:\Program Files\Wireshark\wireshark.exe" -k -i -` | `ssh root@remote "tcpdump -s 0 -U -n -w - -i eth0 not port 22" \| "C:\Program Files\Wireshark\wireshark.exe" -k -i -` | 远端抓包并本地 Wireshark 实时看流量 |
| `netstat -tulpn`                                                                     | `netstat -ano`                                                                                                        | `Get-NetTCPConnection -State Listen \| Select-Object LocalAddress,LocalPort,OwningProcess`                            | 查看端口监听与 PID                  |
| `cat filename.txt`                                                                   | `type filename.txt`                                                                                                   | `Get-Content filename.txt`                                                                                            | 在终端直接查看文件内容              |
| `ls -la`                                                                             | `dir /a`                                                                                                              | `Get-ChildItem -Force`                                                                                                | 列出目录内容（含隐藏文件）          |
| `find / -name "config.txt"`                                                          | `dir config.txt /s`                                                                                                   | `Get-ChildItem -Path . -Filter "config.txt" -Recurse -File`                                                           | 递归查找文件                        |
| `ifconfig` / `ip a`                                                                  | `ipconfig /all`                                                                                                       | `Get-NetIPConfiguration`                                                                                              | 查看网卡和 IP 配置                  |
| `top` / `ps aux`                                                                     | `tasklist`                                                                                                            | `Get-Process`                                                                                                         | 查看进程列表                        |
| `kill -9 1234`                                                                       | `taskkill /F /PID 1234`                                                                                               | `Stop-Process -Id 1234 -Force`                                                                                        | 按 PID 强制结束进程                 |
| `traceroute google.com`                                                              | `tracert google.com`                                                                                                  | `Test-NetConnection google.com -TraceRoute`                                                                           | 路由追踪                            |
| `clear`                                                                              | `cls`                                                                                                                 | `Clear-Host`                                                                                                          | 清屏                                |

## 二、按场景展开

### 1) 过滤命令输出

Linux 常用 `grep`，在 `cmd` 里一般用 `findstr`：

```bash
netstat -n -a | findstr "https"
```

注意关键字建议带双引号，避免空格或特殊字符导致匹配异常。

### 2) 远端抓包并在本机 Wireshark 实时查看

如果你有 Linux 服务器权限，可以直接把 `tcpdump` 输出通过 SSH 管道传给本地 Wireshark：

```bash
ssh root@remote-linux "tcpdump -s 0 -U -n -w - -i eth0 not port 22" | "C:\Program Files\Wireshark\wireshark.exe" -k -i -
```

这里路径有空格，所以需要用双引号包住 `wireshark.exe` 路径。

### 3) 查看端口监听对应进程

```bash
netstat -ano
```

`-o` 会带出 PID，拿到 PID 后可以在任务管理器或配合 `tasklist` 进一步定位进程。

### 4) 递归查找文件

```bash
dir config.txt /s
```

这会在当前目录（及其子目录）递归查找文件。  
如果想从某个盘根目录开始，可先切到该盘符再执行（例如 `C:`）。

## 三、PowerShell 常用一行命令

如果你主要使用 PowerShell，可以直接记这几条：

| 命令                                                                                       | 一句话说明                             |
| ------------------------------------------------------------------------------------------ | -------------------------------------- |
| `Get-NetTCPConnection -State Listen \| Select-Object LocalAddress,LocalPort,OwningProcess` | 查看当前监听端口及对应进程 PID。       |
| `Get-ChildItem -Path . -Filter "*.log" -Recurse -File`                                     | 递归查找当前目录下所有 `.log` 文件。   |
| `Get-Process \| Sort-Object CPU -Descending \| Select-Object -First 10`                    | 按 CPU 使用量排序并显示前 10 个进程。  |
| `Stop-Process -Id 1234 -Force`                                                             | 强制结束 PID 为 `1234` 的进程。        |
| `Test-NetConnection github.com -TraceRoute`                                                | 测试到目标主机的连通性并输出路由路径。 |

## 四、一些实用提醒

- Windows 10/11 默认已内置 OpenSSH 客户端，可直接使用 `ssh`。
- `cmd` 与 PowerShell 命令生态不同，前者偏“经典命令”，后者偏“对象管道”。
- 网络排查场景常用组合是：`netstat -ano` + `tasklist` + `taskkill`。

## 参考

- 原文：[The Windows equivalents of the most used Linux commands](https://techkettle.blogspot.com/2026/04/the-windows-equivalents-of-most-used.html)
