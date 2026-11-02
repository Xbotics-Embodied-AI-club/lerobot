# 第 1 章 板子上手

星禾套件的主从两条机械臂和两路相机都接在一块 RK3566 端侧板上。这块板子不接显示器，所有操作都在浏览器里完成：你只需要一台和板子在同一局域网、能打开网页的电脑，不用装任何软件。

板子上的网页服务端口连号，顺序是「配网 → 终端 → Jupyter → VS Code → 硬件检测 → 标定」，记住开头的 8001 就行：

| 网址 | 做什么 | 什么时候能打开 |
| --- | --- | --- |
| `http://10.42.0.1` | 网络设置：连 Wi-Fi、设网线、连训练电脑、看 IP | 连着板子的热点时 |
| `http://<板子IP>:8001` | 同一个网络设置页 | 板子开机就能打开 |
| `http://<板子IP>:8002` | 网页终端 | 服务开着时 |
| `http://<板子IP>:8003` | JupyterLab | 服务开着时 |
| `http://<板子IP>:8004` | 浏览器版 VS Code | 服务开着时 |
| `http://<板子IP>:8005` | 硬件检测 | 服务开着时 |
| `http://<板子IP>:8006` | 网页标定 | 服务开着时 |
| `http://<板子IP>:8007` | 推理控制台 | 运行 `lerobot-inference-console` 的时候 |
| `http://<板子IP>:6006` | 训练电脑上的 TensorBoard：训练曲线 | 板子连上训练电脑时，见 [1.4 节](#connect-server) |
| `http://<板子IP>:9090` | 录制预览：相机画面和关节曲线 | 录数据的时候 |

8002 到 8006 默认都开着，可以用 `xbotics-config` 开关，见 [1.7 节](#xbotics-config)。板子的用户名是 `root`，默认密码是 `123456`，网页终端、JupyterLab、VS Code 和 SSH 都用它。只在自己的局域网里用板子，不要把这些端口映射到公网。

## 1.1 烧写镜像

板子到手已经装好系统时，可以跳过这一节。新板装系统、重装系统、板子起不来要恢复，都照这一节做。

烧写会清掉板子上的全部内容，包括系统、录的数据和机械臂标定，要留的先拷走。

**准备：**

| 要准备的 | 说明 |
| --- | --- |
| 电脑 | Windows 10 / 11，至少一个 USB 口 |
| 烧写线 | USB-A 公对公数据线，用带屏蔽的 |
| 电源 | 板子的 12V 电源（DC 插头 5.5×2.5 mm） |
| 镜像 | 到 [Hugging Face](https://huggingface.co/Harrysunshine?search_datasets=rk3566) 下载最新版出厂镜像，用 7-Zip 解压成 `.img` 文件 |
| 驱动 | {download}`DriverAssitant_v5.13.zip <attachments/DriverAssitant_v5.13.zip>` |
| 烧写工具 | {download}`RKDevTool_v3.37_for_window.7z <attachments/RKDevTool_v3.37_for_window.7z>` |

镜像仓库是受限访问：用自己的 Hugging Face 账号登录，在页面上点申请，批准后才能下载。

**装驱动（每台电脑一次）：** 解压驱动，运行里面的 `DriverInstall.exe`。先点「驱动卸载」，等弹出「卸载驱动成功」；再点「驱动安装」，等弹出「安装驱动成功」。先卸再装，是因为电脑上别的版本的旧驱动会让工具认不出板子。

![解压后的文件夹，运行 DriverInstall.exe](images/flash-03-driver-open.png)

![点「驱动卸载」，等弹出「卸载驱动成功」](images/flash-04-driver-uninstall.png)

![点「驱动安装」，等弹出「安装驱动成功」](images/flash-05-driver-install.png)

**连接板子：** 解压烧写工具，运行 `RKDevTool.exe`，停在「下载镜像」页。然后拔掉腕部相机，因为烧写口就是平时插腕部相机的那个 USB 口（板子上沿左边那个）。烧写线一头插这个口，一头插电脑。

![烧写要用的三处：OTG 口、升级键、12V 电源](images/flash-01-ports.jpg)

进烧写模式有两种办法：

- **板子能正常开机**：在网页终端里敲 `reboot loader`
- **板子起不来，或是新板**：先断电，按住升级键不放，插上 12V 电源，等工具底部出现「发现一个LOADER设备」后再松手

**烧写：** 工具底部显示「发现一个LOADER设备」后，按下图设置，然后点「执行」：

- 第 2 行 `firmware`：勾上，地址保持 `0x00000000`，点这一行路径栏右边的按钮，选解压出来的 `.img`
- 第 1 行 `rkbin`：不勾
- 勾上「强制按地址写」

![「下载镜像」页：两行怎么勾，勾上「强制按地址写」](images/flash-02-rkdevtool.png)

右边的日志开始滚动，写完板子自动重启。中途不要拔线、不要断电，失败了就重新进烧写模式再烧一遍。

**第一次开机：** 插回腕部相机，上电。第一次开机会把系统分区扩到整片 eMMC，比平时多等一会儿。网线的出厂地址是固定 IP `192.168.1.50`，也可以连热点设置网络。板子连上外网后，照 1.2 节敲一遍一键安装，把板子上的东西升到最新。

## 1.2 一条命令装好，一条命令升级

板子上要用的全部东西都随 LeRobot 一起发布：系统设置、机械臂与相机的固定名字、六个网页服务、LeRobot 环境。新板子、刚烧完镜像的板子，用 root 登录后敲一条命令装好（板子要能上外网）：

```bash
curl -fsSL https://raw.githubusercontent.com/Xbotics-Embodied-AI-club/lerobot/xbotics-v0.6.1/src/lerobot/xbotics/board/bootstrap.sh | bash
```

它先装 uv（管理 Python 和依赖的工具），在 `/root/xbotics` 建好 LeRobot 环境，再由环境里的 LeRobot 把其余的东西装好。最后一行是 `· 装好了：xbotics-config status 看各个服务` 就完成了。已经装过的东西保持不动，缺的补上、旧的换新，所以对哪块板子敲都可以。

以后维护版更新了，板子上只敲一条：

```bash
xbotics-config update
```

![升级：LeRobot 已是最新时，什么都不用重启](images/install-02-update.png)

它把 LeRobot 换成最新的版本，再把板子上的其余东西按新版装一遍，只改有变化的部分。网页服务的开关、改过的密码、你自己用 `uv add` 装的包都保留。在网页终端或 VS Code 的终端里敲也可以：需要重启它们时，会等升级做完再重启，页面断开一下，刷新就好。训练电脑要同样升级，见 [2.7 节](#train-env-upgrade)。

## 1.3 配网

板子一直开着一个热点，名字是 `Xbotics-` 加四位（每块板不同），不用密码。电脑或手机连上它，会弹出「星禾套件 · 网络设置」页。没弹出来就在浏览器打开 `http://10.42.0.1`。如果提示「此网络无法访问互联网」，选保持连接：热点只给这个页面用，本来就不通外网。

页面顶上三块每两秒刷新一次，右上角的小标签一眼看状态：网线和 Wi-Fi 是绿色「能上外网」、橙色「上不了外网」或红色「没连上」，下面是拿到的地址；本热点显示热点的名字和地址。手机上打开是一列，电脑上三块并排。

![网络设置页（电脑上打开）](images/use-08-network-page.png)

**连 Wi-Fi：** 点「扫描附近的 Wi-Fi」，列表按信号强弱排，右边的格数是信号，带锁的要密码。点你的 Wi-Fi，填密码（右边的眼睛可以显示密码），点「连接」。屏幕底边显示 `已连上 …，地址 …` 就好了，点一下关掉。连的过程中热点不断，页面一直开着；密码错了会提示，以前存过的 Wi-Fi 不受影响。连上过的 Wi-Fi 列在「已保存」，板子开机自动连，不要了点名字旁边的 ×。标着「要账号，暂不支持」的网络需要用户名登录，连不了。

![手机上扫描出的 Wi-Fi 列表（周边网络的名字打了码）](images/use-09-network-wifi.png){width=320px}

**设网线：** 出厂是固定 IP，地址 `192.168.1.50`、网关 `192.168.1.1`。你的路由器正好是 `192.168.1.x` 网段时，插上网线就能用。不是这个网段时，点「自动获取（DHCP）」，或点「固定 IP」填你网络里的地址、前缀长度、网关和 DNS，再点「保存」；哪一格填得不对，页面会直接点名。如果是插着网线打开的这个页面，保存时连接会断一下；改成固定 IP 后要用新地址打开。

**从你自己的网络打开板子：** 板子连上网以后，页面上「从你自己的网络打开」一栏列出六个网页。上面一排是板子的各个地址，点哪个，下面的链接就用哪个地址。把电脑切回你自己的 Wi-Fi 或网线，再点这些链接。带 `.local` 的地址要电脑支持这种名字，打不开就选 IP。

(connect-server)=
## 1.4 连训练电脑

推理服务和训练曲线都在训练电脑上。在网络设置页的「连服务器」一栏连上它，板子就把电脑上的端口接到自己身上。训练电脑要先开好 SSH 服务，见 [2.5 节](#train-env-ssh)。

1. 填三格：**服务器地址**（训练电脑的 IP）、**SSH 端口**（一般是 22）、**账号**（训练电脑上的用户名）。
2. 选登录方式：**账号密码**就填这个账号的密码；**私钥**就把私钥整段粘贴进去（从 `-----BEGIN` 那一行到 `-----END` 那一行），或点「选一个文件」，私钥有口令就填在下面。
3. 下面是要映射的端口，默认两条，一般不用改：

| 映射什么 | 板子端口 | 训练电脑上的端口 | 用来做什么 |
| --- | --- | --- | --- |
| 推理服务 | 8080 | 8080 | 部署时板子上的程序连它 |
| TensorBoard | 6006 | 6006 | 看训练曲线 |

4. 点「保存并连接」。几秒后右上角变成绿色「已连上」，下面每条映射标出状态：「通」是电脑上这个端口有服务在答；「服务器上这个端口没有服务」说明连接是好的，只是电脑上还没启动这个服务。

![连上训练电脑：TensorBoard 通，推理服务还没启动](images/use-16-connect-server.png)

连上以后，板子上的程序连 `127.0.0.1:8080` 就是训练电脑的推理服务；你在任何一台电脑上打开 `http://<板子IP>:6006`，就是训练电脑上的 TensorBoard，也可以点映射旁边的「打开」。断网、电脑重启后板子会自己连回，板子重启后也自动连上。不用了点「断开」，开机也不再自动连。

连不上时，页面用中文写出原因：「登录被拒」是账号、密码或私钥不对，板子不会拿错的密码一直重试（那样服务器可能把账号锁掉），改好后再点「保存并连接」；「私钥的口令不对」「私钥格式不对」就检查口令，或重新粘贴整段私钥；「服务器的指纹变了」是电脑重装过系统，再点一次「保存并连接」会重新认它；「找不到这个服务器地址」「连不上服务器」是地址写错，或板子的网络到不了这台电脑；「服务器不允许端口转发」要在电脑的 `/etc/ssh/sshd_config` 里设 `AllowTcpForwarding yes`。

![在电脑上经板子打开训练电脑上的 TensorBoard](images/use-17-tensorboard.png)


## 1.5 网页终端

浏览器打开 `http://<板子IP>:8002`，在 `login:` 后面输入 `root`，再输入密码 `123456`（输入时屏幕上不显示）。提示符前面有 `(xbotics-board)`，说明已经在 LeRobot 环境里，后面的命令可以直接敲。

![登录网页终端](images/use-01-login.png)

网页终端和普通终端一样，命令可以粘贴进去，`Ctrl+C` 中断正在运行的命令。**关掉这个网页，里面正在运行的命令也会跟着结束**，所以录数据、推理时别关它。

## 1.6 VS Code 和 JupyterLab

**浏览器版 VS Code** 适合看文件、改代码、写要反复用的脚本。浏览器打开 `http://<板子IP>:8004`，输入密码 `123456`。

![浏览器版 VS Code 的登录页](images/use-06-vscode-login.png)

- **看文件**：左边 Explorer 是板子上的文件；换目录用菜单 File → Open Folder，或在网址后面加 `?folder=/root/data`
- **改文件**：点开直接改，`Ctrl+S` 保存
- **运行 Python 文件**：打开后点编辑区右上角的 ▷，默认用 LeRobot 环境里的 Python
- **终端**：菜单 Terminal → New Terminal，录数据、推理的命令也可以在这里敲

试一下：在 `~/xbotics` 里新建 `check_board.py`，写进下面几行，点 ▷。

```python
"""板子自检：用的是哪个 Python、LeRobot 版本、四个固定名字各指向什么。"""

import os
import sys

import lerobot

print("Python:", sys.executable)
print("LeRobot:", lerobot.__version__)
for name in ("follower", "leader", "top", "wrist"):
    path = f"/dev/xbotics/{name}"
    print(f"{path:24s}", "->", os.path.realpath(path) if os.path.exists(path) else "没插")
```

![打开 check_board.py，点 ▷ 运行，下方终端打出结果](images/use-07-vscode-run.png)

**JupyterLab** 适合边试边看结果，比如读一组关节读数、看一帧相机画面。浏览器打开 `http://<板子IP>:8003`，输入密码 `123456`。

![JupyterLab 的登录页](images/use-10-jupyter-login.png)

- **新建 notebook**：左上角蓝色的 `+` → Notebook 下点 `Python 3 (ipykernel)`，`import lerobot` 直接能用
- **运行**：在格子里写代码，`Shift+Enter` 运行这一格
- **终端**：蓝色 `+` → Terminal

把上面 `check_board.py` 的代码贴进一格，`Shift+Enter`：

![notebook 里运行自检代码](images/use-11-jupyter-notebook.png)

关掉网页后，notebook 的内核还在后台，变量都在，下次打开接着用。这一点和网页终端相反。不用了就在菜单 Kernel → Shut Down Kernel 关掉，因为导入 LeRobot 后一个内核约占 0.4 GB 内存。

(xbotics-config)=
## 1.7 xbotics-config

8002 到 8006 五个网页服务用 `xbotics-config` 开关，在网页终端里敲：

```bash
xbotics-config
```

弹出一个菜单：方向键上下选，空格勾选或取消，回车生效。勾上的服务开着、开机自动起来；取消勾选的马上关掉、开机也不起。8001 网络设置页不在菜单里，它一直开着，否则就没法从热点进板子改网络。

![xbotics-config 的菜单：五个服务都勾着](images/use-15-xbotics-config-menu.png)

不想用菜单，也可以直接敲命令：

```bash
xbotics-config status
xbotics-config disable calibrate
xbotics-config enable calibrate
```

`status` 列出每个服务开没开、在不在跑、网址。`enable` 和 `disable` 后面跟名字：`terminal`、`jupyter`、`vscode`、`hardware-check`、`calibrate`，写 `all` 就是全部。

![xbotics-config status：五个服务都开着，连训练电脑的隧道还没设](images/use-14-xbotics-config-status.png)

最后一行 `tunnel` 是 1.4 节连训练电脑的连接，它在网络设置页上开关。

如果在网页终端里关掉 `terminal`，页面会马上断开，要再打开就从 VS Code 的终端里敲 `xbotics-config enable terminal`。用不上的服务可以关掉省内存：硬件检测、网页标定各占约 0.3 GB。
