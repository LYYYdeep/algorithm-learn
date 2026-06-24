# Linux / Shell Notes

## 1. 为什么算法实习要学 Linux

算法工程师实习中，很多训练、部署和实验都在 Linux 服务器上完成。

需要熟悉：

- 目录和文件操作
- 查看日志
- 搜索文件和内容
- 管理进程
- 查看磁盘和资源
- 远程连接服务器
- 后台运行训练任务
- GPU 状态查看

---

## 2. 路径和目录

### 查看当前目录

```bash
pwd
```

`pwd` = print working directory，打印当前所在目录。

### 查看目录内容

```bash
ls
ls -l
ls -a
ls -lh
ls -lah
```

常用含义：

- `-l`：显示详细信息
- `-a`：显示隐藏文件
- `-h`：以人类可读方式显示文件大小

### 切换目录

```bash
cd path
cd ..
cd ~
cd -
```

含义：

- `cd path`：进入指定目录
- `cd ..`：返回上一级目录
- `cd ~`：回到用户主目录
- `cd -`：回到上一次所在目录

---

## 3. 绝对路径和相对路径

### 绝对路径

从根目录 `/` 开始，例如：

```bash
/Users/liangyang/LYYY/algorithm-learn
```

特点：无论当前在哪，都能定位到同一个路径。

### 相对路径

相对于当前目录，例如：

```bash
ch1_ml_basics/01_binary_metrics.py
../some_folder
```

特点：依赖当前所在目录。

---

## 4. 创建文件和目录

### 创建目录

```bash
mkdir folder_name
mkdir -p a/b/c
```

`mkdir -p` 可以递归创建多级目录；目录已存在时也不会报错。

### 创建空文件

```bash
touch notes.md
```

如果文件不存在，会创建文件；如果文件存在，会更新修改时间。

---

## 5. 复制、移动、重命名

### 复制文件

```bash
cp source target
```

### 复制目录

```bash
cp -r source_dir target_dir
```

### 移动或重命名

```bash
mv old_path new_path
```

例子：

```bash
mv old_name.py new_name.py
```

---

## 6. 删除文件和目录

### 删除文件

```bash
rm file.txt
```

### 删除目录

```bash
rm -r folder
```

### 强制递归删除

```bash
rm -rf folder
```

`rm -rf` 非常危险：

- 不进入回收站
- 删除后通常难以恢复
- 路径写错可能造成严重数据丢失

不要随便运行：

```bash
rm -rf *
rm -rf /
rm -rf ~/.ssh
```

---

## 7. 查看文件内容

```bash
cat file.txt
less file.txt
head file.txt
head -n 20 file.txt
tail file.txt
tail -n 20 file.txt
tail -f train.log
```

常用场景：

- `cat`：一次性打印小文件
- `less`：分页查看大文件
- `head`：查看文件开头
- `tail`：查看文件末尾
- `tail -f`：实时查看日志

---

## 8. 搜索文件和内容

### grep：搜索文件内容

```bash
grep "error" train.log
grep -i "error" train.log
grep -r "TODO" .
grep -n "loss" train.log
```

常用参数：

- `-i`：忽略大小写
- `-r`：递归搜索目录
- `-n`：显示行号

### find：搜索文件路径

```bash
find . -name "*.py"
find . -name "*.md"
find . -type f
find . -type d
```

---

## 9. 查看进程

```bash
ps aux
ps aux | grep python
top
htop
kill PID
kill -9 PID
```

常用场景：

- 找到正在运行的训练进程
- 查看 CPU / 内存占用
- 终止卡住的进程

`kill -9` 是强制终止，谨慎使用。

---

## 10. 磁盘和空间

```bash
df -h
du -sh folder
du -h --max-depth=1
```

含义：

- `df -h`：查看磁盘整体空间
- `du -sh folder`：查看某个目录大小
- `du -h --max-depth=1`：查看当前目录下一级目录大小

---

## 11. 权限

```bash
chmod +x script.sh
chmod 755 script.sh
```

常见场景：让 shell 脚本可执行。

---

## 12. 远程服务器

### ssh 登录

```bash
ssh username@server_ip
```

### scp 传文件

```bash
scp local_file username@server_ip:/remote/path
scp username@server_ip:/remote/file ./local_path
```

### 传目录

```bash
scp -r local_dir username@server_ip:/remote/path
```

---

## 13. 后台训练常用

### nohup

```bash
nohup python train.py > train.log 2>&1 &
```

含义：

- `nohup`：终端断开后继续运行
- `> train.log`：标准输出写入日志
- `2>&1`：错误输出也写入日志
- `&`：后台运行

### 查看日志

```bash
tail -f train.log
```

---

## 14. tmux 常用

```bash
tmux new -s train
tmux attach -t train
tmux ls
```

常用快捷键：

```text
Ctrl-b d  detach，退出但保持会话运行
```

---

## 15. GPU 常用

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=1 python train.py
```

含义：

- `nvidia-smi`：查看 GPU 状态
- `CUDA_VISIBLE_DEVICES`：指定使用哪张 GPU

---

## 16. 面试/实习常问

### 绝对路径和相对路径区别？

绝对路径从根目录开始，能唯一定位文件；相对路径相对于当前工作目录，会随着当前目录变化而变化。

### `rm -rf` 为什么危险？

它会递归且强制删除文件或目录，不进入回收站，路径写错可能造成严重数据丢失。

### `mkdir -p` 的作用？

递归创建多级目录；如果目录已存在也不会报错。

### 如何查看训练日志？

```bash
tail -f train.log
```

### 如何让训练断开终端后继续运行？

可以使用 `nohup` 或 `tmux`。
