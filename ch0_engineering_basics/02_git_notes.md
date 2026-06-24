# Git / GitHub Notes

## 1. 为什么算法实习要学 Git

Git 用于代码版本管理。算法项目中常用于：

- 保存实验代码版本
- 回滚错误修改
- 管理不同功能分支
- 上传项目到 GitHub
- 和他人协作
- 展示简历项目

---

## 2. 基本概念

### Working Directory

工作区，当前正在编辑的文件。

### Staging Area

暂存区，用 `git add` 加入。

### Repository

本地仓库，用 `git commit` 保存版本。

### Remote Repository

远程仓库，例如 GitHub。

---

## 3. 初始化和克隆

### 初始化仓库

```bash
git init
```

### 克隆远程仓库

```bash
git clone <repo_url>
```

---

## 4. 查看状态

```bash
git status
```

常用来查看：

- 哪些文件被修改
- 哪些文件还没加入暂存区
- 哪些文件已经 staged

---

## 5. 添加和提交

### 添加文件到暂存区

```bash
git add file.py
git add .
```

### 提交版本

```bash
git commit -m "message"
```

示例：

```bash
git commit -m "add logistic regression notes"
```

提交信息建议简洁明确。

---

## 6. 查看历史和差异

### 查看提交历史

```bash
git log
git log --oneline
```

### 查看未暂存改动

```bash
git diff
```

### 查看已暂存改动

```bash
git diff --staged
```

---

## 7. 分支

### 查看分支

```bash
git branch
```

### 创建分支

```bash
git branch feature-name
```

### 切换分支

```bash
git switch feature-name
```

旧写法：

```bash
git checkout feature-name
```

### 创建并切换

```bash
git switch -c feature-name
```

---

## 8. 合并分支

```bash
git switch main
git merge feature-name
```

含义：把 `feature-name` 分支的修改合并到 `main`。

---

## 9. 远程仓库

### 查看远程仓库

```bash
git remote -v
```

### 添加远程仓库

```bash
git remote add origin <repo_url>
```

### 推送

```bash
git push -u origin main
git push
```

### 拉取

```bash
git pull
```

---

## 10. .gitignore

`.gitignore` 用来忽略不应该提交的文件。

算法项目常见忽略：

```gitignore
__pycache__/
*.pyc
.env
.venv/
venv/
.ipynb_checkpoints/
data/raw/
outputs/
checkpoints/
*.pt
*.pth
*.ckpt
.DS_Store
```

注意：

- 大数据集通常不要直接提交到 Git
- 模型权重通常不要直接提交到 Git
- 密钥、token、密码绝对不要提交

---

## 11. 常见工作流

```bash
git status
git add .
git commit -m "update notes"
git push
```

更完整：

```bash
git switch -c feature-x
# 修改代码
git status
git add .
git commit -m "add feature x"
git push -u origin feature-x
```

---

## 12. 回退和撤销：谨慎使用

### 撤销工作区某个文件修改

```bash
git restore file.py
```

### 取消暂存

```bash
git restore --staged file.py
```

### 查看某次提交

```bash
git show <commit_id>
```

### 回退提交

```bash
git revert <commit_id>
```

`revert` 会生成一个新的提交来撤销旧提交，比较安全。

`reset --hard` 会丢弃历史和工作区修改，初学阶段谨慎使用。

---

## 13. GitHub 项目建议结构

```text
project-name/
├── README.md
├── requirements.txt
├── configs/
├── data/
├── notebooks/
├── src/
├── scripts/
└── outputs/
```

README 至少写：

- 项目目标
- 数据说明
- 环境安装
- 训练命令
- 评估命令
- 实验结果
- 面试讲解要点

---

## 14. 面试/实习常问

### Git 的工作区、暂存区、本地仓库分别是什么？

工作区是当前编辑文件；暂存区是 `git add` 后准备提交的内容；本地仓库是 `git commit` 后保存的版本历史。

### `git add` 和 `git commit` 区别？

`git add` 把修改加入暂存区；`git commit` 把暂存区内容保存成一次版本提交。

### `git pull` 和 `git push` 区别？

`git pull` 从远程拉取更新；`git push` 把本地提交推送到远程。

### `.gitignore` 的作用？

指定不需要 Git 跟踪的文件，例如缓存、虚拟环境、数据集、模型权重、密钥等。

### 为什么不要提交大模型权重和密钥？

大文件会让仓库膨胀，影响协作；密钥泄露会带来安全风险。
