# CIS Kubernetes Benchmark 核心要点

## 控制项摘要

### 1. 集群授权（Control 1.x）
- 1.1 禁用匿名访问
- 1.2 禁用不安全的认证代理
- 1.3 避免使用 Node 身份作为用户
- 1.4 定期轮换 ServiceAccount Token

### 2. 凭证配置（Control 2.x）
- 2.1 使用加密的 etcd 存储
- 2.2 保护 Kubernetes Secrets
- 2.3 最小化 ServiceAccount 权限
- 2.4 定期清理过期凭证

### 3. 控制平面（Control 3.x）
- 3.1 限制 API Server 端口暴露
- 3.2 启用审计日志
- 3.3 限制 Admission Controllers
- 3.4 保护 kubelet 配置文件

### 4. 容器运行时（Control 4.x）
- 4.1 限制容器特权
- 4.2 禁用容器内的内核模块加载
- 4.3 保护容器运行时 socket
- 4.4 限制 cgroup 权限

### 5. Kubernetes 策略（Control 5.x）
- 5.1 使用 Pod 安全策略/Admission
- 5.2 网络策略实施
- 5.3 限制 Privilege Escalation
- 5.4 审计异常行为

### 6. 集群节点（Control 6.x）
- 6.1 最小化系统服务
- 6.2 限制 kubelet 权限
- 6.3 保护节点凭证
- 6.4 配置审计日志

### 7. 工作负载安全（Control 7.x）
- 7.1 资源请求和限制
- 7.2 限制容器用户
- 7.3 启用只读根文件系统
- 7.4 限制进程通信

## 评分标准

- Level 1（基础安全）：必须满足的核心控制项
- Level 2（增强安全）：需要额外投入的强化控制项

达到 Level 1 是生产环境的最低要求。
