# Kubernetes 常见攻击模式与防御

## 常见攻击路径

### 1. 容器逃逸
**攻击手法：**
- 利用特权容器（privileged: true）
- 挂载宿主机 /var/run/docker.sock
- 利用内核漏洞（如 DirtyPipe）
- 通过 hostPath 卷访问宿主机文件系统

**防御措施：**
- 禁用特权容器
- 使用 PodSecurityPolicy / PodSecurityAdmission
- 限制 hostPath 卷使用
- 定期更新内核版本

### 2. 权限提升
**攻击手法：**
- 利用 RBAC 配置错误
- 读取 ServiceAccount Token
- 滥用 ClusterRole 绑定
- 通过 ConfigMap 注入恶意配置

**防御措施：**
- 严格限制 ServiceAccount 权限
- 启用自动挂载 token 的限制
- 定期审计 RBAC 策略
- 使用 Kyverno 进行策略验证

### 3. 横向移动
**攻击手法：**
- 利用网络策略缺失
- 通过 Pod 间通信访问敏感服务
- 利用内网服务暴露
- 扫描和攻击同节点 Pod

**防御措施：**
- 实施网络微隔离
- 使用服务网格（Istio）进行 mTLS
- 限制 Pod 到 Pod 的通信
- 监控异常网络流量

### 4. 数据泄露
**攻击手法：**
- 读取 Pod 中的 Secret
- 拦截容器间通信流量
- 利用日志聚合器漏洞
- 通过挂载卷访问持久化数据

**防御措施：**
- 加密静态数据
- 使用外部密钥管理服务
- 实施网络加密
- 限制日志访问权限

## 安全检测指标

| 指标 | 说明 | 危险信号 |
|------|------|----------|
| 特权容器数量 | 使用 privileged 的 Pod | 任何特权容器 |
| 主机网络 | 使用 hostNetwork | 非必要的 hostNetwork |
| 根用户运行 | 以 root 运行容器 | 任何 root 进程 |
| 危险 Capabilities | ADD 高风险能力 | SYS_ADMIN, NET_ADMIN |
| 宿主机路径挂载 | hostPath 挂载 | 挂载 /、/etc、/var |
| 公开 Service | NodePort/LoadBalancer | 敏感服务外部暴露 |
