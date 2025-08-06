# 秒级AI交易系统 - 快速启动指南

## 🚀 快速开始

### 1. 环境准备

确保您的系统已安装以下软件：

- **Docker** (>= 20.10)
- **Docker Compose** (>= 2.0)
- **Git**
- **Bash** (Linux/macOS) 或 **WSL** (Windows)

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd ai_trading_system
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
nano .env  # 或使用您喜欢的编辑器
```

**重要配置项：**
```bash
# 币安API密钥（必须）
BINANCE_API_KEY=your_actual_api_key
BINANCE_SECRET_KEY=your_actual_secret_key
BINANCE_TESTNET=true  # 建议首次使用测试网

# 交易标的
TRADING_SUPPORTED_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT

# 风险控制
TRADING_MAX_POSITION_SIZE=10000.0
TRADING_MAX_DAILY_LOSS=1000.0
```

### 4. 一键部署

```bash
./scripts/deploy.sh
```

### 5. 访问系统

部署完成后，您可以访问：

- **监控面板**: http://localhost:3000 (admin/admin123)
- **系统指标**: http://localhost:8000/metrics
- **Kafka管理**: http://localhost:8080
- **Prometheus**: http://localhost:9090

## 📊 监控和管理

### 查看系统状态
```bash
./scripts/deploy.sh status
```

### 查看日志
```bash
# 查看所有服务日志
./scripts/deploy.sh logs

# 查看特定服务日志
./scripts/deploy.sh logs trading-system
```

### 健康检查
```bash
./scripts/deploy.sh health
```

### 重启系统
```bash
./scripts/deploy.sh restart
```

### 停止系统
```bash
./scripts/deploy.sh stop
```

## 🔧 常见问题

### Q: 系统无法启动？
**A:** 检查以下事项：
1. Docker服务是否运行
2. 端口是否被占用
3. .env文件是否正确配置
4. 查看日志：`./scripts/deploy.sh logs`

### Q: 交易信号不生成？
**A:** 可能原因：
1. API密钥配置错误
2. 网络连接问题
3. 市场数据源异常
4. 检查日志中的错误信息

### Q: 如何调整风险参数？
**A:** 编辑 `.env` 文件中的风险配置：
```bash
TRADING_MAX_POSITION_SIZE=50000.0
TRADING_MAX_DAILY_LOSS=5000.0
TRADING_MAX_DRAWDOWN=0.03
```

### Q: 如何添加新的交易标的？
**A:** 修改 `.env` 文件：
```bash
TRADING_SUPPORTED_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,DOTUSDT
```
然后重启系统。

## 📈 性能优化

### 系统资源监控
```bash
# 查看容器资源使用
docker stats

# 查看系统负载
htop
```

### 优化建议

1. **内存优化**：
   - 调整Redis内存限制
   - 优化ClickHouse缓存设置

2. **网络优化**：
   - 使用CDN加速数据源
   - 配置本地DNS缓存

3. **存储优化**：
   - 使用SSD存储
   - 定期清理历史数据

## 🛡️ 安全建议

1. **API密钥安全**：
   - 定期轮换API密钥
   - 使用最小权限原则
   - 启用IP白名单

2. **网络安全**：
   - 配置防火墙规则
   - 使用VPN访问
   - 启用SSL/TLS

3. **数据备份**：
   - 定期备份配置文件
   - 备份交易历史数据
   - 测试恢复流程

## 📚 进阶配置

### 自定义AI策略
1. 修改 `src/ai_engine/trading_ai.py`
2. 添加新的技术指标
3. 调整模型参数

### 扩展数据源
1. 在 `src/data_pipeline/` 添加新的数据收集器
2. 配置新的Kafka主题
3. 更新数据处理逻辑

### 多交易所支持
1. 在 `config/config.py` 添加新交易所配置
2. 实现交易所适配器
3. 配置智能路由

## 🆘 获取帮助

- **查看详细文档**: `README.md`
- **报告问题**: 创建GitHub Issue
- **技术讨论**: 加入社区讨论
- **商业支持**: 联系技术团队

## ⚠️ 免责声明

本系统仅供学习和研究使用。实际交易存在风险，可能导致资金损失。使用前请：

1. 充分了解交易风险
2. 在测试环境验证策略
3. 从小额资金开始
4. 持续监控系统运行

**重要提醒**: 永远不要投入超过您承受能力的资金！