## 🌍 Language / 语言

**English speakers**: Please read README_EN.md for the English version of this documentation.

# 交易机器人
一个支持多个交易所（目前包括 EdgeX 和 Backpack）的模块化交易机器人。该机器人实现了自动下单并在盈利时自动平仓的策略。

---

## 关注我

- **X (Twitter)**: [@yourQuantGuy](https://x.com/yourQuantGuy)

## 邀请链接 (获得返佣以及福利)

### EdgeX 交易所: [https://pro.edgex.exchange/referral/QUANT](https://pro.edgex.exchange/referral/QUANT)
- 即时获得 VIP 1 费率
- 10% 手续费返佣 (实际费率 0.0117%)
- 10% 额外奖励积分

### Backpack 交易所: [https://backpack.exchange/join/quant](https://backpack.exchange/join/quant)
使用我的推荐链接获得 30% 手续费返佣

## 安装

1. **克隆仓库**：

   ```bash
   git clone <repository-url>
   cd perp-dex-tools
   ```

2. **创建并激活虚拟环境**：

   ```bash
   python3 -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```

3. **安装依赖**：

   ```bash
   pip install -r requirements.txt
   ```

4. **设置环境变量**：
   使用 env_example.txt 在项目根目录创建`.env`文件。

## 示例命令：

### EdgeX 交易所：

ETH：

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH（带网格步长控制）：

```bash
python runbot.py --exchange edgex --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --grid-step 0.5
```

BTC：

```bash
python runbot.py --exchange edgex --ticker BTC --quantity 0.05 --take-profit 0.02 --max-orders 40 --wait-time 450
```

### Backpack 交易所：

ETH 永续合约：

```bash
python runbot.py --exchange backpack --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450
```

ETH 永续合约（带网格步长控制）：

```bash
python runbot.py --exchange backpack --ticker ETH --quantity 0.1 --take-profit 0.02 --max-orders 40 --wait-time 450 --grid-step 0.3
```

## 配置

### 环境变量

#### EdgeX 配置

- `EDGEX_ACCOUNT_ID`: 您的 EdgeX 账户 ID
- `EDGEX_STARK_PRIVATE_KEY`: 您的 EdgeX API 私钥
- `EDGEX_BASE_URL`: EdgeX API 基础 URL（默认：https://pro.edgex.exchange）
- `EDGEX_WS_URL`: EdgeX WebSocket URL（默认：wss://quote.edgex.exchange）

#### Backpack 配置

- `BACKPACK_PUBLIC_KEY`: 您的 Backpack 公钥
- `BACKPACK_SECRET_KEY`: 您的 Backpack 私钥（base64 编码）

### 命令行参数

- `--exchange`: 使用的交易所：'edgex'或'backpack'（默认：edgex）
- `--ticker`: 标的资产符号（例如：ETH、BTC、SOL）。合约 ID 自动解析。
- `--quantity`: 订单数量（默认：0.1）
- `--take-profit`: 止盈百分比（例如 0.02 表示 0.02%）
- `--direction`: 交易方向：'buy'或'sell'（默认：buy）
- `--max-orders`: 最大活跃订单数（默认：40）
- `--wait-time`: 订单间等待时间（秒）（默认：450）
- `--grid-step`: 与下一个平仓订单价格的最小距离百分比（默认：-100，表示无限制）
- `--instance-id`: 实例标识符，用于区分不同的脚本实例（默认：default）

## 交易策略

该机器人实现了简单的剥头皮策略：

1. **订单下单**：在市场价格附近下限价单
2. **订单监控**：等待订单成交
3. **平仓订单**：在止盈水平自动下平仓单
4. **持仓管理**：监控持仓和活跃订单
5. **风险管理**：限制最大并发订单数
6. **网格步长控制**：通过 `--grid-step` 参数控制新订单与现有平仓订单之间的最小价格距离

### 网格步长功能

`--grid-step` 参数用于控制新订单的平仓价格与现有平仓订单之间的最小距离：

- **默认值 -100**：无网格步长限制，按原策略执行
- **正值（如 0.5）**：新订单的平仓价格必须与最近的平仓订单价格保持至少 0.5% 的距离
- **作用**：防止平仓订单过于密集，提高成交概率和风险管理

例如，当看多且 `--grid-step 0.5` 时：
- 如果现有平仓订单价格为 2000 USDT
- 新订单的平仓价格必须低于 1990 USDT（2000 × (1 - 0.5%)）
- 这样可以避免平仓订单过于接近，提高整体策略效果

## 架构

该机器人采用支持多个交易所的模块化架构：

### 1. 交易所客户端

#### EdgeX 客户端（官方 SDK）

- 使用官方 SDK 的 EdgeX REST API 客户端
- 处理身份验证和 API 请求
- 管理订单下单、取消和状态查询
- 获取持仓和账户信息

#### Backpack 客户端（官方 SDK）

- 使用官方 BPX SDK 的 Backpack REST API 客户端
- 处理身份验证和 API 请求
- 管理订单下单、取消和状态查询
- 获取持仓和账户信息

### 2. WebSocket 管理器

#### EdgeX WebSocket 管理器（官方 SDK）

- 使用官方 SDK 的 WebSocket 连接管理
- 实时市场数据流
- 订单更新通知
- 自动连接处理

#### Backpack WebSocket 管理器

- Backpack 的 WebSocket 连接管理
- 实时订单更新通知
- ED25519 签名身份验证
- 自动连接处理

### 3. 主交易机器人（`runbot.py`）

- 核心剥头皮策略逻辑
- 订单下单和监控
- 持仓管理
- 主交易循环
- 多交易所支持

## 日志记录

该机器人提供全面的日志记录：

- **交易日志**：包含订单详情的 CSV 文件
- **调试日志**：带时间戳的详细活动日志
- **控制台输出**：实时状态更新
- **错误处理**：全面的错误日志记录和处理

## 安全功能

- **订单限制**：可配置的最大订单数量
- **超时处理**：超时时自动取消订单
- **持仓监控**：持续监控持仓和订单状态
- **错误恢复**：优雅处理 API 错误和断开连接

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 如适用，添加测试
5. 提交拉取请求

## 许可证

本项目采用 MIT 许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 免责声明

本软件仅供教育和研究目的。加密货币交易涉及重大风险，可能导致重大财务损失。使用风险自负，切勿用您无法承受损失的资金进行交易。

## 支持

相关问题：

- **EdgeX API**：查看[EdgeX API 文档](https://docs.edgex.exchange)
- **EdgeX SDK**：查看[EdgeX Python SDK 文档](https://github.com/edgex-Tech/edgex-python-sdk)
- **此机器人**：在此仓库中提交问题
