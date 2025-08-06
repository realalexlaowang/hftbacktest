"""
系统监控和指标收集 - 实时性能监控、告警和健康检查
"""
import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from loguru import logger
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest,
    start_http_server, CONTENT_TYPE_LATEST
)
import redis

from config.config import get_config


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    load_average: List[float]


@dataclass
class ApplicationMetrics:
    """应用指标"""
    timestamp: int
    total_orders: int
    successful_orders: int
    failed_orders: int
    avg_latency_ms: float
    active_positions: int
    total_pnl: float
    current_drawdown: float
    risk_level: str


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PrometheusMetrics:
    """Prometheus指标收集器"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # 系统指标
        self.cpu_usage = Gauge(
            'trading_system_cpu_usage_percent',
            'CPU使用率百分比',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'trading_system_memory_usage_percent',
            '内存使用率百分比',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'trading_system_disk_usage_percent',
            '磁盘使用率百分比',
            registry=self.registry
        )
        
        # 应用指标
        self.order_counter = Counter(
            'trading_system_orders_total',
            '订单总数',
            ['status', 'symbol'],
            registry=self.registry
        )
        
        self.latency_histogram = Histogram(
            'trading_system_latency_seconds',
            '系统延迟分布',
            ['component'],
            registry=self.registry
        )
        
        self.active_positions = Gauge(
            'trading_system_active_positions',
            '活跃持仓数量',
            registry=self.registry
        )
        
        self.pnl_gauge = Gauge(
            'trading_system_pnl_total',
            '总盈亏',
            registry=self.registry
        )
        
        self.drawdown_gauge = Gauge(
            'trading_system_drawdown_percent',
            '当前回撤百分比',
            registry=self.registry
        )
        
        # 数据流指标
        self.data_points_counter = Counter(
            'trading_system_data_points_total',
            '处理的数据点总数',
            ['source', 'symbol'],
            registry=self.registry
        )
        
        self.ai_predictions_counter = Counter(
            'trading_system_ai_predictions_total',
            'AI预测总数',
            ['signal_type', 'strategy'],
            registry=self.registry
        )
        
        # 错误指标
        self.error_counter = Counter(
            'trading_system_errors_total',
            '错误总数',
            ['component', 'error_type'],
            registry=self.registry
        )
    
    def update_system_metrics(self, metrics: SystemMetrics):
        """更新系统指标"""
        self.cpu_usage.set(metrics.cpu_usage)
        self.memory_usage.set(metrics.memory_usage)
        self.disk_usage.set(metrics.disk_usage)
    
    def record_order(self, status: str, symbol: str):
        """记录订单"""
        self.order_counter.labels(status=status, symbol=symbol).inc()
    
    def record_latency(self, component: str, latency_seconds: float):
        """记录延迟"""
        self.latency_histogram.labels(component=component).observe(latency_seconds)
    
    def update_positions(self, count: int):
        """更新持仓数量"""
        self.active_positions.set(count)
    
    def update_pnl(self, pnl: float):
        """更新盈亏"""
        self.pnl_gauge.set(pnl)
    
    def update_drawdown(self, drawdown: float):
        """更新回撤"""
        self.drawdown_gauge.set(drawdown)
    
    def record_data_point(self, source: str, symbol: str):
        """记录数据点"""
        self.data_points_counter.labels(source=source, symbol=symbol).inc()
    
    def record_prediction(self, signal_type: str, strategy: str):
        """记录AI预测"""
        self.ai_predictions_counter.labels(signal_type=signal_type, strategy=strategy).inc()
    
    def record_error(self, component: str, error_type: str):
        """记录错误"""
        self.error_counter.labels(component=component, error_type=error_type).inc()
    
    def get_metrics(self) -> str:
        """获取Prometheus格式的指标"""
        return generate_latest(self.registry).decode('utf-8')


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.config = get_config()
        self.metrics = PrometheusMetrics()
        self.redis_client = None
        self.alerts: List[Alert] = []
        self.running = False
        
        # 告警阈值
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'latency_ms': 500.0,
            'error_rate': 0.05,
            'drawdown': 0.1
        }
    
    async def initialize(self):
        """初始化监控器"""
        # Redis连接
        self.redis_client = redis.Redis(
            host=self.config.redis.host,
            port=self.config.redis.port,
            password=self.config.redis.password,
            db=self.config.redis.db,
            decode_responses=True
        )
        
        # 启动Prometheus HTTP服务器
        start_http_server(
            port=self.config.monitoring.prometheus_port,
            registry=self.metrics.registry
        )
        
        logger.info(f"Prometheus指标服务启动在端口 {self.config.monitoring.prometheus_port}")
        logger.info("系统监控器初始化完成")
    
    async def start_monitoring(self):
        """开始监控"""
        self.running = True
        
        # 启动监控任务
        tasks = [
            asyncio.create_task(self._monitor_system_metrics()),
            asyncio.create_task(self._monitor_application_health()),
            asyncio.create_task(self._check_alerts()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _monitor_system_metrics(self):
        """监控系统指标"""
        while self.running:
            try:
                # 收集系统指标
                metrics = await self._collect_system_metrics()
                
                # 更新Prometheus指标
                self.metrics.update_system_metrics(metrics)
                
                # 缓存到Redis
                await self._cache_metrics("system", asdict(metrics))
                
                # 检查阈值
                await self._check_system_thresholds(metrics)
                
                # 每10秒收集一次
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"系统指标监控错误: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # 网络IO
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # 进程数量
        process_count = len(psutil.pids())
        
        # 负载平均值
        load_avg = list(psutil.getloadavg())
        
        return SystemMetrics(
            timestamp=int(time.time() * 1000),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            process_count=process_count,
            load_average=load_avg
        )
    
    async def _monitor_application_health(self):
        """监控应用健康状态"""
        while self.running:
            try:
                # 从Redis获取应用指标
                app_metrics = await self._collect_application_metrics()
                
                if app_metrics:
                    # 更新Prometheus指标
                    self.metrics.update_positions(app_metrics.active_positions)
                    self.metrics.update_pnl(app_metrics.total_pnl)
                    self.metrics.update_drawdown(app_metrics.current_drawdown)
                    
                    # 缓存应用指标
                    await self._cache_metrics("application", asdict(app_metrics))
                    
                    # 检查应用阈值
                    await self._check_application_thresholds(app_metrics)
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"应用健康监控错误: {e}")
                await asyncio.sleep(10)
    
    async def _collect_application_metrics(self) -> Optional[ApplicationMetrics]:
        """收集应用指标"""
        try:
            # 从Redis获取统计数据
            order_stats = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, "stats:orders"
            )
            
            position_stats = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, "stats:positions"
            )
            
            pnl_stats = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, "stats:pnl"
            )
            
            # 解析数据
            order_data = json.loads(order_stats) if order_stats else {}
            position_data = json.loads(position_stats) if position_stats else {}
            pnl_data = json.loads(pnl_stats) if pnl_stats else {}
            
            return ApplicationMetrics(
                timestamp=int(time.time() * 1000),
                total_orders=order_data.get('total', 0),
                successful_orders=order_data.get('successful', 0),
                failed_orders=order_data.get('failed', 0),
                avg_latency_ms=order_data.get('avg_latency_ms', 0),
                active_positions=position_data.get('active_count', 0),
                total_pnl=pnl_data.get('total_pnl', 0),
                current_drawdown=pnl_data.get('current_drawdown', 0),
                risk_level=pnl_data.get('risk_level', 'low')
            )
            
        except Exception as e:
            logger.error(f"收集应用指标错误: {e}")
            return None
    
    async def _check_system_thresholds(self, metrics: SystemMetrics):
        """检查系统阈值"""
        # CPU使用率检查
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            await self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"CPU使用率过高: {metrics.cpu_usage:.1f}%",
                {"cpu_usage": metrics.cpu_usage}
            )
        
        # 内存使用率检查
        if metrics.memory_usage > self.thresholds['memory_usage']:
            await self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"内存使用率过高: {metrics.memory_usage:.1f}%",
                {"memory_usage": metrics.memory_usage}
            )
        
        # 磁盘使用率检查
        if metrics.disk_usage > self.thresholds['disk_usage']:
            await self._create_alert(
                AlertLevel.ERROR,
                "system",
                f"磁盘使用率过高: {metrics.disk_usage:.1f}%",
                {"disk_usage": metrics.disk_usage}
            )
    
    async def _check_application_thresholds(self, metrics: ApplicationMetrics):
        """检查应用阈值"""
        # 延迟检查
        if metrics.avg_latency_ms > self.thresholds['latency_ms']:
            await self._create_alert(
                AlertLevel.WARNING,
                "application",
                f"平均延迟过高: {metrics.avg_latency_ms:.1f}ms",
                {"avg_latency_ms": metrics.avg_latency_ms}
            )
        
        # 错误率检查
        if metrics.total_orders > 0:
            error_rate = metrics.failed_orders / metrics.total_orders
            if error_rate > self.thresholds['error_rate']:
                await self._create_alert(
                    AlertLevel.ERROR,
                    "application",
                    f"错误率过高: {error_rate:.2%}",
                    {"error_rate": error_rate}
                )
        
        # 回撤检查
        if metrics.current_drawdown > self.thresholds['drawdown']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "trading",
                f"回撤过大: {metrics.current_drawdown:.2%}",
                {"current_drawdown": metrics.current_drawdown}
            )
    
    async def _create_alert(self, level: AlertLevel, component: str, 
                           message: str, metadata: Dict[str, Any]):
        """创建告警"""
        alert = Alert(
            alert_id=f"{component}_{int(time.time())}",
            level=level,
            component=component,
            message=message,
            timestamp=int(time.time() * 1000),
            metadata=metadata
        )
        
        self.alerts.append(alert)
        
        # 记录日志
        if level == AlertLevel.CRITICAL:
            logger.critical(f"[{component}] {message}")
        elif level == AlertLevel.ERROR:
            logger.error(f"[{component}] {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"[{component}] {message}")
        else:
            logger.info(f"[{component}] {message}")
        
        # 发送告警通知
        await self._send_alert_notification(alert)
        
        # 缓存告警
        await self._cache_alert(alert)
    
    async def _send_alert_notification(self, alert: Alert):
        """发送告警通知"""
        if not self.config.monitoring.alert_webhook_url:
            return
        
        try:
            payload = {
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "component": alert.component,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.monitoring.alert_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"告警通知发送失败: {response.status}")
                        
        except Exception as e:
            logger.error(f"发送告警通知错误: {e}")
    
    async def _cache_metrics(self, metric_type: str, data: Dict[str, Any]):
        """缓存指标数据"""
        try:
            key = f"metrics:{metric_type}:{int(time.time())}"
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(key, 3600, json.dumps(data))
            )
        except Exception as e:
            logger.error(f"缓存指标错误: {e}")
    
    async def _cache_alert(self, alert: Alert):
        """缓存告警"""
        try:
            key = f"alerts:{alert.alert_id}"
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(key, 86400, json.dumps(asdict(alert)))
            )
        except Exception as e:
            logger.error(f"缓存告警错误: {e}")
    
    async def _check_alerts(self):
        """检查告警状态"""
        while self.running:
            try:
                # 清理过期的告警
                current_time = time.time() * 1000
                self.alerts = [
                    alert for alert in self.alerts
                    if current_time - alert.timestamp < 3600000  # 1小时内的告警
                ]
                
                await asyncio.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                logger.error(f"告警检查错误: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        while self.running:
            try:
                # 清理1天前的指标数据
                cutoff_time = int(time.time()) - 86400
                
                # 获取所有指标键
                pattern = f"metrics:*"
                keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.keys, pattern
                )
                
                # 删除过期键
                for key in keys:
                    try:
                        timestamp = int(key.split(':')[-1])
                        if timestamp < cutoff_time:
                            await asyncio.get_event_loop().run_in_executor(
                                None, self.redis_client.delete, key
                            )
                    except (ValueError, IndexError):
                        continue
                
                # 每小时清理一次
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"数据清理错误: {e}")
                await asyncio.sleep(1800)  # 出错时等待30分钟
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        logger.info("系统监控已停止")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 获取最新的系统指标
            system_key = f"metrics:system:*"
            system_keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, system_key
            )
            
            if system_keys:
                latest_key = sorted(system_keys)[-1]
                system_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, latest_key
                )
                system_metrics = json.loads(system_data) if system_data else {}
            else:
                system_metrics = {}
            
            # 获取最新的应用指标
            app_key = f"metrics:application:*"
            app_keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, app_key
            )
            
            if app_keys:
                latest_key = sorted(app_keys)[-1]
                app_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, latest_key
                )
                app_metrics = json.loads(app_data) if app_data else {}
            else:
                app_metrics = {}
            
            # 获取最近的告警
            recent_alerts = [
                asdict(alert) for alert in self.alerts[-10:]  # 最近10个告警
            ]
            
            return {
                "status": "healthy",
                "timestamp": int(time.time() * 1000),
                "system_metrics": system_metrics,
                "application_metrics": app_metrics,
                "recent_alerts": recent_alerts,
                "alert_count": len(self.alerts)
            }
            
        except Exception as e:
            logger.error(f"获取健康状态错误: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            }
    
    async def cleanup(self):
        """清理资源"""
        self.running = False
        
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.close
            )
        
        logger.info("系统监控器资源清理完成")


# 使用示例
async def main():
    monitor = SystemMonitor()
    await monitor.initialize()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())