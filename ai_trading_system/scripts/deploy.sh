#!/bin/bash

# 秒级AI交易系统部署脚本
# Author: AI Trading Team
# Version: 1.0.0

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_step "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查环境变量文件
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_warn "未找到.env文件，复制示例文件..."
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_warn "请编辑.env文件并填入您的配置"
        exit 1
    fi
    
    log_info "依赖检查完成"
}

# 创建必要目录
create_directories() {
    log_step "创建必要目录..."
    
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/models"
    mkdir -p "$PROJECT_DIR/deployment/prometheus"
    mkdir -p "$PROJECT_DIR/deployment/grafana/dashboards"
    mkdir -p "$PROJECT_DIR/deployment/grafana/datasources"
    mkdir -p "$PROJECT_DIR/deployment/nginx"
    mkdir -p "$PROJECT_DIR/deployment/clickhouse"
    
    log_info "目录创建完成"
}

# 生成配置文件
generate_configs() {
    log_step "生成配置文件..."
    
    # Prometheus配置
    cat > "$PROJECT_DIR/deployment/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'trading-system'
    static_configs:
      - targets: ['trading-system:8000']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'clickhouse'
    static_configs:
      - targets: ['clickhouse:8123']
EOF

    # Grafana数据源配置
    cat > "$PROJECT_DIR/deployment/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Nginx配置
    cat > "$PROJECT_DIR/deployment/nginx/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream trading_system {
        server trading-system:8001;
    }
    
    upstream grafana {
        server grafana:3000;
    }
    
    upstream kafka_ui {
        server kafka-ui:8080;
    }
    
    server {
        listen 80;
        server_name _;
        
        location / {
            proxy_pass http://grafana;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/ {
            proxy_pass http://trading_system;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /kafka/ {
            proxy_pass http://kafka_ui/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

    # ClickHouse配置
    cat > "$PROJECT_DIR/deployment/clickhouse/config.xml" << 'EOF'
<?xml version="1.0"?>
<yandex>
    <logger>
        <level>information</level>
        <console>true</console>
    </logger>
    
    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>
    
    <listen_host>::</listen_host>
    
    <max_connections>4096</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>100</max_concurrent_queries>
    
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    <mark_cache_size>5368709120</mark_cache_size>
    
    <path>/var/lib/clickhouse/</path>
    <tmp_path>/var/lib/clickhouse/tmp/</tmp_path>
    <user_files_path>/var/lib/clickhouse/user_files/</user_files_path>
    
    <users_config>users.xml</users_config>
    <default_profile>default</default_profile>
    <default_database>default</default_database>
    
    <timezone>UTC</timezone>
    
    <remote_servers incl="clickhouse_remote_servers" />
    <zookeeper incl="zookeeper-servers" optional="true" />
    <macros incl="macros" optional="true" />
    
    <builtin_dictionaries_reload_interval>3600</builtin_dictionaries_reload_interval>
    
    <max_table_size_to_drop>0</max_table_size_to_drop>
    <max_partition_size_to_drop>0</max_partition_size_to_drop>
</yandex>
EOF

    log_info "配置文件生成完成"
}

# 构建Docker镜像
build_images() {
    log_step "构建Docker镜像..."
    
    cd "$PROJECT_DIR"
    docker-compose build --no-cache trading-system
    
    log_info "镜像构建完成"
}

# 启动服务
start_services() {
    log_step "启动服务..."
    
    cd "$PROJECT_DIR"
    
    # 启动基础设施服务
    log_info "启动基础设施服务..."
    docker-compose up -d redis clickhouse zookeeper kafka
    
    # 等待服务启动
    log_info "等待基础服务启动..."
    sleep 30
    
    # 启动应用服务
    log_info "启动应用服务..."
    docker-compose up -d trading-system
    
    # 启动监控服务
    log_info "启动监控服务..."
    docker-compose up -d prometheus grafana kafka-ui
    
    # 启动代理服务
    log_info "启动代理服务..."
    docker-compose up -d nginx
    
    log_info "所有服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_step "等待服务就绪..."
    
    # 等待交易系统启动
    log_info "等待AI交易系统启动..."
    timeout=300
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/metrics &> /dev/null; then
            log_info "AI交易系统已就绪"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    if [ $timeout -le 0 ]; then
        log_error "AI交易系统启动超时"
        exit 1
    fi
    
    # 等待Grafana启动
    log_info "等待Grafana启动..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 &> /dev/null; then
            log_info "Grafana已就绪"
            break
        fi
        sleep 5
        timeout=$((timeout - 5))
    done
    
    log_info "所有服务已就绪"
}

# 显示服务状态
show_status() {
    log_step "显示服务状态..."
    
    cd "$PROJECT_DIR"
    docker-compose ps
    
    echo ""
    log_info "服务访问地址："
    echo "  - Grafana监控面板: http://localhost:3000 (admin/admin123)"
    echo "  - Prometheus指标: http://localhost:9090"
    echo "  - Kafka UI: http://localhost:8080"
    echo "  - AI交易系统指标: http://localhost:8000/metrics"
    echo "  - 系统总览: http://localhost"
    echo ""
    log_info "日志查看："
    echo "  - 查看所有日志: docker-compose logs -f"
    echo "  - 查看交易系统日志: docker-compose logs -f trading-system"
    echo "  - 查看实时日志: tail -f logs/trading.log"
}

# 健康检查
health_check() {
    log_step "执行健康检查..."
    
    local all_healthy=true
    
    # 检查Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        log_info "✓ Redis健康"
    else
        log_error "✗ Redis不健康"
        all_healthy=false
    fi
    
    # 检查ClickHouse
    if curl -f http://localhost:8123/ping &> /dev/null; then
        log_info "✓ ClickHouse健康"
    else
        log_error "✗ ClickHouse不健康"
        all_healthy=false
    fi
    
    # 检查Kafka
    if docker-compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &> /dev/null; then
        log_info "✓ Kafka健康"
    else
        log_error "✗ Kafka不健康"
        all_healthy=false
    fi
    
    # 检查AI交易系统
    if curl -f http://localhost:8000/metrics &> /dev/null; then
        log_info "✓ AI交易系统健康"
    else
        log_error "✗ AI交易系统不健康"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        log_info "所有服务健康检查通过"
        return 0
    else
        log_error "部分服务健康检查失败"
        return 1
    fi
}

# 停止服务
stop_services() {
    log_step "停止服务..."
    
    cd "$PROJECT_DIR"
    docker-compose down
    
    log_info "服务已停止"
}

# 清理
cleanup() {
    log_step "清理资源..."
    
    cd "$PROJECT_DIR"
    docker-compose down -v --remove-orphans
    docker system prune -f
    
    log_info "清理完成"
}

# 显示帮助
show_help() {
    echo "秒级AI交易系统部署脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  deploy      完整部署系统"
    echo "  start       启动服务"
    echo "  stop        停止服务"
    echo "  restart     重启服务"
    echo "  status      显示服务状态"
    echo "  health      健康检查"
    echo "  logs        查看日志"
    echo "  cleanup     清理资源"
    echo "  help        显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 deploy   # 完整部署"
    echo "  $0 status   # 查看状态"
    echo "  $0 logs     # 查看日志"
}

# 查看日志
show_logs() {
    cd "$PROJECT_DIR"
    if [ $# -eq 0 ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$1"
    fi
}

# 主函数
main() {
    case "${1:-deploy}" in
        deploy)
            log_info "开始部署秒级AI交易系统..."
            check_dependencies
            create_directories
            generate_configs
            build_images
            start_services
            wait_for_services
            health_check
            show_status
            log_info "🚀 部署完成！"
            ;;
        start)
            start_services
            wait_for_services
            show_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 5
            start_services
            wait_for_services
            ;;
        status)
            show_status
            ;;
        health)
            health_check
            ;;
        logs)
            shift
            show_logs "$@"
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"