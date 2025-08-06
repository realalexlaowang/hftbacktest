# 多阶段构建：基础镜像
FROM python:3.11-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装额外的性能优化包
RUN pip install --no-cache-dir \
    uvloop \
    orjson \
    cython \
    psutil

# 生产镜像
FROM base as production

# 创建非root用户
RUN useradd -m -u 1000 trader && \
    mkdir -p /app/logs /app/models /app/config && \
    chown -R trader:trader /app

# 切换到非root用户
USER trader

# 复制应用代码
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader config/ ./config/

# 设置Python路径
ENV PYTHONPATH=/app/src

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露端口
EXPOSE 8000 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# 启动命令
CMD ["python", "src/main.py"]