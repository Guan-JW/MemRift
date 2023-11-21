import time, psutil, subprocess

interval = 0.2
max_gpu = 0
max_cpu = 0

while True:
    # GPU: 统计所有卡的总用量（单位MB）
    try:
        # 调用 tegrastats
        out = subprocess.check_output("tegrastats --interval 100 --logfile /tmp/ts.log & sleep 0.1; pkill tegrastats; cat /tmp/ts.log", shell=True)
        text = out.decode("utf-8")
        # 解析 GPU/GR3D 内存
        match = re.search(r'RAM (\d+)/(\d+)MB', text)
        if match:
            used = int(match.group(1))
            if used > max_gpu:
                max_gpu = used
    except Exception:
        pass

    # CPU: 统计系统总已用内存（单位字节）
    cpu_mem = psutil.virtual_memory().used
    if cpu_mem > max_cpu:
        max_cpu = cpu_mem

    # 打印最新峰值
    print(f"[{time.strftime('%H:%M:%S')}] Peak CPU: {max_cpu/1024/1024:.1f} MB, Peak GPU: {max_gpu/1024:.1f} MB, Total: {max_cpu/1024/1024 + max_gpu/1024:.1f}", flush=True)

    time.sleep(interval)
