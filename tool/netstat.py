import re
import subprocess


def kill_process_using_port(port):
    # 使用 subprocess.run 执行命令
    result = subprocess.run(['netstat', '-tulpn'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 使用 grep 过滤输出
    result_lines = result.stdout.splitlines()
    filtered_lines = [line for line in result_lines if 'LISTEN' in line and f"{port}" in line]

    # 打印过滤后的结果
    for line in filtered_lines:
        pid_match = re.search(r"\b(\d+)/python3\b", line)
        if pid_match:
            pid = pid_match.group(1)
            subprocess.run(["kill", pid])
