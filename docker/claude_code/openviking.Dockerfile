# Claude Code + OpenViking server — for stateful memory experiments.
#
# Builds on the base claude_code image and adds:
#   - openviking Python package (server + CLI)
#
# The user must provide ~/.openviking/ov.conf with embedding + VLM config.
# cage debug auto-mounts the host's ~/.openviking/ov.conf into the container.
#
# Usage:
#   cage build --agent claude_code --variant openviking
#   cage debug --agent claude_code --model ... --plugin openviking-memory \
#              --image cage/claude-code:openviking

FROM cage/claude-code:pentestenv

# Install OpenViking server (uses Tsinghua PyPI mirror)
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple openviking

# Prepare directories (ov.conf is mounted at runtime from host)
RUN mkdir -p /home/agent/.openviking/data /home/agent/.openviking/logs && \
    chown -R agent:agent /home/agent/.openviking

CMD ["sleep", "infinity"]
