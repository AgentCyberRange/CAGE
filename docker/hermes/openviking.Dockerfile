FROM cage/hermes:pentestenv

# Install OpenViking server (uses Tsinghua PyPI mirror)
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple openviking

# Prepare directories (ov.conf is mounted at runtime from host)
RUN mkdir -p /home/agent/.openviking/data /home/agent/.openviking/logs && \
    chown -R agent:agent /home/agent/.openviking

CMD ["sleep", "infinity"]