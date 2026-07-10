# 把自定义Agent 接到 agent_pentest_bench 上

[English](adding-a-new-agent.md) · **中文**

手把手把**你写的自定义 Agent** 接到 `agent_pentest_bench` 上,跑一个真实靶标,在
inspector 里看它得分。下面每条命令都能照抄——benchmark、sample、model 全是真的。

- 想跑**现成**的 codex / claude_code,看 [benchmark 的 README](../examples/agent_pentest_bench/README.md);这里讲的是**加你自己的**。
- 仓库里已有的自定义 agent 真身,照着抄最省:[`qitos`](../cage/agents/custom/qitos/README.md)(单镜像,最简单)、[`cairn`](../cage/agents/custom/cairn/README.md)(多镜像 DinD)。你的 agent 就跟它们并排放。

> **唯一的硬规矩:** 你的模型客户端必须调 Cage 给的 `{base_url}`(容器内的 proxy),
> 每次调用才会被拦截、记录、打分。硬编码别的地址 = 什么都记不到、打不了分。

> **想要另一种模式** —— 保留你自己的 harness、让 agent 通过 API 自己拉靶场
> (`list → launch → attack → submit → close`)、完全不接进 Cage?那是
> [Benchmark-only(serve)模式](benchmark-serve-mode-CN.md)。代价是拿不到记录的
> trajectory,换来零集成——取舍见那篇的对比表。

## 0. 准备测试靶场环境

用自带的 `pb-postexp-range-4`(离线可跑)。取靶三步少一步都 build 不起来;靶标必须先
单独 build——`cage run` 从不建靶:

```bash
git lfs install
git submodule update --init --recursive examples/agent_pentest_bench/datasets/post_exploit_bench
git -C examples/agent_pentest_bench/datasets/post_exploit_bench lfs pull
cage benchmark build post_exploit_bench --sample pb-postexp-range-4
```

模型在 `config/models.yml` 里,下面用 `gpt-5.5`;没配就照 benchmark README 第 2 节的
`cage model set …` 加一个 OpenAI 兼容的。

## 1. 拷贝 custom agent 代码到 `cage/agents/custom/`

你的 agent 跟 qitos、cairn 并排住,随仓库一起走:

```
cage/agents/custom/<your_agent_name>/
├── agent.yml
└── <your_agent_name>/     # 你的代码,或一个入口脚本
```

## 2. 构建 dockerfile `docker/<your_agent_name>/Dockerfile`

agent 在容器里跑,你得给它一个镜像——**自己写一个 Dockerfile**,装好你的依赖、再满足
Cage 的容器约定。照抄 qitos 的 [`docker/qitos/Dockerfile`](../docker/qitos/Dockerfile) 最省。
约定就这几条:

```dockerfile
# docker/<your_agent_name>/Dockerfile
FROM ubuntu:22.04                                         # ← 换成带你所需工具的基础镜像
RUN apt-get update && apt-get install -y python3 python3-pip \
 && /usr/bin/python3 -m pip install --no-cache-dir httpx h2 <你的依赖>   # httpx/h2 是 sidecar proxy 要的
COPY cage/proxy/sidecar.py /opt/cage-proxy/container_proxy.py           # 容器内 proxy
RUN useradd -m agent && mkdir -p /home/agent/workspace /opt/cage-agent \
 && chown -R agent:agent /home/agent /opt/cage-agent                    # 非 root 的 agent 用户
ENV HOME=/home/agent
CMD ["sleep", "infinity"]
```

**构建**——和 qitos/cairn 一样,用 `cage agent build` 一条命令搞定:在 `agent.yml`(下一步)
里声明一个构建脚本,`cage agent build` 就去跑它。

```yaml
# agent.yml 里加这一段
build:
  script: docker/<your_agent_name>/build.sh
```

```bash
# docker/<your_agent_name>/build.sh —— 单镜像就一行(-t 的 tag 要和 agent.yml 的 image: 一致)
docker build -f docker/<your_agent_name>/Dockerfile -t cage/<your_agent_name>:latest .
```

```bash
cage agent build --agent <your_agent_name>
```

> 嫌多写一个 `build.sh`?单镜像也可以不声明 `build:`,直接跑那条 `docker build`。多镜像
> (像 cairn 烤 3 个)或构建前要 fetch 子模块(像 qitos),就必须走 `build.sh` 这条路。

> **你的 agent 是 LangGraph / LangChain 写的?** 上面这些样板 + langgraph 全家桶 + node
> 自动 trace,仓库已经烤成了 `cage/custom-langgraph:base`。这种情况直接
> `FROM cage/custom-langgraph:base` 再装你的依赖就行,不用从头写。

## 3. 创建 `agent.yml`(告诉 Cage 怎么启动你的 agent)

在 `cage/agents/custom/<your_agent_name>/agent.yml` 建这个文件——它说明:用哪个镜像、
怎么启动你的程序、把模型请求接到哪。**只有 `image` 和 `command` 必填。** 最小模板,
照抄改成你的:

```yaml
name: <your_agent_name>
image: cage/<your_agent_name>:latest
command: >-
  python3 -m <your_agent_name> {workspace_dir}
  --model {model_name}
  --instruction {task_instruction}
env:
  OPENAI_BASE_URL: "{base_url}"
  OPENAI_API_KEY: "{api_key}"
  OPENAI_MODEL: "{model_name}"
```

- `command` 就是**启动你自己程序**的那一行,把 `{...}` 当命令行参数传进去。
- `{...}` 是 Cage 每个 trial 替你填的保留占位符:`{task_instruction}`(任务,已 shell
  转义)、`{model_name}` / `{api_key}`、`{base_url}`(proxy 地址,OpenAI 协议带 `/v1`)、
  `{workspace_dir}`(`/home/agent/workspace`)、`{max_rounds}`(轮次上限)。其余任何
  `{xxx}` 都是你自己的 param,`--param k=v` 覆盖。
- **照抄一个真实的:** [`qitos/agent.yml`](../cage/agents/custom/qitos/agent.yml)(单镜像,
  最接近这个模板)、[`cairn/agent.yml`](../cage/agents/custom/cairn/agent.yml)(带 `build:`
  脚本 + 多镜像)。

## 4. 接进 `default_post_exploit.yml`

`cage run --agent <name>` 只认 yaml 里**已声明**的 agent(没有自动发现),所以要在
`examples/agent_pentest_bench/default_post_exploit.yml` 的 `agents:` 下加一条,`source:`
指到第 1 步的目录。**三行就够**——`home` / `max_concurrent` 之类都有默认,可省:

```yaml
  - id: <your_agent_name>
    source: ../../cage/agents/custom/<your_agent_name>
    models: [gpt-5.5]
```

## 5. 跑一个 trial

```bash
cage run post_exploit_bench \
  --agent <your_agent_name> \
  --model gpt-5.5 \
  --sample pb-postexp-range-4 \
  --prompt-level l0 \
  --passk 1 \
  --max-concurrent 1 \
  --run-id smoke-001
```

`--prompt-level l0` 不给提示(`l1`/`l2` 逐级泄露漏洞位置/网络拓扑);`--passk 1` 跑一次;
`--run-id` 给这次运行命名。

## 6. 看结果

`cage run` 跑完自动开 inspector 并打印 URL——点进你的 run → trial,看到完整 trajectory
(每次 LLM / 工具调用)和分数就成了。

