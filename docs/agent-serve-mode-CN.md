# CAGE 拉取式(serve)评测模式:外部 Agent 自驱动的靶场接入

[English](agent-serve-mode.md) · **中文**

CAGE 的 benchmark-only(serve)模式把一个 benchmark 暴露成一组按需实例化的隔离靶场,由**外部 agent 自己驱动** `list → launch → attack → submit → close` 这条回路;CAGE 不托管 agent 运行时,也不拦截其模型调用。相比集成式(CAGE 接管,`cage run`),serve 模式放弃轨迹采集、换来近乎零集成,适合外部、黑盒或异构语言实现的 agent,以及自助评测与打榜。

## 哪种模式适合你?(快速判断)

**如果你的 agent 已经有自己成熟的日志系统和查看运行的前端,那 benchmark-only(serve)模式更适合你。** CAGE 接管模式的核心价值就是把执行轨迹采集下来、在 CAGE 自带的 inspector 里回放;你若已经有这套,再把 agent 塞进 CAGE 的容器 + proxy 约定纯属额外负担——直接让现成的 agent 去打服务出来的靶场,按你自己的方式记录即可。

反过来,只有当你**希望由 CAGE 替你采集并标准化轨迹**(通常是还没有自己观测体系的新 agent)时,才值得去做 CAGE 接管那条更繁琐的集成。

一句话:成熟 agent / 框架(LangGraph、已有 harness、团队自研 agent)→ **serve 模式**;想让 CAGE 端到端记录的新 agent → CAGE 接管。

---

## 两种接入方式对比

CAGE 支持以两种范式评测自研 Agent,二者的本质差异在于 **CAGE 是否托管并记录 Agent 的执行过程**。

- **集成式(CAGE 接管)**,详见[《接入自定义 Agent》](agent-cage-managed-CN.md)。调用方将 Agent 以容器形式接入 CAGE,由 `cage run` 统一编排整条 trial:构建容器、经容器内代理(proxy)**拦截每一次模型调用**、执行前后状态快照、并完成判分。
- **拉取式(benchmark-only / serve,本文)**。CAGE 仅提供可实例化的隔离靶场,评测回路由**外部 Agent 自行驱动**。CAGE 不运行该 Agent,因而无法观测其模型调用。

**表 1** 两种范式的对比

| 维度 | 拉取式 / serve(本文) | 集成式 / CAGE 接管([文档](agent-cage-managed-CN.md)) |
|---|---|---|
| Agent 执行方 | 外部进程(调用方) | CAGE(`cage run`) |
| 集成成本 | 近乎为零:无需 Dockerfile / `agent.yml` / proxy 约定 | 需 Dockerfile 与 `agent.yml`,且模型客户端须指向 CAGE 提供的 `{base_url}` |
| 语言与框架 | 任意;CAGE 不介入 Agent 代码 | 任意,但须运行于 CAGE 的容器模型内 |
| 执行轨迹(逐次 LLM / 工具调用) | 不采集;仅返回最终分值与判定 | 全量采集;proxy 逐条记录,可于 inspector 复盘 |
| 中途快照 / 断点续跑 | 不支持 | 支持 |
| 跨 Agent 可比性 | 较弱;运行时由调用方掌控 | 强;运行时经 CAGE 标准化 |
| 回路 / 重试 / 并发的管理方 | 调用方 | CAGE |
| 结果落入 `.cage_runs` 及 inspector | 是(仅判定) | 是(判定 + 完整轨迹) |

**选型准则**:若需 CAGE 对执行过程进行观测、记录并支持严格横向比较,且 Agent 可容器化,应采用**集成式**;若仅需将外部或黑盒 Agent 接入靶场并获得分值、追求零集成,则采用 **serve 模式**。

## 1 环境准备

与常规 CAGE 评测一致,靶标须预先构建;`cage run` 与 serve 均不在评测期建靶。下例采用随仓库分发、可离线运行的 `web_exploit_bench` 靶标 `pb-prestashop`:

```bash
git lfs install
git submodule update --init --recursive examples/agent_pentest_bench/datasets/web_exploit_bench
git -C examples/agent_pentest_bench/datasets/web_exploit_bench lfs pull
cage benchmark build web_exploit_bench --sample pb-prestashop
```

`web_exploit_bench` 靶标的判分还需在服务端配置一个裁判模型(见第 2 节);`post_exploit_bench` 靶标则无此需求。

## 2 配置模型(web_exploit_bench 判分需要)

| 靶标类型 | 判分信号 | 是否需要模型 |
|---|---|---|
| `web_exploit_bench` | `verifier`(evaluator 运行 `verify.py`)+ **`LLM_judge`** | 部分需要 |
| `post_exploit_bench` | 靶机实时状态(marker) | 不需要 |


**配置步骤如下。**

**(1) 复制本地模型注册表。** `config/models.yml` 为本地文件(含密钥,已被 gitignore,不入库),由示例文件生成:

```bash
cp config/models.example.yml config/models.yml
```

**(2) 登记裁判模型。** 以 OpenAI 兼容端点为例(字段含义详见[《模型注册表》](models.md)):

```bash
export DEEPSEEK_API_KEY=...
cage model set deepseek-v4-pro \
  --provider openai \
  --model deepseek-v4-pro \
  --endpoint https://<your-endpoint>/v1 \
  --api-key '${DEEPSEEK_API_KEY}'
cage model list      # 确认已登记
```

**(3) 默认裁判与覆盖。** `agent_pentest_bench` 的 `web_exploit_bench` 题目已在其配置中声明默认裁判为 `deepseek-v4-pro`,故:

- 若 `config/models.yml` 中存在 id 为 `deepseek-v4-pro` 的条目,则 `cage benchmark serve agent_pentest_bench` **开箱即用**,无需额外参数;
- 若改用其他模型,以 `--judge-model <id>` 覆盖,其中 `<id>` 必须是 `config/models.yml` 中已登记的条目:

```bash
cage benchmark serve agent_pentest_bench --judge-model <your-model-id>
```

裁判模型 id 于判分时对 `config/models.yml` 解析:若该 id 不存在,`web_exploit_bench` 题目的 `LLM_judge` 信号将报"模型未配置",相应漏洞无法判为通过。`post_exploit_bench` 靶标不涉及模型,可跳过本节。

## 3 服务启动

```bash
cage benchmark serve agent_pentest_bench
#   console : http://localhost:8000/     (只读,供运维者查看)
#   api     : http://localhost:8000/challenges
#   judge   : deepseek-v4-pro (benchmark default)
```

该命令自动发现并挂载此 benchmark 的全部题目索引(`web_exploit_bench` 与 `post_exploit_bench`)。裁判模型可经 `--judge-model <id>` 覆盖(见第 2 节);对外暴露(供他机访问)可用 `--host 0.0.0.0 --external-token "$(openssl rand -hex 16)"`。

## 4 评测回路(Python SDK)

为避免手工构造 HTTP 与 multipart 报文,建议使用随仓库提供的**纯标准库**客户端(`from cage.target.serve_client import ServeClient`;亦可将单文件 `cage/target/serve_client.py` 直接复制入 Agent 工程):

```python
from cage.target.serve_client import ServeClient

client = ServeClient("http://localhost:8000", client_id="team-red")

# (1) 枚举题目——不实例化任何容器
for ch in client.list_challenges():
    print(ch["id"], ch["category"])

# (2) 开靶——获得专属隔离实例:唯一 run_id + 独立 docker 网络 + 全新容器。
#     多个 Agent 可并发 launch 同一题目而互不干扰(target_scope=per_agent、
#     network_only=True 为默认值)。
inst = client.launch("pb-prestashop")
print(inst.run_id, inst.container_addr)   # 如 ['172.31.x.y:80'],位于 inst.network_name 上

# (3) 获取任务简报——与集成式一字不差的任务描述(任务、靶机地址、final_answer
#     输出契约),且已填入**本实例**的实时靶机地址,可直接下发给 Agent。
task = inst.task_prompt()

# (4) 攻击——将 Agent 以容器形式运行于 serve 宿主机、接入该实例网络,并访问
#     inst.container_addr(反作弊机制见第 5 节)。以下接入一个已启动的容器:
client.attach(inst, "my-agent-container")
#     ... Agent 依据 task 攻击靶机,并将结果写入 final_answer/<vuln_id>.json ...

# (5) 提交——对**仍在运行**的靶机判分。每个实例仅可提交一次(终局;重复提交返回
#     同一判定,already_submitted=True)。
verdict = inst.submit(final_answer_dir="./final_answer")
print(verdict["scores"])   # {scorer: {value, answer, explanation, metadata}}

# (6) 关闭——销毁该实例,释放其容器与网络(亦可在 submit 时传 close=True 一并完成)。
client.close(inst)
```

### 4.1 任务简报的生成与字段

`inst.task_prompt()` 返回可直接下发的简报,其中**本实例的实时靶机地址已完成填充**。若需未填充的原始模板,`inst.prompt()` 返回完整响应 `{task_prompt, task_prompt_template, prompt_level, …}`,其中 `task_prompt_template` 为同一简报、但靶机地址以占位符(如 `{{APPLICATION_TARGETS}}`)呈现。简报由 benchmark 自身的 `build_prompt` 渲染,故与 `cage run` 下发给集成式 Agent 的内容完全一致。

### 4.2 提示档位的实例级绑定

**提示档位(hint tier)于 launch 时绑定至具体实例**,因此同一服务上的不同实例可运行于不同档位而**无需重启**:

```python
inst = client.launch("pb-postexp-range-4", prompt_level="l1")   # 该实例 → l1
```

其中 `l0` 表示无提示,`l1` / `l2` 逐级揭示漏洞位置与网络拓扑。未指定时采用服务级默认值 `--prompt-level`(缺省为 `l0`)。档位是**记录于实例之上的 launch 参数**,由发起 launch 的一方决定;若需严格公平评测(即 Agent 自助 launch),应由运维方统一驱动 launch,或经默认值确立策略。

### 4.3 实例回收

`close=True` 在判分后即时销毁实例;否则须显式调用 `client.close(inst)`,或使用上下文管理器 `with client.session("pb-prestashop") as inst:` 以保证退出时自动回收。**实例不会被自动回收**:未经 `DELETE` 的实例将持续占用其容器与网络。

### 4.4 `post_exploit_bench` 靶标的提交(marker,无载荷)

`post_exploit_bench` 靶标由靶机实时状态(marker)判分,故 `submit` 无需携带任何载荷——其语义即"攻击完成,请对已落下的 marker 判分":

```python
with client.session("pb-postexp-range-4") as inst:
    client.attach(inst, "my-agent-container")
    # ... Agent 于靶场内落下 marker ...
    print(inst.submit()["scores"])
```

### 4.5 评测流程(伪代码)

评测由调用方自行实现——语言、并发方式、agent 如何启动均自定;CAGE 只提供 §4 的 SDK。整体流程如下(伪代码):

```
配置:
  LEVELS        # 要评测的档位子集,如 {l0, l1, l2}
  CONCURRENCY   # 并发度
  MAX_ROUNDS    # 单题轮次上限(软终止条件)
  TIME_BUDGET   # 单题墙钟预算(硬终止条件)

并发(至多 CONCURRENCY 路)遍历每个 (level, chal) ∈ LEVELS × list_challenges():

    inst  ← launch(chal, prompt_level = level)    # 独立实例:唯一 run_id + 独立网络
    agent ← 起一个**本任务专属**的 agent 容器       # 并发不串扰的关键:每任务各一个
    attach(inst, agent)                           # 把 agent 接入本实例隔离网络(反作弊,§5)

    在 MAX_ROUNDS / TIME_BUDGET 内,让 agent 攻击 inst.container_addr:
        web_exploit_bench  → 按 inst.task_prompt() 的 Reporting 段产出漏洞报告
        post_exploit_bench → 在靶机落下 user / root marker

    verdict ← submit(inst, 报告目录)   # web 带报告目录;post 空载荷(实时 marker);一次性终局

    记录整份 verdict(见「审计」),而非只留分数
    close(inst);回收 agent 容器
```

**审计**——`verdict["scores"]` 远不止一个分值,而是**逐漏洞判定明细**:每个 vuln 的 `passed` / `verifier_status`(evaluator `verify.py` 的结论)/ `judge_status`,以及原始的 `verifier_results` 与 `judge_findings`。请把每题的 verdict,连同你下发的 `task` 与 agent 产出,一并留存,作为你侧的审计记录。**且无需只依赖这份自记**:CAGE 已在服务端把同一份判定持久化到 `.cage_runs`(见 §6),可用 `cage inspect` 独立复核——两侧互为交叉校验。

**并发与终止**——serve 采用 `per_agent`,每次 launch 都是独立实例,故天然可并行;唯一要求是**每个并发任务用独立的 agent 容器**(共用会同时接入多张实例网络而串扰)。终止为两层:`MAX_ROUNDS`(软,交由 agent 自控)与 `TIME_BUDGET`(硬,由你的调度强制),任一触发即收尾并提交。

**档位**——任务集是 `LEVELS × 全部题目`,同一题可在 `l0` / `l1` / `l2` 各评一遍;每题于 launch 时 `prompt_level = level` 绑定到其实例(见 §4.2)。只评一档就把 `LEVELS` 取单元素。

SDK 各调用(`list_challenges` / `launch` / `session` / `task_prompt` / `attach` / `submit` / `close`)见 §4;具体实现、并发库、agent 启动方式由你决定。

## 5 隔离与反作弊机制

`container_addr` 是**该实例隔离 docker 网络上**的地址,而非宿主机端口。受支持且抗作弊的部署方式为:**将 Agent 以容器形式运行于 serve 宿主机,并接入该实例网络**——既可于启动时接入(`docker run --network <inst.network_name> your-agent`),亦可事后接入(`client.attach(inst, container)`)——继而以对等节点(peer)身份访问靶机。

需要强调的是,**不得从宿主机直接访问靶机,亦不得向 Agent 授予 docker socket**。`network_only` 虽已剥离宿主机端口,却无法约束运行于宿主机之上的进程:宿主机对**每一个**实例的网桥子网均存在路由(宿主机进程既可访问自身靶机,亦可访问他人靶机);而一旦挂载 `/var/run/docker.sock`,进程即可经 `docker exec` 直接进入靶机读取 flag / marker。由于 docker 网络为宿主机本地资源,上述强隔离仅在"同机容器且仅接入自身实例网络"时成立;真正跨机的远程 Agent 无法加入该网络,只能退化为弱隔离的宿主机发布端口(`entry_urls`,`network_only=false`),此时端口将暴露于宿主机。

## 6 评测结果的持久化与可视化

每次 submit 将作为一条 trial 持久化至 `.cage_runs/serve__<client_id>/serve/`——即 **inspector 所读取的同一 `.cage_runs` 目录树**,故 `cage inspect` 会将其与 `cage run` 的结果并列展示。就语义而言,一次被服务的 benchmark 对应"每个外部 Agent 一个实验"(不同 `client_id` 各自拥有独立的评测 run),每次 submit 向其追加一条 trial。

调用方获得的是**判定(verdict)而非轨迹(trajectory)**:`trials/<id>/scores/<scorer>.json` 载有完整判分细节(逐漏洞的 `passed` / `verifier_status` / `judge_status`,以及原始的 `verifier_results` 与 `judge_findings`),`submit` 的返回值即为同一份数据。由于 CAGE 从未运行该 Agent,故**不存在逐步的 LLM / 工具调用轨迹**——此为 serve 模式的固有局限。如需轨迹,应改用[集成式](agent-cage-managed-CN.md)。

## 参考

- [Serve External Audience](serve-external-audience.md)——完整 HTTP 契约:各端点定义、两类 audience 的端口绑定模型、`--external-token` 鉴权、并发与隔离的内部机制,以及 SDK 各调用对应的原始 `curl` 形式。
- [《接入自定义 Agent》](agent-cage-managed-CN.md)——另一范式:由 CAGE 运行并记录 Agent。
