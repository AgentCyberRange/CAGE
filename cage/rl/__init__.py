"""RL training integration (Layer 1, mechanism only).

Cage can act as the environment + rollout + scoring layer for an external RL
trainer (e.g. verl): the trainer serves the policy as an OpenAI-compatible
endpoint, Cage's agents drive it, and each trial's reward is reported back. This
package ships only the *mechanism* — the per-trial join key and a fault-tolerant
reward POST. The reward *value* is benchmark domain knowledge and lives in
``Benchmark.reward`` (Layer 2). Everything here is inert unless a model declares
``rl_reward_sink``.
"""
