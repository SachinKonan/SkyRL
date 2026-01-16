# SkyRL Metrics Guide

This document explains how all metrics logged to WandB are calculated.

## Table of Contents
- [Generation Metrics](#generation-metrics)
- [Reward Metrics](#reward-metrics)
- [Loss Metrics](#loss-metrics)
- [Policy Metrics](#policy-metrics)
- [Timing Metrics](#timing-metrics)

---

## Generation Metrics

All generation metrics are computed in `skyrl_train/generators/utils.py::get_rollout_metrics()`.

### Token Count Metrics

**Important**: `response_ids` includes **all tokens after the initial prompt**, which means both assistant-generated tokens AND environment observation tokens (e.g., search results).

```python
# From skyrl_gym_generator.py
response_ids = input_ids[initial_prompt_length:]  # Everything after prompt

# From utils.py::get_rollout_metrics()
num_tokens_arr = np.array([len(response) for response in responses])
```

#### `generate/avg_num_tokens`
Average response length across all trajectories in the batch.

```python
avg_num_tokens = np.mean(num_tokens_arr)

# Example with batch_size=128, n_samples_per_prompt=10 (1280 trajectories):
# num_tokens_arr = [3100, 2950, 3200, ...]  # 1280 values
# avg_num_tokens = 3050
```

#### `generate/min_num_tokens` / `generate/max_num_tokens`
Minimum and maximum response lengths in the batch.

```python
min_num_tokens = np.min(num_tokens_arr)
max_num_tokens = np.max(num_tokens_arr)
```

#### `generate/std_num_tokens`
Standard deviation of response lengths.

```python
std_num_tokens = np.std(num_tokens_arr)
```

#### `generate/avg_tokens_non_zero_rewards`
Average response length for trajectories that received reward > 0.

```python
flat_rewards_arr = np.array([sum(r) if isinstance(r, list) else r for r in rewards])
non_zero_rewards_arr = flat_rewards_arr > 0.0

avg_tokens_non_zero_rewards = np.mean(num_tokens_arr[non_zero_rewards_arr])
```

**Interpretation**: In `search_arxiv`, non-zero reward means the model produced a valid `\boxed{Accept}` or `\boxed{Reject}` answer (at least format reward of 0.1).

#### `generate/avg_tokens_zero_rewards`
Average response length for trajectories that received reward = 0.

```python
zero_rewards_arr = flat_rewards_arr == 0.0

avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr])
```

**Interpretation**: Zero reward means no valid boxed answer was produced.

### Turn Count Metrics

#### `generate/avg_num_turns` / `generate/min_num_turns` / `generate/max_num_turns`
Number of LLM generation calls (model turns) per trajectory.

```python
all_num_turns = []
for traj_steps in all_steps:
    # Count steps where type == "model"
    model_turns = sum(1 for step in traj_steps if step["type"] == "model")
    all_num_turns.append(model_turns)

avg_num_turns = np.mean(all_num_turns)
min_num_turns = np.min(all_num_turns)
max_num_turns = np.max(all_num_turns)
```

---

## Reward Metrics

Computed in `skyrl_train/generators/utils.py::get_avg_reward_and_pass_at_n()`.

### `reward/avg_raw_reward`
Mean reward across all trajectories.

```python
# Sum token-level rewards to get trajectory reward
flat_rewards = []
for r in rewards:
    if isinstance(r, list):
        flat_rewards.append(sum(r))  # Token-level: sum to trajectory
    else:
        flat_rewards.append(r)

mean_raw_reward = np.mean(flat_rewards)
```

### `reward/avg_pass_at_N`
Fraction of unique prompts (UIDs) where at least one trajectory got reward > 0.

```python
uid_to_trajectory_rewards = defaultdict(list)
for i, uid in enumerate(uids):
    uid_to_trajectory_rewards[uid].append(flat_rewards[i])

# Count UIDs where max reward > 0
pass_count = sum(1 for rewards in uid_to_trajectory_rewards.values() if max(rewards) > 0)
pass_at_n = pass_count / len(uid_to_trajectory_rewards)

# Example with n_samples_per_prompt=10:
# UID "paper_1": [1.1, 0.1, 0.0, 1.1, ...] -> max > 0 -> pass
# UID "paper_2": [0.0, 0.0, 0.0, ...] -> max = 0 -> fail
# pass_at_10 = num_passing_uids / total_uids
```

### `reward/reward_std_in_group`
Average standard deviation of rewards within each group (same UID).

```python
stds = []
for uid, uid_rewards in uid_to_trajectory_rewards.items():
    if len(uid_rewards) > 1:
        stds.append(np.std(uid_rewards))
    else:
        stds.append(0.0)

reward_std_in_group = np.mean(stds)
```

**Why it matters for GRPO**: If this is 0, all samples for each prompt get identical rewards, meaning GRPO advantages are 0 and there's no learning signal.

```python
# Example with n_samples_per_prompt=10:
# UID "paper_1" rewards: [1.1, 0.1, 1.1, 0.0, 1.1, 0.1, 1.1, 1.1, 0.0, 1.1]
# std = 0.47

# UID "paper_2" rewards: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# std = 0.0 (no variance = no learning signal for this group)

# reward_std_in_group = mean([0.47, 0.0, ...])
```

### `reward/avg_majority_pass_at_k`
Fraction of UIDs where majority vote prediction matches ground truth.

```python
uid_to_predictions = defaultdict(list)
uid_to_ground_truth = {}

for i, uid in enumerate(uids):
    pred = parse_prediction_from_response(decoded_responses[i])  # "Accept" or "Reject"
    if pred:
        uid_to_predictions[uid].append(pred)
    uid_to_ground_truth[uid] = ground_truths[uid]

correct = 0
total = 0
for uid, predictions in uid_to_predictions.items():
    if predictions:
        # Majority vote
        majority = Counter(predictions).most_common(1)[0][0]
        if majority == uid_to_ground_truth[uid]:
            correct += 1
        total += 1

avg_majority_pass_at_k = correct / total
```

---

## Loss Metrics

Computed in `skyrl_train/trainer.py`.

### `loss/avg_final_rewards`
Average trajectory reward after all processing (KL penalty, advantage computation).

```python
# In trainer.py::compute_advantages_and_returns()
avg_rewards = data.metadata["avg_rewards"]  # From postprocess step
```

### `loss/avg_raw_advantages`
Mean advantage value across all response tokens.

```python
valid_advantages = torch.masked_select(
    data["advantages"],
    data["response_mask"].bool()  # Only where loss_mask=1
)
avg_advantages = valid_advantages.mean()
```

### `loss/avg_raw_advantages_abs`
Mean absolute advantage (useful when advantages are centered around 0).

```python
avg_advantages_abs = valid_advantages.abs().mean()
```

### `loss/avg_kl`
Average KL divergence between policy and reference model.

```python
# In trainer.py::compute_ref_log_probs()
kl = rollout_logprobs - ref_log_probs  # Per-token KL
avg_kl = (kl * response_mask).sum() / response_mask.sum()
```

### `loss/avg_kl_max`
Maximum KL divergence across tokens (useful for detecting distribution collapse).

```python
avg_kl_max = kl.max()
```

### `loss/kl_loss_coef`
Current KL loss coefficient (may change with adaptive KL control).

```python
kl_loss_coef = self.cfg.trainer.algorithm.kl_loss_coef  # e.g., 0.001
```

---

## Policy Metrics

Computed in `skyrl_train/workers/worker.py::training_step()`.

### `policy/policy_loss`
PPO clipped policy gradient loss.

```python
# Simplified from worker.py
ratio = torch.exp(new_log_probs - old_log_probs)
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
policy_loss = torch.max(pg_loss1, pg_loss2).mean()
```

### `policy/policy_entropy`
Entropy of the policy distribution (higher = more exploration).

```python
entropy = -(log_probs * probs).sum(dim=-1).mean()
```

### `policy/policy_kl`
KL divergence loss term (when `use_kl_loss=true`).

```python
# Computed during training step
kl_loss = kl_loss_coef * kl.mean()
```

### `policy/policy_lr`
Current learning rate (changes with scheduler).

```python
policy_lr = optimizer.param_groups[0]["lr"]
```

### `policy/raw_grad_norm`
Gradient norm before clipping (useful for monitoring training stability).

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

### `policy/response_length`
Number of tokens in the batch being trained on.

```python
response_length = response_mask.sum().item()
```

### `policy/clipfrac`
Fraction of tokens where the ratio was clipped.

```python
clipfrac = ((ratio - 1).abs() > eps_clip).float().mean()
```

### `policy/rollout_train_prob_diff_mean` / `policy/rollout_train_prob_diff_std`
Difference between rollout log-probs and training log-probs (staleness indicator).

```python
prob_diff = rollout_logprobs - new_log_probs
prob_diff_mean = prob_diff.mean()
prob_diff_std = prob_diff.std()
```

---

## GRPO Advantage Computation

For `advantage_estimator: "grpo"`, advantages are computed per-group (same UID).

```python
# From ppo_utils.py::compute_grpo_outcome_advantage()

# 1. Sum token-level rewards to get trajectory reward
scores = token_level_rewards.sum(dim=-1)  # [batch_size]

# 2. Group by UID
id2score = defaultdict(list)
for i in range(batch_size):
    id2score[uid[i]].append(scores[i])

# 3. Compute group mean and std
for uid in id2score:
    id2mean[uid] = torch.mean(torch.tensor(id2score[uid]))
    id2std[uid] = torch.std(torch.tensor(id2score[uid]))

# 4. Normalize within group
for i in range(batch_size):
    if grpo_norm_by_std:
        advantage[i] = (scores[i] - id2mean[uid[i]]) / (id2std[uid[i]] + epsilon)
    else:
        advantage[i] = scores[i] - id2mean[uid[i]]

# 5. Broadcast to all response tokens
advantages = advantage.unsqueeze(-1) * response_mask
```

**Example**:
```python
# Group with UID "paper_1", n_samples_per_prompt=10
scores = [1.1, 0.1, 1.1, 0.0, 1.1, 0.1, 1.1, 1.1, 0.0, 1.1]
mean = 0.68
std = 0.47

# Advantages (normalized):
# traj with reward 1.1: (1.1 - 0.68) / 0.47 = 0.89  (positive, upweight)
# traj with reward 0.0: (0.0 - 0.68) / 0.47 = -1.45 (negative, downweight)
```

---

## Timing Metrics

All timing metrics are prefixed with `timing/` and measured in seconds.

| Metric | Description |
|--------|-------------|
| `timing/generate` | Time spent in generation loop |
| `timing/compute_ref_log_probs` | Time computing reference model log-probs |
| `timing/compute_advantages_and_returns` | Time computing advantages |
| `timing/train` | Time in training loop |
| `timing/eval` | Time in evaluation |

---

## Environment-Specific Metrics

For `search_arxiv` environment, additional metrics are logged under `environment/search_arxiv/`:

| Metric | Description |
|--------|-------------|
| `environment/search_arxiv/avg_num_search_calls` | Average number of search API calls per trajectory |
| `environment/search_arxiv/avg_turns` | Average number of turns taken |

---

## Interpreting Common Patterns

### High `avg_num_tokens` but low `avg_tokens_non_zero_rewards`
Model is generating long responses but failing to produce valid answers.

### `reward_std_in_group` = 0
All trajectories for each prompt get identical rewards. Check:
- Is reward function deterministic?
- Is model always producing same output (low temperature)?

### `avg_num_tokens` ≈ `avg_tokens_non_zero_rewards`
Most trajectories are succeeding (good format compliance).

### `policy/clipfrac` too high (> 0.3)
Training is unstable, ratio is being clipped too often. Consider:
- Lower learning rate
- Increase KL penalty
- Check for reward hacking

### `loss/avg_kl` increasing rapidly
Model is diverging from reference. Consider:
- Increase `kl_loss_coef`
- Lower learning rate
