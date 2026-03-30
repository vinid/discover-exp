"""
RL Training Loop for KernelBench.

This module implements the GRPO-style training loop using Tinker,
following the patterns from tinker_cookbook.rl.train.

The training loop:
1. Samples rollouts from the current policy
2. Evaluates kernels and computes rewards
3. Computes advantages within groups
4. Updates the model with importance sampling loss
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Any

import chz
import numpy as np
import tinker
import torch
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl.data_processing import (
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

from kernelbench_tinker.envs.kernelbench_env import KernelBenchDatasetBuilder
from kernelbench_tinker.training.completers import build_token_completer
from kernelbench_tinker.training.data_processing import assemble_training_data
from kernelbench_tinker.training.models import get_adam_params
from kernelbench_tinker.training.tensorboard_logger import (
    TensorBoardConfig,
    TensorBoardLogger,
    create_tensorboard_logger,
)
from kernelbench_tinker.training.trace_logger import TraceLogger, set_trace_logger


def remove_mask(datum: tinker.Datum) -> tinker.Datum:
    """Remove mask from datum loss_fn_inputs before sending to forward_backward.

    The Tinker API doesn't expect the mask key in loss_fn_inputs, so we need to
    remove it before sending the datum to forward_backward.
    """
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )

logger = logging.getLogger(__name__)


@chz.chz
class TrainingConfig:
    """Configuration for KernelBench RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    lora_rank: int = 32
    learning_rate: float = 1e-4

    # Generation configuration
    max_tokens: int = 4096
    temperature: float = 1.0
    use_two_phase_sampling: bool = False
    phase1_max_tokens: int = 26000
    context_window: int = 32768
    context_buffer: int = 50

    # Dataset configuration (single-turn default)
    dataset_builder: KernelBenchDatasetBuilder = chz.field(
        default_factory=KernelBenchDatasetBuilder
    )

    # Training configuration
    num_substeps: int = 1  # Optimizer steps per batch
    loss_fn: LossFnType = "importance_sampling"
    max_steps: int | None = None

    # KL regularization
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Logging and checkpointing
    log_path: str = "./runs/kernelbench_tinker"
    save_every: int = 10  # Save checkpoint every N batches

    # Remove groups where all rewards are the same (no learning signal)
    remove_constant_reward_groups: bool = True

    # Wandb logging
    wandb_project: str | None = None
    wandb_name: str | None = None

    # TensorBoard logging
    tensorboard_enabled: bool = True
    tensorboard_log_histograms_every: int = 5
    tensorboard_log_per_level: bool = True

    # Tinker API
    base_url: str | None = None

    # Resume from checkpoint
    load_checkpoint_path: str | None = None
    resume_from_batch: int | None = None  # Resume from specific batch number (uses existing checkpoints file)
    start_batch: int = 0  # Start batch index (used with load_checkpoint_path for new runs)


async def do_group_rollout_and_filter(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
    do_remove_constant_reward_groups: bool,
    use_two_phase_sampling: bool,
    phase1_max_tokens: int,
    context_window: int,
    context_buffer: int,
) -> TrajectoryGroup | None:
    """
    Perform rollouts for a group and optionally filter constant reward groups.

    Args:
        sampling_client: Tinker sampling client
        env_group_builder: Builder for environment group
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_remove_constant_reward_groups: Whether to filter constant groups

    Returns:
        TrajectoryGroup or None if filtered out
    """
    policy = build_token_completer(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
        use_two_phase_sampling=use_two_phase_sampling,
        phase1_max_tokens=phase1_max_tokens,
        context_window=context_window,
        context_buffer=context_buffer,
    )

    trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Filter if all rewards are the same
    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups([trajectory_group])
        if len(trajectory_groups) == 0:
            return None
        trajectory_group = trajectory_groups[0]

    return trajectory_group



def compute_trajectory_metrics(
    trajectory_groups: list[TrajectoryGroup],
    taglist: list[list[str]] | None = None,
) -> dict[str, Any]:
    """
    Compute aggregate metrics from trajectory groups.

    Args:
        trajectory_groups: List of trajectory groups
        taglist: Optional tags for each group

    Returns:
        Dictionary of metrics
    """
    metrics: dict[str, Any] = {}

    all_rewards = []
    all_format_ok = []
    all_compiled = []
    all_correct = []
    all_eval_times = []
    all_step_times = []
    all_ref_load_times = []
    all_modal_eval_times = []

    for tg in trajectory_groups:
        rewards = tg.get_total_rewards()
        all_rewards.extend(rewards)

        # Extract per-trajectory metrics
        for traj in tg.trajectories_G:
            for trans in traj.transitions:
                if trans.metrics:
                    all_format_ok.append(trans.metrics.get("format_ok", 0))
                    all_compiled.append(trans.metrics.get("compiled", 0))
                    all_correct.append(trans.metrics.get("correctness", 0))
                    if "time/eval" in trans.metrics:
                        all_eval_times.append(trans.metrics["time/eval"])
                    if "time/step_total" in trans.metrics:
                        all_step_times.append(trans.metrics["time/step_total"])
                    if "time/ref_load" in trans.metrics:
                        all_ref_load_times.append(trans.metrics["time/ref_load"])
                    if "time/modal_eval" in trans.metrics:
                        all_modal_eval_times.append(trans.metrics["time/modal_eval"])

    if all_rewards:
        metrics["reward/mean"] = float(np.mean(all_rewards))
        metrics["reward/std"] = float(np.std(all_rewards))
        metrics["reward/min"] = float(np.min(all_rewards))
        metrics["reward/max"] = float(np.max(all_rewards))

    if all_format_ok:
        metrics["kernel/format_rate"] = float(np.mean(all_format_ok))
    if all_compiled:
        metrics["kernel/compile_rate"] = float(np.mean(all_compiled))
    if all_correct:
        metrics["kernel/correct_rate"] = float(np.mean(all_correct))
    if all_eval_times:
        metrics["time/eval_mean"] = float(np.mean(all_eval_times))
        metrics["time/eval_max"] = float(np.max(all_eval_times))
    if all_step_times:
        metrics["time/step_mean"] = float(np.mean(all_step_times))
        metrics["time/step_max"] = float(np.max(all_step_times))
    if all_ref_load_times:
        metrics["time/ref_load_mean"] = float(np.mean(all_ref_load_times))
        metrics["time/ref_load_max"] = float(np.max(all_ref_load_times))
    if all_modal_eval_times:
        metrics["time/modal_eval_mean"] = float(np.mean(all_modal_eval_times))
        metrics["time/modal_eval_max"] = float(np.max(all_modal_eval_times))

    metrics["batch/num_groups"] = len(trajectory_groups)
    metrics["batch/num_trajectories"] = sum(
        len(tg.trajectories_G) for tg in trajectory_groups
    )

    return metrics



async def train_step(
    data: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
) -> list[torch.Tensor]:
    """
    Perform a training step with gradient accumulation.

    Args:
        data: List of training datums
        training_client: Tinker training client
        learning_rate: Learning rate
        num_substeps: Number of optimizer steps
        loss_fn: Loss function type

    Returns:
        List of training logprobs tensors
    """
    # Split data into substeps
    substep_size = max(1, len(data) // num_substeps)
    training_logprobs = []

    for i in range(0, len(data), substep_size):
        batch = data[i : i + substep_size]

        # Forward-backward pass (remove mask key from datums)
        fwd_bwd_future = await training_client.forward_backward_async(
            [remove_mask(d) for d in batch], loss_fn=loss_fn
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()

        # Extract logprobs
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs.append(output["logprobs"].to_torch())

        # Optimizer step
        adam_params = get_adam_params(learning_rate)
        optim_future = await training_client.optim_step_async(adam_params)
        await optim_future.result_async()

    return training_logprobs


async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    batch_idx: int,
    dataset_batch_idx: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    Save checkpoint and get updated sampling client.

    Args:
        training_client: Tinker training client
        batch_idx: Current batch index
        log_path: Path for saving logs/checkpoints
        save_every: Save checkpoint every N batches
        start_batch: Starting batch index

    Returns:
        Tuple of (sampling_client, metrics)
    """
    metrics: dict[str, Any] = {}

    with timed("save_checkpoint", metrics):
        # Always save checkpoint: at start (batch_idx == start_batch) or every save_every batches
        # This ensures we can resume even if training crashes early
        should_save = (
            save_every > 0 and (
                batch_idx == start_batch or  # Initial checkpoint
                (batch_idx > start_batch and batch_idx % save_every == 0)  # Periodic checkpoints
            )
        )
        if should_save:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=log_path,
                loop_state={"batch": batch_idx, "dataset_batch": dataset_batch_idx},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics


async def run_training_loop(
    cfg: TrainingConfig,
) -> None:
    """
    Main RL training loop for KernelBench.

    This implements synchronous on-policy training with GRPO-style
    grouped rollouts.

    Args:
        cfg: Training configuration
    """
    # Setup logging
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    # Setup TensorBoard logging
    tb_logger: TensorBoardLogger | None = None
    if cfg.tensorboard_enabled:
        tb_config = TensorBoardConfig(
            log_histograms_every=cfg.tensorboard_log_histograms_every,
            log_per_level_metrics=cfg.tensorboard_log_per_level,
        )
        tb_logger = create_tensorboard_logger(cfg.log_path, tb_config)
        tb_logger.log_training_config(cfg)

    logger.info("Starting KernelBench RL training")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Log path: {cfg.log_path}")
    if tb_logger:
        logger.info(f"TensorBoard: {tb_logger.log_dir}")

    # Set up trace logger (per-step prompts/outputs/evals)
    trace_logger = TraceLogger(Path(cfg.log_path) / "traces.jsonl")
    set_trace_logger(trace_logger)

    # Check for resume
    if cfg.resume_from_batch is not None:
        # Resume from a specific batch number (uses existing checkpoints file)
        checkpoints = checkpoint_utils.load_checkpoints_file(cfg.log_path)
        resume_info = None
        for ckpt in checkpoints:
            if ckpt.get("batch") == cfg.resume_from_batch and "state_path" in ckpt:
                resume_info = ckpt
                break
        if resume_info:
            start_batch = resume_info["batch"]
            dataset_batch_idx = resume_info.get("dataset_batch", start_batch)
            logger.info(f"Resuming from specific batch {start_batch}")
        else:
            raise ValueError(f"No checkpoint found for batch {cfg.resume_from_batch}")
    elif cfg.load_checkpoint_path:
        # Starting new run from external checkpoint
        resume_info = None  # Don't use optimizer state from checkpoints file
        start_batch = cfg.start_batch
        dataset_batch_idx = cfg.start_batch
        logger.info(f"Starting new run from external checkpoint at batch {start_batch}")
        logger.info(f"  Checkpoint: {cfg.load_checkpoint_path}")
    else:
        # Resume from last checkpoint
        resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
        if resume_info:
            start_batch = resume_info["batch"]
            dataset_batch_idx = resume_info.get("dataset_batch", start_batch)
            logger.info(f"Resuming from batch {start_batch}")
        else:
            start_batch = cfg.start_batch
            dataset_batch_idx = cfg.start_batch

    # Create Tinker clients
    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    if resume_info:
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"]
            )
        )
    elif cfg.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )

    tokenizer = training_client.get_tokenizer()

    # Create dataset (pass tokenizer for renderer)
    dataset_builder = cfg.dataset_builder
    logger.info("Using KernelBenchDatasetBuilder")

    train_dataset, test_dataset = await dataset_builder(tokenizer=tokenizer)
    total_batches = len(train_dataset)
    target_steps = total_batches
    if cfg.max_steps is not None:
        if cfg.max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        target_steps = min(target_steps, cfg.max_steps)
    logger.info(f"Training on up to {target_steps} steps")
    if cfg.max_steps is not None and target_steps != total_batches:
        logger.info(f"Capped training to {target_steps} steps with max_steps={cfg.max_steps}")

    if start_batch >= target_steps or dataset_batch_idx >= total_batches:
        logger.info("No training steps remaining")
        if tb_logger:
            tb_logger.flush()
            tb_logger.close()
        ml_logger.close()
        return

    # Get initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        start_batch,
        dataset_batch_idx,
        cfg.log_path,
        cfg.save_every,
        start_batch,
    )

    # Training loop
    completed_steps = start_batch
    while completed_steps < target_steps and dataset_batch_idx < total_batches:
        batch_idx = dataset_batch_idx
        step_idx = completed_steps
        dataset_batch_idx += 1
        t_start = time.time()
        metrics = {
            "progress/batch": step_idx,
            "progress/done_frac": (step_idx + 1) / target_steps,
            "optim/lr": cfg.learning_rate,
        }

        # Get batch of env group builders
        env_group_builders = train_dataset.get_batch(batch_idx)

        # Collect rollouts (single-turn)
        with timed("rollout", metrics):
            try:
                results = await asyncio.gather(*[
                    do_group_rollout_and_filter(
                        sampling_client,
                        builder,
                        tokenizer=tokenizer,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                        use_two_phase_sampling=cfg.use_two_phase_sampling,
                        phase1_max_tokens=cfg.phase1_max_tokens,
                        context_window=cfg.context_window,
                        context_buffer=cfg.context_buffer,
                    )
                    for builder in env_group_builders
                ], return_exceptions=True)
            except Exception:
                logger.exception("Group rollout failed during gather")
                raise

        # Filter out None (removed constant reward groups) and exceptions
        trajectory_groups = []
        for tg in results:
            if isinstance(tg, Exception):
                logger.error("Group rollout failed", exc_info=tg)
            elif tg is not None:
                trajectory_groups.append(tg)

        if len(trajectory_groups) == 0:
            logger.warning(
                f"Dataset batch {batch_idx}: All groups filtered out, skipping optimizer update"
            )
            continue

        # Compute metrics
        traj_metrics = compute_trajectory_metrics(trajectory_groups)
        metrics.update(traj_metrics)

        # Compute advantages and assemble training data
        with timed("assemble_data", metrics):
            advantages = compute_advantages(trajectory_groups)
            data, _metadata = assemble_training_data(trajectory_groups, advantages)

        # Training step
        with timed("train", metrics):
            await train_step(
                data,
                training_client,
                cfg.learning_rate,
                cfg.num_substeps,
                cfg.loss_fn,
            )
        completed_steps += 1

        # Save checkpoint and get new sampling client
        sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
            training_client,
            completed_steps,
            dataset_batch_idx,
            cfg.log_path,
            cfg.save_every,
            start_batch,
        )
        metrics.update(checkpoint_metrics)

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=step_idx)

        # TensorBoard logging
        if tb_logger:
            tb_logger.log_training_metrics(metrics, step_idx)
            tb_logger.log_trajectory_histograms(trajectory_groups, step_idx)
            tb_logger.log_per_level_metrics(trajectory_groups, step_idx)
            tb_logger.log_advantage_statistics(advantages, step_idx)

        logger.info(
            f"Batch {completed_steps}/{target_steps}: "
            f"reward={metrics.get('reward/mean', 0):.3f}, "
            f"compile={metrics.get('kernel/compile_rate', 0):.1%}, "
            f"correct={metrics.get('kernel/correct_rate', 0):.1%}"
        )

    if completed_steps < target_steps and dataset_batch_idx >= total_batches:
        logger.warning(
            "Dataset exhausted before reaching target steps: completed %s / %s",
            completed_steps,
            target_steps,
        )

    # Save final checkpoint
    if start_batch < completed_steps:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": completed_steps, "dataset_batch": dataset_batch_idx},
        )

    # Close loggers
    if tb_logger:
        tb_logger.flush()
        tb_logger.close()
    ml_logger.close()
    logger.info("Training completed!")


async def main(cfg: TrainingConfig) -> None:
    """Entry point for training."""
    await run_training_loop(cfg)
