"""RLPD v2 with Dataset v2 priors and resumable dual replay."""
from __future__ import annotations

import copy
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .dataset_replay import DatasetReplay
from .hybrid_replay import RLPDBatchSampler
from .replay import ReplayBuffer as _ReplayBuffer


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
RLPD_STATE_SCHEMA_VERSION = "aiogym.rlpd_state.v2"


def mlp(sizes, layernorm=False):
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            if layernorm:
                layers.append(nn.LayerNorm(sizes[index + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        hidden = self.net(obs)
        return self.mu(hidden), self.log_std(hidden).clamp(
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

    def sample(self, obs):
        mu, log_std = self(obs)
        distribution = torch.distributions.Normal(mu, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_probability = distribution.log_prob(raw).sum(-1)
        log_probability -= torch.log(
            1 - action.pow(2) + 1e-6
        ).sum(-1)
        return action, log_probability

    def act(self, obs, deterministic=False):
        mu, log_std = self(obs)
        if deterministic:
            return torch.tanh(mu)
        return torch.tanh(mu + log_std.exp() * torch.randn_like(log_std))


class Critic(nn.Module):
    """Layer-normalized Q function used by the RLPD critic ensemble."""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp(
            [obs_dim + act_dim, hidden, hidden, 1],
            layernorm=True,
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


class RLPD:
    """Offline-to-online SAC with canonical 50/50 replay sampling."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden=256,
        n_critics=5,
        subset=2,
        utd=5,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch=256,
        device="cpu",
        entropy_scale=1.0,
        init_alpha=0.1,
        online_capacity=1_000_000,
        offline_fraction=0.5,
        canonical=True,
        seed=0,
    ):
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch = int(batch)
        self.n_critics = int(n_critics)
        self.subset = int(subset)
        self.utd = int(utd)
        self.offline_fraction = float(offline_fraction)
        self.canonical = bool(canonical)
        self.seed = int(seed)
        if not 1 <= self.subset <= self.n_critics:
            raise ValueError("critic subset must be in [1, n_critics]")
        if self.utd <= 0 or self.batch <= 0:
            raise ValueError("utd and batch must be positive")
        if self.canonical and self.batch % 2:
            raise ValueError("canonical RLPD requires an even batch size")

        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critics = nn.ModuleList(
            [
                Critic(obs_dim, act_dim, hidden)
                for _ in range(self.n_critics)
            ]
        ).to(self.device)
        self.targets = nn.ModuleList(
            [
                Critic(obs_dim, act_dim, hidden)
                for _ in range(self.n_critics)
            ]
        ).to(self.device)
        self.targets.load_state_dict(self.critics.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.c_opt = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.log_alpha = torch.tensor(
            np.log(init_alpha),
            device=self.device,
            requires_grad=True,
        )
        self.al_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(entropy_scale) * float(act_dim)

        self.online = _ReplayBuffer(online_capacity, seed=self.seed + 1)
        self.offline: DatasetReplay | None = None
        self.batch_sampler: RLPDBatchSampler | None = None
        self.environment_transitions = 0
        self.gradient_updates = 0
        self.actor_updates = 0
        self.offline_samples = 0
        self.online_samples = 0

    @staticmethod
    def to_env(normalized_action):
        return (np.asarray(normalized_action) + 1.0) * 0.5

    @staticmethod
    def to_sac(physical_action):
        return np.asarray(physical_action) * 2.0 - 1.0

    def load_dataset(
        self,
        path,
        *,
        stratify=True,
        verify_checksums=False,
    ) -> DatasetReplay:
        replay = DatasetReplay(
            path,
            seed=self.seed + 2,
            stratify=stratify,
            verify_checksums=verify_checksums,
        )
        if replay.observation_dim != self.obs_dim:
            raise ValueError("offline dataset observation dimension mismatch")
        if replay.action_dim != self.act_dim:
            raise ValueError("offline dataset action dimension mismatch")
        self.offline = replay
        self.batch_sampler = RLPDBatchSampler(
            replay,
            self.online,
            offline_fraction=self.offline_fraction,
            canonical=self.canonical,
        )
        return replay

    def bc_warmstart(self, steps=4000, batch=256):
        if self.offline is None or len(self.offline) == 0:
            return {"steps": 0, "final_mse": None}
        final_loss = None
        for _ in range(int(steps)):
            sampled = self._offline_numpy(int(batch))
            observation = torch.as_tensor(
                sampled["observation"],
                device=self.device,
            )
            target = torch.as_tensor(
                sampled["action"],
                device=self.device,
            )
            mu, _ = self.actor(observation)
            loss = functional.mse_loss(torch.tanh(mu), target)
            self.a_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.a_opt.step()
            final_loss = float(loss.detach())
        return {"steps": int(steps), "final_mse": final_loss}

    def push(
        self,
        observation,
        normalized_action,
        reward,
        next_observation,
        terminated,
        truncated=False,
        bootstrap_mask=None,
    ):
        self.online.add(
            observation=np.asarray(observation, np.float32),
            action=np.asarray(normalized_action, np.float32),
            reward=reward,
            next_observation=np.asarray(next_observation, np.float32),
            terminated=bool(terminated),
            truncated=bool(truncated),
            bootstrap_mask=bootstrap_mask,
        )
        self.environment_transitions += 1

    def push_physical(self, observation, action, reward, next_observation, done):
        self.push(
            observation,
            self.to_sac(action),
            reward,
            next_observation,
            done,
        )

    @torch.no_grad()
    def policy_action_batch(self, observations, deterministic=False):
        value = torch.as_tensor(
            np.asarray(observations, np.float32),
            device=self.device,
        )
        return (
            self.actor.act(value, deterministic)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def policy_action(self, observation, deterministic=False):
        return self.policy_action_batch(
            np.asarray(observation)[None, :],
            deterministic=deterministic,
        )[0]

    def act(self, obs, deterministic=False):
        """Controller-facing physical action for unwrapped evaluation envs."""

        normalized = self.policy_action(obs, deterministic=deterministic)
        return np.clip(self.to_env(normalized), 0.0, 1.0).astype(np.float32)

    def sample_batch(self) -> dict[str, np.ndarray]:
        batch = self._sample_numpy(self.batch)
        source = batch["source"]
        self.offline_samples += int(np.sum(source == "offline"))
        self.online_samples += int(np.sum(source == "online"))
        return batch

    def update(self, actor=True, *, critic_updates=None):
        updates = self.utd if critic_updates is None else int(critic_updates)
        if updates <= 0:
            raise ValueError("critic_updates must be positive")
        alpha = self.log_alpha.exp().detach()
        q_loss = None
        for _ in range(updates):
            batch = self.sample_batch()
            observation, action, reward, next_observation, mask = (
                self._torch_batch(batch)
            )
            with torch.no_grad():
                next_action, next_log_probability = self.actor.sample(
                    next_observation
                )
                selected = self._rng.choice(
                    self.n_critics,
                    self.subset,
                    replace=False,
                )
                next_q = torch.stack(
                    [
                        self.targets[int(index)](
                            next_observation,
                            next_action,
                        )
                        for index in selected
                    ],
                    0,
                ).min(0).values
                target = reward + self.gamma * mask * (
                    next_q - alpha * next_log_probability
                )
            q_loss = sum(
                functional.mse_loss(critic(observation, action), target)
                for critic in self.critics
            )
            self.c_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            self.c_opt.step()
            self.gradient_updates += 1

        actor_loss = 0.0
        if actor:
            observation = self._torch_batch(self.sample_batch())[0]
            action, log_probability = self.actor.sample(observation)
            selected = self._rng.choice(
                self.n_critics,
                self.subset,
                replace=False,
            )
            policy_q = torch.stack(
                [
                    self.critics[int(index)](observation, action)
                    for index in selected
                ],
                0,
            ).min(0).values
            actor_tensor = (
                alpha * log_probability - policy_q
            ).mean()
            self.a_opt.zero_grad(set_to_none=True)
            actor_tensor.backward()
            self.a_opt.step()

            alpha_loss = -(
                self.log_alpha.exp()
                * (log_probability.detach() + self.target_entropy)
            ).mean()
            self.al_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.al_opt.step()
            actor_loss = float(actor_tensor.detach())
            self.actor_updates += 1

        with torch.no_grad():
            for critic, target in zip(
                self.critics.parameters(),
                self.targets.parameters(),
            ):
                target.mul_(1 - self.tau).add_(self.tau * critic)
        return {
            "q_loss": float(q_loss.detach()),
            "a_loss": actor_loss,
            "alpha": float(alpha),
        }

    def accounting(self) -> dict[str, int | float | str | None]:
        total_samples = self.offline_samples + self.online_samples
        data = {
            "offline_transitions": (
                len(self.offline) if self.offline is not None else 0
            ),
            "online_transitions": self.environment_transitions,
            "online_replay_transitions": len(self.online),
            "offline_samples": self.offline_samples,
            "online_samples": self.online_samples,
            "sampled_offline_fraction": (
                self.offline_samples / total_samples
                if total_samples
                else 0.0
            ),
            "gradient_updates": self.gradient_updates,
            "actor_updates": self.actor_updates,
            "dataset_id": None,
            "dataset_hash": None,
        }
        if isinstance(self.offline, DatasetReplay):
            data.update(
                {
                    "dataset_id": self.offline.dataset_id,
                    "dataset_hash": self.offline.dataset_hash,
                }
            )
        return data

    def save_onnx(self, path):
        """Export deterministic observations to normalized ``[-1, 1]`` actions."""

        class DeterministicPolicy(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                mu, _ = self.actor(obs)
                return torch.tanh(mu)

        model = DeterministicPolicy(self.actor).eval()
        export_options = {
            "input_names": ["obs"],
            "output_names": ["action"],
            "dynamic_axes": {
                "obs": {0: "batch"},
                "action": {0: "batch"},
            },
            "opset_version": 17,
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_options["dynamo"] = False
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.zeros(1, self.obs_dim, device=self.device),
                path,
                **export_options,
            )

    def state_dict(self):
        offline_state = None
        if isinstance(self.offline, DatasetReplay):
            offline_state = {
                "kind": "dataset",
                "state": self.offline.state_dict(),
            }
        return {
            "schema_version": RLPD_STATE_SCHEMA_VERSION,
            "dimensions": {
                "observation": self.obs_dim,
                "action": self.act_dim,
                "critics": self.n_critics,
            },
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "targets": self.targets.state_dict(),
            "actor_optimizer": self.a_opt.state_dict(),
            "critic_optimizer": self.c_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.al_opt.state_dict(),
            "online_replay": self.online.state_dict(),
            "offline_replay": offline_state,
            "numpy_rng": copy.deepcopy(self._rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "accounting": self.accounting(),
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != RLPD_STATE_SCHEMA_VERSION:
            self._load_legacy_state_dict(state)
            return
        dimensions = state["dimensions"]
        if (
            int(dimensions["observation"]) != self.obs_dim
            or int(dimensions["action"]) != self.act_dim
            or int(dimensions["critics"]) != self.n_critics
        ):
            raise ValueError("RLPD checkpoint dimensions do not match")
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.targets.load_state_dict(state["targets"])
        self.a_opt.load_state_dict(state["actor_optimizer"])
        self.c_opt.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.al_opt.load_state_dict(state["alpha_optimizer"])
        self.online = _ReplayBuffer.from_state_dict(
            state["online_replay"]
        )
        offline = state["offline_replay"]
        if offline is None:
            self.offline = None
            self.batch_sampler = None
        elif offline["kind"] == "dataset":
            self.offline = DatasetReplay.from_state_dict(offline["state"])
            self.batch_sampler = RLPDBatchSampler(
                self.offline,
                self.online,
                offline_fraction=self.offline_fraction,
                canonical=self.canonical,
            )
        else:
            raise ValueError("unknown RLPD offline replay checkpoint kind")
        self._rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["torch_cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
        accounting = state["accounting"]
        self.environment_transitions = int(accounting["online_transitions"])
        self.gradient_updates = int(accounting["gradient_updates"])
        self.actor_updates = int(accounting["actor_updates"])
        self.offline_samples = int(accounting["offline_samples"])
        self.online_samples = int(accounting["online_samples"])

    def _sample_numpy(self, batch_size):
        if self.offline is None:
            if len(self.online) == 0:
                raise ValueError("RLPD replay is empty")
            batch = self.online.sample(batch_size)
            batch["source"] = np.full(
                batch_size,
                "online",
                dtype="<U16",
            )
            return batch
        if self.batch_sampler is not None:
            return self.batch_sampler.sample(batch_size)
        raise RuntimeError("offline replay is missing its batch sampler")

    def _offline_numpy(self, batch_size):
        if not isinstance(self.offline, DatasetReplay):
            raise RuntimeError("RLPD warm start requires Dataset v2")
        batch = self.offline.sample(batch_size)
        return {
            "observation": batch["observation"],
            "action": batch["action_policy_normalized"],
        }

    def _torch_batch(self, batch):
        tensor = lambda name: torch.as_tensor(
            batch[name],
            device=self.device,
            dtype=torch.float32,
        )
        return (
            tensor("observation"),
            tensor("action"),
            tensor("reward"),
            tensor("next_observation"),
            tensor("bootstrap_mask"),
        )

    def _load_legacy_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.targets.load_state_dict(state["targets"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])


__all__ = [
    "RLPD_STATE_SCHEMA_VERSION",
    "Actor",
    "Critic",
    "RLPD",
]
