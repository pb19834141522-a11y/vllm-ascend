"""Device-only minimax CPU/transfer scheduling with protected cold eviction."""
import torch

from vllm_ascend.expert_offload.frequency_transfer_policy import FREQUENCY_DECAY


class LoadBalanceTransferPolicy:
    """Initial cost model; coefficients are estimates, not online measurements.

    Cost per CPU expert = base + extra-route cost * (route count - 1).
    Enumerate fixed prefix lengths on device; no scalar D2H or host branching.
    """

    def __init__(self, experts, slots, device, config):
        self.experts = torch.arange(experts, device=device, dtype=torch.int64)
        self.slots = torch.arange(slots, device=device, dtype=torch.int64)
        self.scores = torch.zeros(experts, device=device, dtype=torch.float32)
        self.counts = torch.zeros_like(self.scores)
        self.limit = min(config.balance_max_transfers, slots, experts)
        if self.limit < 1:
            raise ValueError("Load balancing requires at least one NPU expert slot")
        self.prefixes = torch.arange(self.limit + 1, device=device)
        self.entries = torch.arange(self.limit, device=device)
        self.cpu_expert_us = config.balance_cpu_expert_us
        self.cpu_extra_route_us = config.balance_cpu_extra_route_us
        self.copy_us = config.balance_copy_us
        self.npu_route_us = config.balance_npu_route_us
        self.min_gain_us = config.balance_min_gain_us

    def select(self, ids, mapping):
        flat = ids.reshape(-1).long()
        valid = (flat >= 0) & (flat < mapping.numel())
        safe = flat.clamp(0, mapping.numel() - 1)
        self.counts.zero_().scatter_add_(0, safe, valid.float())
        self.scores.mul_(FREQUENCY_DECAY).add_(self.counts)
        routed = self.counts > 0
        missing = routed & (mapping < 0)
        owners = mapping[:, None] == self.slots[None, :]
        protected = (owners & routed[:, None]).any(dim=0)
        available = ~protected
        cost = self.cpu_expert_us + (self.counts - 1).clamp(min=0) * self.cpu_extra_route_us
        benefit = torch.where(missing, cost, 0)
        # Historical heat breaks equal-cost ties; it cannot override current demand.
        tie = self.scores / (self.scores.amax() + 1) * 0.001
        rank = torch.where(missing, benefit + tie, float("-inf"))
        candidates = torch.topk(rank, self.limit, sorted=True).indices
        slot_scores = torch.where(owners, self.scores[:, None], 0).amax(dim=0)
        victims = torch.topk(torch.where(available, -slot_scores, float("-inf")),
                             self.limit, sorted=True).indices
        saved = torch.cat((benefit.new_zeros(1), benefit[candidates].cumsum(0)))
        moved_routes = torch.cat((benefit.new_zeros(1), self.counts[candidates].cumsum(0)))
        cpu_finish = (benefit.sum() - saved).clamp(min=0)
        resident_routes = torch.where(routed & ~missing, self.counts, 0).sum()
        npu_finish = self.prefixes * self.copy_us + (resident_routes + moved_routes) * self.npu_route_us
        finish = torch.maximum(cpu_finish, npu_finish)
        feasible = (self.prefixes <= missing.sum()) & (self.prefixes <= available.sum())
        finish = torch.where(feasible, finish, float("inf"))
        best = finish.argmin()
        gain = finish[0] - finish.gather(0, best.reshape(1)).squeeze(0)
        chosen = torch.where(gain > self.min_gain_us, best, torch.zeros_like(best))
        active = self.entries < chosen
        evicted = ((mapping[:, None] == victims[None, :]) & active[None, :]).any(dim=1)
        updated = torch.where(evicted, -1, mapping)
        admitted = (self.experts[:, None] == candidates[None, :]) & active[None, :]
        new_slots = torch.where(admitted, victims[None, :], -1).amax(dim=1)
        mapping.copy_(torch.where(new_slots >= 0, new_slots.to(mapping.dtype), updated))
        return candidates, victims, active
