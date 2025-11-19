import numpy as np
from collections import deque
import random

def safe_positive(x, floor=1e-6):
    return max(x, floor)

class ArmStats:
    def __init__(self, prior_a=1.0, prior_b=1.0, cusum_h=6.0, cusum_k=0.02, buffer_size=40, forced_explore_after_reset=6):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.alpha = prior_a
        self.beta = prior_b

        # CUSUM
        self.G_pos = 0.0
        self.G_neg = 0.0
        self.h = cusum_h
        self.k = cusum_k
        self.buffer = deque(maxlen=buffer_size)

        # forced exploration after reset
        self.forced_explore = 0
        self.forced_explore_after_reset = forced_explore_after_reset

    def sample_theta(self):
        # ensure alpha/beta > 0
        a = safe_positive(self.alpha)
        b = safe_positive(self.beta)
        return np.random.beta(a, b)

    def posterior_mean(self):
        a = safe_positive(self.alpha)
        b = safe_positive(self.beta)
        return a / (a + b)

    def update_posterior(self, successes, trials):
        # successes/trials must be local to the batch (not cumulative)
        # clamp values
        s = max(0, int(successes))
        t = max(1, int(trials))
        # update
        self.alpha += s
        self.beta += (t - s)
        # update buffer with proportion
        p_obs = s / t
        self.buffer.append(p_obs)

    def check_and_update_cusum(self, latest_proportion):
        if len(self.buffer) < 2:
            return False
        ref_mean = np.mean(list(self.buffer)[:-1])
        s_pos = latest_proportion - ref_mean - self.k
        self.G_pos = max(0.0, self.G_pos + s_pos)
        s_neg = ref_mean - latest_proportion - self.k
        self.G_neg = max(0.0, self.G_neg + s_neg)
        if self.G_pos > self.h or self.G_neg > self.h:
            self.G_pos = 0.0
            self.G_neg = 0.0
            return True
        return False

    def reset(self, partial=True):
        if partial:
            # soft shrink to keep tiny memory
            self.alpha = max(self.prior_a, self.alpha * 0.2)
            self.beta = max(self.prior_b, self.beta * 0.2)
        else:
            self.alpha = self.prior_a
            self.beta = self.prior_b
        self.buffer.clear()
        self.G_pos = 0.0
        self.G_neg = 0.0
        self.forced_explore = self.forced_explore_after_reset


class ChangeAwareStickyTS:
    """
    Change-aware Thompson Sampling with stickiness and a switch margin.
    - stickiness: remain on current_arm for min_stick pulls before switching
    - switch_margin: require new sample > current_sample + margin to switch
    - epsilon_probe: small prob to explore randomly other arms
    """

    def __init__(self, n_arms=3, prior_a=1, prior_b=1,
                 cusum_h=6.0, cusum_k=0.02, buffer_size=40,
                 forced_explore_after_reset=6, partial_reset=True,
                 epsilon_probe=0.02, min_stick=3, switch_margin=0.05):
        self.n_arms = n_arms
        self.arms = [
            ArmStats(prior_a, prior_b, cusum_h, cusum_k, buffer_size, forced_explore_after_reset)
            for _ in range(n_arms)
        ]
        self.partial_reset = partial_reset

        self.epsilon_probe = epsilon_probe
        self.min_stick = min_stick
        self.switch_margin = switch_margin

        self.current_arm = None
        self.current_arm_streak = 0

    def select_arm(self):
        for i, arm in enumerate(self.arms):
            if arm.forced_explore > 0:
                return i

        if random.random() < self.epsilon_probe:
            choices = list(range(self.n_arms))
            if self.current_arm is not None and len(choices) > 1:
                choices.remove(self.current_arm)
            return random.choice(choices)

        samples = [arm.sample_theta() for arm in self.arms]
        best_idx = int(np.argmax(samples))
        best_sample = samples[best_idx]

        if self.current_arm is None:
            self.current_arm = best_idx
            self.current_arm_streak = 0
            return best_idx

        current_sample = samples[self.current_arm]

        if self.current_arm_streak < self.min_stick:
            if best_idx == self.current_arm:
                # same arm, increment streak
                return self.current_arm
            else:
                # switch only if best_sample > current_sample + switch_margin
                if best_sample > current_sample + self.switch_margin:
                    return best_idx
                else:
                    return self.current_arm

        if best_sample > current_sample + self.switch_margin:
            return best_idx
        else:
            return self.current_arm

    def observe(self, arm_index, successes, trials):
        arm = self.arms[arm_index]

        arm.update_posterior(successes, trials)

        # Update CUSUM using observed proportion if trials>0
        if trials > 0:
            p_obs = successes / trials
            change = arm.check_and_update_cusum(p_obs)
            if change:
                arm.reset(partial=self.partial_reset)

        for a in self.arms:
            if a.forced_explore > 0:
                a.forced_explore -= 1

        if self.current_arm is None:
            self.current_arm = arm_index
            self.current_arm_streak = 1
        else:
            if arm_index == self.current_arm:
                self.current_arm_streak += 1
            else:
                self.current_arm = arm_index
                self.current_arm_streak = 1
