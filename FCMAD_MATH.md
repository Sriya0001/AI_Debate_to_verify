# FC-MAD Math (Final, Minimal)

Single source of truth for the `fcmad` mode. Do not change a formula in code without changing it here first.

**Notation:** $N$ debaters, rounds $t=0..M$, classes $\{S,R,N\}$ (Supported, Refuted, Not-Enough-Info).
Each debater $i$ holds a belief $p_i=[p_S,p_R,p_N]$, non-negative and summing to 1.

---

## 1. Disagreement (total variation)

$$d(p,q)=\tfrac12\sum_k |p_k-q_k| \quad\in[0,1]$$

## 2. Credibility of each debater

Closeness to the average of the *others*, times the Judge's argument-quality score $q_i\in[0,1]$ (use $q_i=1$ if the Judge is switched off):

$$s_i=1-d\big(p_i,\ \bar p_{-i}\big),\quad \bar p_{-i}=\tfrac1{N-1}\sum_{j\ne i}p_j$$

$$c_i=\frac{s_i\,q_i+0.01}{\sum_k (s_k\,q_k+0.01)}$$

The $q_i$ term stops a confident-but-wrong majority from automatically earning the most weight. The 0.01 keeps every $c_i$ above zero. At the start use $c_i=1/N$.

## 3. DeGroot update (one step per round)

$$p_i^{new}=\alpha\,p_i+(1-\alpha)\sum_{j\ne i}\frac{c_j}{\sum_{k\ne i}c_k}\,p_j$$

Each round: (a) the Summarizer writes a short debate summary; (b) each debater re-reasons from the claim, evidence and that summary and returns a revised belief $\hat p_i$; (c) apply the update above to the revised beliefs.

**Important:** debaters see the *summary of arguments*, not their peers' numeric probabilities or credibilities. The numeric mixing is done once, by this formula. This avoids counting peer influence twice.

## 4. Stop rule

$$E=\max_{i,j}\ d(p_i,p_j)$$

Stop when $E\le\varepsilon$ (consensus) or $t=M$ (record "no consensus").

LLMs give coarse numbers (e.g. 0.9/0.05/0.05), so $\varepsilon=0.10$ is the default. Tune it on the dev split.

## 5. Final answer and confidence

$$p^*=\sum_i c_i\,p_i \qquad \hat y=\arg\max_k p^*_k \qquad \text{raw confidence}=\max_k p^*_k$$

The Output Agent only writes the explanation. It never changes $\hat y$ or $p^*$.

## 6. Calibration (one parameter, dev split only)

$$p^{cal}_k=\frac{(p^*_k)^{1/T}}{\sum_l (p^*_l)^{1/T}}\qquad T=\arg\min_T\ -\sum_n \ln p^{cal}_{n}[y_n]$$

Fit $T$ on a held-out dev split, freeze it, then apply it to the test split. Calibrated confidence $=\max_k p^{cal}_k$.

## 7. Abstain

If calibrated confidence $<\tau$, output "uncertain". Choose $\tau$ on the dev split as the lowest value whose accuracy on kept claims reaches your target (e.g. 90%). This is separate from the NEI label: NEI says the evidence is insufficient; abstain says the system is unsure of its own verdict.

## 8. Metrics

- Accuracy and macro-F1
- **ECE**: $\sum_b \frac{|B_b|}{n}\,|\text{acc}(B_b)-\text{conf}(B_b)|$ over 10 equal-width bins
- **Brier**: $\frac1n\sum_n\sum_k (p_{n,k}-\mathbf 1[y_n=k])^2$
- Mean ± standard error over several seeds
- API or parse failures are excluded and counted, never turned into a label

---

## Defaults

| Parameter | Value |
|---|---|
| $N$ | 3 |
| $\alpha$ | 0.7 |
| $\varepsilon$ | 0.10 |
| $M$ | 4 |
| Samples per debater (averaged) | 3 |
| Debater temperature | 0.7 (Judge and Output at 0) |

## Ablations to run (only these)

1. Without Judge/credibility (equal weights)
2. Without DeGroot ($\alpha=1$)
3. Without Summarizer

## Required tests

1. All beliefs, $p^*$ and $p^{cal}$ are valid probabilities.
2. Credibilities are positive and sum to 1.
3. Identical beliefs give $E=0$ and $p^*=p_i$.
4. Shuffling debater order does not change the result.
5. A failed LLM call never produces a belief or a label.
