# Open Banking and Heterogeneous Borrower Effects (CausalML)

This application takes the economic mechanism in Zhiguo He's open-banking research and asks an empirical question that an average treatment effect can hide:

> **Which borrowers gain or lose from data portability?**

Open banking can improve screening and competition, but data sharing, strategic lender responses, opt-in behavior, and inference about borrowers who do not share can generate heterogeneous effects.

## ML object

The treatment is open-banking participation / data sharing. The outcome is a borrowing spread. An X-learner estimates

\[
\tau(x)=E[Y(1)-Y(0)\mid X=x]
\]

as a function of:

- borrower credit quality,
- opacity,
- incumbent relationship strength,
- fintech access,
- local competition,
- privacy cost,
- income.

The synthetic design deliberately permits treatment effects to change sign for some borrower states. That makes the distribution of effects economically meaningful.

## Run

```bash
python economics_applications/zhiguo_he_open_banking/open_banking_cate.py
```

## Extension beyond a representative borrower

The theoretical mechanism highlights strategic information effects. This CausalML application adds a **distributional empirical layer**:

- Who benefits most from portability?
- Does opacity amplify the gain?
- Do strong incumbent relationships reduce the gain?
- Are effects different in concentrated credit markets?
- Is voluntary sharing itself a source of selection?

## Real-data identification

Possible designs include staggered open-banking/data-portability reforms or platform rollouts. In a real project, treatment timing and exposure should be justified institutionally; causal ML then characterizes heterogeneity conditional on that design.

A stronger version would combine:

1. DiD/event-study identification of the reform,
2. causal ML for treatment-effect heterogeneity,
3. a market-level competition model for spillovers onto nonparticipants.

This is an **inspired application, not a replication** and does not imply author endorsement.

References:
- https://zhiguohe.net/publications/research/
- https://github.com/uber/causalml
