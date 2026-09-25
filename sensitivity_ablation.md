# Dahlquist Stiffness/Sensitivity Ablation

I want to add a controlled stiffness experiment using the scalar Dahlquist test equation to the existing Neural ODE benchmark codebase.

Please inspect the existing codebase first and reuse the current abstractions for:
- ODE problems/datasets
- architectures
- training methods
- experiment configuration
- TOML result logging
- bash experiment scripts

Do not unnecessarily duplicate existing functionality or change unrelated experiments.

## 1. Add the Dahlquist test problem

Implement the scalar Dahlquist equation

\[
\frac{dy}{dt} = \lambda y,
\qquad
y(0)=y_0,
\]

where `lambda` is an experiment parameter.

The exact solution is

\[
y(t) = y_0 \exp(\lambda t).
\]

We are interested in the stable/stiff regime, so lambda should be negative.

Make `lambda` configurable in the same way as the parameters of the other benchmark ODEs. The experiment script should therefore be able to run a sweep over different values of lambda without modifying source code.

Use the existing conventions in the repository for time interval, number of observations, train/test data, tolerances, etc. Where a Dahlquist-specific choice is required and there is no obvious convention, document the choice clearly rather than silently introducing it.

## 2. Compute parameter sensitivity after training

For a trained Neural ODE

\[
\dot y = f_\theta(y),
\]

define the trajectory parameter sensitivity at time \(t_i\) as

\[
s(t_i)
=
\left\|
\frac{\partial y_\theta(t_i)}
{\partial \theta}
\right\|_2.
\]

Here, \(y_\theta(t_i)\) means the trajectory obtained by solving the learned Neural ODE from the prescribed initial condition using the final trained network parameters.

IMPORTANT:
- Compute this only after the FINAL training epoch/iteration.
- Do NOT compute/save sensitivity at every training epoch.
- This is sensitivity with respect to ALL trainable neural-network parameters theta.
- Flatten/concatenate the parameter derivatives as necessary before computing the Euclidean norm.
- Evaluate sensitivity on the trajectory time points used by the experiment.
- Do not include the initial point \(t=0\) in the aggregation if its state is prescribed independently of theta.

Aggregate the sensitivities across time using RMS:

\[
S =
\sqrt{
    \frac{1}{N}
    \sum_{i=1}^{N}
    \left\|
        \frac{\partial y_\theta(t_i)}
        {\partial\theta}
    \right\|_2^2
}.
\]

This should produce ONE scalar sensitivity value per completed training run.

Please implement this as reusable functionality rather than embedding it directly into the Dahlquist experiment if doing so fits the current code architecture.

## 3. Save sensitivity in the TOML results

The final aggregated sensitivity `S` must be written to the existing TOML results file together with the other metrics for the run.

Use a clear field name, preferably something like

    trajectory_parameter_sensitivity_rms

unless the existing naming conventions suggest something better.

Also make sure the TOML output contains enough metadata to identify:
- lambda
- seed
- architecture
- training method

Do not remove or change existing metrics.

## 4. Experiment grid

Create a bash script for running the complete Dahlquist stiffness ablation.

We want the Cartesian product of:

### Architectures
1. Stiffened Net / StiffNet
2. GELU baseline

Use the existing implementations/configurations of these architectures.

### Training methods
1. Shooting
2. Derivative matching

Use the existing implementations/configurations of these methods.

Thus, for every lambda and seed, run:

1. StiffNet + Shooting
2. StiffNet + Derivative Matching
3. GELU + Shooting
4. GELU + Derivative Matching

### Random seeds

Run every configuration for exactly 10 seeds:

    800
    801
    802
    803
    804
    805
    806
    807
    808
    809

### Lambda sweep

Make the lambda values easy to modify near the top of the bash script.

Use a logarithmically spaced sweep of negative lambda values. If there is no existing choice in the repository, initially use something along the lines of

    -1
    -10
    -100
    -1000
    -10000

but keep this list explicitly defined in the bash script so I can easily change it.

This means that with 5 lambda values the default experiment consists of

    5 lambda values × 4 method/architecture combinations × 10 seeds
    = 200 runs.

## 5. Bash script

Add a bash script with an informative name such as

    run_dahlquist_sensitivity_ablation.sh

Follow the style of the existing experiment scripts.

The script should loop over lambda values and seeds and invoke the existing experiment/training entry point with the appropriate:
- Dahlquist problem
- lambda
- architecture
- training method
- seed

Make the experiment grid obvious and easy to edit.

Print enough information before each run to make logs understandable, e.g.

    Dahlquist | lambda=-100 | architecture=stiffnet | method=shooting | seed=803

Do not introduce a separate training pipeline if the existing pipeline can support this experiment.

## 6. Expected final analysis

The experiment is intended to eventually produce a plot with

\[
x = |\lambda|
\]

on a logarithmic x-axis and

\[
y =
\sqrt{
\frac{1}{N}
\sum_i
\left\|
\frac{\partial y_{\theta_K}(t_i)}
{\partial\theta}
\right\|_2^2
}
\]

on the y-axis.

Here \(\theta_K\) denotes the FINAL trained parameters.

There should eventually be four curves:

- StiffNet + Shooting
- StiffNet + Derivative Matching
- GELU + Shooting
- GELU + Derivative Matching

with results aggregated across seeds 800--809.

You do NOT need to implement the plotting/aggregation across seeds unless there is already an obvious analysis framework in the repository. The important thing for this task is that every individual TOML result contains the final scalar sensitivity so that this plot can be generated later.

## 7. Verification

Before considering the task complete:

1. Verify the Dahlquist data against the analytical solution
   \(y(t)=y_0e^{\lambda t}\).

2. Run at least one small/cheap configuration end-to-end.

3. Verify that the final TOML contains:
   - lambda
   - seed
   - architecture
   - training method
   - `trajectory_parameter_sensitivity_rms`

4. Check that the sensitivity is finite for the test run.

5. Confirm that sensitivity is computed using the FINAL network parameters
   and is not accidentally accumulated during training.

6. Confirm that the bash script generates exactly the intended Cartesian
   product of lambda × architecture × training method × seed.

At the end, summarize:
- files added,
- files modified,
- exact sensitivity implementation,
- any assumptions you had to make,
- and the command I should use to launch the full ablation.