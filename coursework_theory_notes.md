# Coursework Theory Notes

This note is aligned with the brief in [coursework_task.md](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/coursework_task.md) and the current implementation in [Coursework3/RNO_1D_Skeleton.py](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/Coursework3/RNO_1D_Skeleton.py).

## 1. What Data You Have, And What Part (a) Is Asking

### What is in the dataset

The supplied data file is:

- `Coursework3/viscodata_3mat.mat`

It contains exactly two arrays:

- `epsi_tol`: macroscopic strain histories, shape `(400, 1001)`
- `sigma_tol`: macroscopic stress histories, shape `(400, 1001)`

So:

- there are `400` loading samples
- each sample is one full time history
- each history has `1001` time points
- the data are sequence-to-sequence, not single-step input-output pairs

In other words, sample `k` is:

```text
{ \bar{\epsilon}_k(t_n) }_{n=0}^{1000}  ->  { \bar{\sigma}_k(t_n) }_{n=0}^{1000}
```

where the overbar denotes a macroscopic quantity.

### What is not in the dataset

You are not given:

- the microscopic displacement field `u(x,t)`
- the microscopic strain field `\epsilon(x,t)`
- the microscopic stress field `\sigma(x,t)`
- the phasewise material parameters `E_i` and viscosity parameters `\eta_i` directly
- the phase boundaries or phase fractions explicitly in the dataset

So this coursework is not an inverse problem for identifying microscopic `E(x)` and `v(x)` from data. It is a forward surrogate-learning problem for the homogenized constitutive law.

Also, the symbol `v(x)` in the brief is being used as a viscosity coefficient, not Poisson's ratio. In 1D, Poisson's ratio would not appear anyway.

### What part (a) is really asking

Part (a) asks for the input and output of the macroscopic constitutive model implied by the unit-cell problem.

The clean answer is:

- input: macroscopic strain history `\bar{\epsilon}_{[0,t]}` or, equivalently, the current macroscopic strain and whatever internal variables are needed to encode past history
- output: current macroscopic stress `\bar{\sigma}(t)`

So the macroscopic constitutive law is an operator of the form

```math
\bar{\sigma}(t) = \mathcal{G}\left(\bar{\epsilon}_{[0,t]}\right)
```

where `\bar{\epsilon}_{[0,t]}` means the strain history from time `0` up to time `t`.

In discrete form, matching the dataset:

```math
\{\bar{\epsilon}_n\}_{n=0}^{1000} \mapsto \{\bar{\sigma}_n\}_{n=0}^{1000}.
```

### What the code uses as the practical input and output

The recurrent model does not feed the whole history in one shot. It processes the history step by step.

At time step `n`, the implemented per-step features are:

```math
x_n = \left[\bar{\epsilon}_n,\; \bar{\epsilon}_{n-1},\; \dot{\bar{\epsilon}}_n\right],
\qquad
\dot{\bar{\epsilon}}_n \approx \frac{\bar{\epsilon}_n - \bar{\epsilon}_{n-1}}{\Delta t}.
```

The model output at that step is:

```math
\hat{\bar{\sigma}}_n.
```

So for part (a), a concise report answer would be:

> The macroscopic constitutive model takes the applied macroscopic strain history as input and returns the macroscopic stress history as output. In discrete recurrent form, the model uses the current strain, previous strain, and strain rate at each time step, while the hidden state stores the unresolved history dependence.

## 2. Theory Of The Micro-To-Macro Model

### Microscopic problem

The unit-cell equations in the brief are:

```math
\epsilon(x,t) = \frac{\partial u(x,t)}{\partial x},
```

```math
\frac{d\sigma(x,t)}{dx} = 0,
```

```math
\sigma(x,t) = E(x)\epsilon(x,t) + v(x)\,(\text{viscous term}).
```

The heading says "visco-plasticity", but the written constitutive law is a linear elastic-viscous law, so the correct theoretical lens is linear viscoelasticity.

For a standard 1D Kelvin-Voigt phase, this is interpreted as

```math
\sigma_i(t) = E_i \epsilon_i(t) + \eta_i \dot{\epsilon}_i(t),
```

for each phase `i`, where `\eta_i` is the viscosity coefficient.

### Why the stress is macroscopic but the strain is phase-dependent

Because equilibrium gives

```math
\frac{d\sigma}{dx} = 0,
```

the stress is uniform through the 1D bar:

```math
\sigma(x,t) = \bar{\sigma}(t).
```

But the strain generally differs from phase to phase because `E(x)` and `\eta(x)` are piecewise constant and different in each phase.

If the three phases are arranged along the loading direction, then this is a series composite. In that case:

- stress is common to all phases
- total strain is the volume-fraction-weighted average of the phase strains

Let the phase fractions be `f_1`, `f_2`, `f_3`, with `f_1 + f_2 + f_3 = 1`. Then

```math
\bar{\epsilon}(t) = f_1 \epsilon_1(t) + f_2 \epsilon_2(t) + f_3 \epsilon_3(t),
```

and

```math
\bar{\sigma}(t) = \sigma_1(t) = \sigma_2(t) = \sigma_3(t).
```

### Why the macroscopic law has memory

Each phase obeys

```math
\bar{\sigma}(t) = E_i \epsilon_i(t) + \eta_i \dot{\epsilon}_i(t),
\qquad i = 1,2,3.
```

Rearranging,

```math
\dot{\epsilon}_i(t) = \frac{1}{\eta_i}\bar{\sigma}(t) - \frac{E_i}{\eta_i}\epsilon_i(t).
```

So the phase strains evolve in time. Since the macroscopic strain is the average of these evolving phase strains, the macroscopic stress cannot depend only on the instantaneous macroscopic strain. It depends on how the internal phase strains have evolved, so the homogenized constitutive law is history-dependent.

That is exactly why a recurrent model is appropriate: it needs memory.

### Equivalent homogenized operator

Taking Laplace transforms with zero initial conditions:

```math
\epsilon_i(s) = \frac{\bar{\sigma}(s)}{E_i + \eta_i s}.
```

Therefore,

```math
\bar{\epsilon}(s)
=
\bar{\sigma}(s)\sum_{i=1}^3 \frac{f_i}{E_i + \eta_i s}.
```

So the macroscopic creep compliance is

```math
J(s) = \sum_{i=1}^3 \frac{f_i}{E_i + \eta_i s},
\qquad
\bar{\epsilon}(s) = J(s)\bar{\sigma}(s).
```

This is a compact way of saying:

- the micro problem can be homogenized
- the resulting macro constitutive law is linear but history-dependent
- the hidden state in the RNO is playing the role of the internal memory needed to represent that history dependence in Markov form

## 3. Theory Of The Code And Why It Counts As An RNO

### Overall idea

The current code implements a recurrent constitutive operator, not a spatial neural operator.

The object being learned is still an operator:

```math
\mathcal{G} : \bar{\epsilon}_{[0,t]} \mapsto \bar{\sigma}(t),
```

but the operator is represented through a recurrent time-marching model.

### What happens at each time step

The model class is [RNO](/Users/jackhenry/Library/CloudStorage/OneDrive-UniversityofCambridge/Labs%20and%20Courswork/IIB%20Coursework/4C11/4C11_assignment_3/Coursework3/RNO_1D_Skeleton.py#L306).

At step `n`, the code forms the feature vector

```math
\xi_n
=
\left[
\bar{\epsilon}_n,\;
\bar{\epsilon}_{n-1},\;
\frac{\bar{\epsilon}_n - \bar{\epsilon}_{n-1}}{\Delta t}
\right].
```

This is passed through a feature MLP:

```math
\tilde{\xi}_n = \phi(\xi_n).
```

Then the recurrent core updates the hidden state:

For vanilla RNN:

```math
h_n = \mathrm{RNNCell}(\tilde{\xi}_n, h_{n-1}).
```

For GRU:

```math
h_n = \mathrm{GRUCell}(\tilde{\xi}_n, h_{n-1}).
```

For LSTM:

```math
(h_n, c_n) = \mathrm{LSTMCell}(\tilde{\xi}_n, (h_{n-1}, c_{n-1})).
```

Finally, the stress is read out from the current kinematic features together with the updated hidden state:

```math
\hat{\bar{\sigma}}_n = \psi\left(\bar{\epsilon}_n,\bar{\epsilon}_{n-1},\dot{\bar{\epsilon}}_n,h_n\right).
```

That is the key constitutive idea:

- explicit current loading information enters directly
- past history is compressed into the hidden state
- the output stress is a function of both

### Important skeleton detail

The variable name `output` inside `RNO.forward()` is inherited from the starter skeleton, but in the current implementation it is actually the previous strain, not the previous stress. So the model is driven by:

- current strain
- previous strain
- strain rate
- hidden state

not by autoregressive stress feedback.

### Why this is a recurrent neural operator

It is reasonable to call this an RNO because:

- the target object is a constitutive operator mapping an input history to an output history
- the recurrence gives a learned Markovian representation of history dependence
- the hidden state plays the role of learned internal variables

So the recurrence is not incidental. It is the mechanism that turns the path-dependent operator into a one-step update law.

### Role of the three recurrent cores

The three architectures are identical except for the hidden-state update:

- `RNN`: simplest recurrence, most vulnerable to vanishing gradients on long sequences
- `GRU`: gated recurrence, usually easier to train and better at retaining useful history
- `LSTM`: strongest explicit memory mechanism through a separate cell state

That makes the comparison defensible in the report: the outer constitutive operator is the same, and only the memory-update mechanism changes.

### Initial step handling

The data start at zero strain, but the first stress value is generally nonzero because it reflects the initial strain rate. The code therefore uses a small fitted model for the initial stress:

```math
\hat{\sigma}_0 \approx a\,\dot{\bar{\epsilon}}_0 + b,
```

estimated from the training set. During training, the implementation can optionally use the true first stress value, but during pure inference it uses this fitted rate-to-initial-stress relation.

### How the model incorporates physics

The important point is that this is not a fully physics-informed neural network in the strict sense. The code does not enforce the microscopic field equations by adding residuals such as

```math
\frac{d\sigma}{dx} = 0
```

or

```math
\sigma = E(x)\epsilon + \eta(x)\dot{\epsilon}
```

directly into the loss. Instead, the physics enters mainly through the model structure and the choice of variables.

There are four main ways the physics is incorporated.

#### 1. The inputs are physically meaningful constitutive variables

The model is not given arbitrary black-box features. At each step it is driven by:

```math
\bar{\epsilon}_n,\qquad \bar{\epsilon}_{n-1},\qquad \dot{\bar{\epsilon}}_n.
```

This matches the structure of rate-dependent constitutive laws, where stress depends on strain, strain rate, and past internal state.

#### 2. The recurrence represents internal variables

The hidden state is being used as a learned set of internal variables:

```math
z_n \equiv h_n.
```

Then the model has the constitutive form

```math
z_n = \Phi(z_{n-1}, \bar{\epsilon}_n, \bar{\epsilon}_{n-1}, \dot{\bar{\epsilon}}_n),
```

```math
\hat{\bar{\sigma}}_n = \Psi(z_n, \bar{\epsilon}_n, \bar{\epsilon}_{n-1}, \dot{\bar{\epsilon}}_n).
```

That is exactly the standard internal-variable viewpoint used in mechanics: unresolved microstructural history is compressed into a small state vector, and stress is evaluated from the current loading state plus that memory.

#### 3. Causality is built in

The model only uses present and past information. At step `n` it uses:

- current strain
- previous strain
- current discrete strain rate
- previous hidden state

It does not access future loading. So the learned constitutive law is causal, which is a basic physical requirement for time-dependent materials.

#### 4. Rate dependence is built in explicitly

The viscoelastic unit-cell law depends on time derivatives. The model therefore includes

```math
\dot{\bar{\epsilon}}_n \approx \frac{\bar{\epsilon}_n - \bar{\epsilon}_{n-1}}{\Delta t}
```

as an explicit input, rather than expecting the network to infer rate effects indirectly from a raw history window.

This is important because the first stress value is strongly controlled by the initial loading rate, and the code also uses a fitted initial relation

```math
\hat{\sigma}_0 \approx a\,\dot{\bar{\epsilon}}_0 + b
```

to reflect that.

### What is physics-based, and what is still purely data-driven

So the right way to describe the model is:

- physics-guided architecture
- data-driven constitutive identification

What is physics-based:

- choice of macroscopic strain history as input
- macroscopic stress as output
- explicit use of strain rate
- hidden state interpreted as internal variables
- causal recurrent update

What is not explicitly enforced:

- microscopic equilibrium inside the unit cell
- microscopic constitutive equations for each phase
- exact homogenization formulas
- thermodynamic constraints such as dissipation inequality

So in the report, I would avoid saying the model is "derived from the governing equations" or "physics-informed" in the strong PINN sense. A better description is:

> The model is a physics-guided recurrent constitutive surrogate. Physics enters through the constitutive choice of variables, explicit rate dependence, causality, and the interpretation of the hidden state as internal variables, while the actual constitutive mapping is learned from homogenized data.

## 4. Can Part (c) Be Predicted From First Principles?

### Short answer

Yes, at least approximately.

For a generic 1D three-phase Kelvin-Voigt series composite, the first-principles expectation is:

- the minimal number of independent internal variables is `2`
- so the first principled candidate for the minimum useful hidden dimension in the RNO is also `2`

### Why it is `2` and not `3`

There are three phase strains:

```math
\epsilon_1(t),\; \epsilon_2(t),\; \epsilon_3(t),
```

but they are constrained by the macroscopic strain average:

```math
\bar{\epsilon}(t) = f_1 \epsilon_1(t) + f_2 \epsilon_2(t) + f_3 \epsilon_3(t).
```

That removes one degree of freedom. So only two phase strains are independent.

Choose

```math
z_1(t) = \epsilon_1(t),
\qquad
z_2(t) = \epsilon_2(t).
```

Then

```math
\epsilon_3(t) = \frac{\bar{\epsilon}(t) - f_1 z_1(t) - f_2 z_2(t)}{f_3}.
```

These two variables are enough to reconstruct the full microscopic state consistent with the macroscopic strain.

### Derivation of a two-state constitutive model

For phases 1 and 2:

```math
\bar{\sigma}(t) = E_1 z_1(t) + \eta_1 \dot{z}_1(t),
```

```math
\bar{\sigma}(t) = E_2 z_2(t) + \eta_2 \dot{z}_2(t),
```

so

```math
\dot{z}_1(t) = \frac{1}{\eta_1}\bar{\sigma}(t) - \frac{E_1}{\eta_1}z_1(t),
```

```math
\dot{z}_2(t) = \frac{1}{\eta_2}\bar{\sigma}(t) - \frac{E_2}{\eta_2}z_2(t).
```

For phase 3:

```math
\bar{\sigma}(t) = E_3 \epsilon_3(t) + \eta_3 \dot{\epsilon}_3(t),
```

with

```math
\epsilon_3(t) = \frac{\bar{\epsilon}(t) - f_1 z_1(t) - f_2 z_2(t)}{f_3}.
```

Therefore

```math
\bar{\sigma}(t)
=
\frac{E_3}{f_3}\left(\bar{\epsilon}(t) - f_1 z_1(t) - f_2 z_2(t)\right)

+ \frac{\eta_3}{f_3}\left(\dot{\bar{\epsilon}}(t) - f_1 \dot{z}_1(t) - f_2 \dot{z}_2(t)\right).
```

Substitute the expressions for `\dot{z}_1` and `\dot{z}_2`. After collecting terms, this becomes

```math
\bar{\sigma}(t)
=
A_0 \bar{\epsilon}(t)
+ A_1 \dot{\bar{\epsilon}}(t)
+ B_1 z_1(t)
+ B_2 z_2(t),
```

for constants `A_0`, `A_1`, `B_1`, `B_2` determined by the phase fractions and material parameters.

So the complete constitutive model can be written in state-space form as

```math
\dot{z}(t) = M z(t) + p_0 \bar{\epsilon}(t) + p_1 \dot{\bar{\epsilon}}(t),
```

```math
\bar{\sigma}(t) = c_0 \bar{\epsilon}(t) + c_1 \dot{\bar{\epsilon}}(t) + c^\top z(t),
```

where

```math
z(t) = \begin{bmatrix} z_1(t) \\ z_2(t) \end{bmatrix}.
```

This is a two-internal-variable constitutive law.

### Equivalent Laplace-domain view

From

```math
\bar{\epsilon}(s) = \bar{\sigma}(s)\sum_{i=1}^3 \frac{f_i}{E_i + \eta_i s},
```

we get

```math
\bar{\epsilon}(s) = \bar{\sigma}(s)\,\frac{N_2(s)}{D_3(s)},
```

where `D_3(s)` is cubic and `N_2(s)` is quadratic.

So

```math
\bar{\sigma}(s) = \bar{\epsilon}(s)\,\frac{D_3(s)}{N_2(s)}.
```

Because the numerator degree is one higher than the denominator degree, this can be viewed as:

- an instantaneous part depending on `\bar{\epsilon}` and `\dot{\bar{\epsilon}}`
- plus a strictly proper dynamic remainder of order `2`

Again, that points to `2` dynamic internal variables.

### What this means for part (c)

A strong theoretical answer is:

> For a generic three-phase 1D Kelvin-Voigt series composite, the homogenized constitutive law can be written using two independent internal variables because the three phase strains are linked by one macroscopic compatibility constraint. Therefore the first-principles expectation for the minimum hidden-state dimension is `2`.

### Important caveat for the neural model

The theoretical minimum and the empirical minimum do not have to match exactly.

Reasons:

- the neural hidden state is only an approximation to the true internal variables
- the model is trained in discrete time, not derived analytically
- optimization noise and finite data can make `n_hidden = 2` harder to train robustly
- extra hidden units may be used inefficiently but still improve training stability

So the report-friendly position is:

- theory suggests the answer should be around `2`
- the experiments in part (c) test whether the trained RNO actually reaches that minimum in practice

### One subtle point

If you were given the entire strain history and allowed to evaluate the constitutive operator offline in one shot, then you would not need a hidden state at all because the full history is already present in the input.

But that is not the recurrent constitutive setting.

In the RNO, the hidden state is needed to compress the past into a Markovian update law. In that setting, the relevant first-principles question is the minimum state dimension of the homogenized constitutive model, which is why `2` is the natural target here.
