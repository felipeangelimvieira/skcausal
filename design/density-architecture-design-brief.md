# Density Estimator Design Brief

## Purpose

This document frames the general design problem for treatment density estimation in `skcausal`.

It is intentionally written to avoid anchoring on current classes, tags, or implementation history. Backwards compatibility is not a constraint. The goal is to define what a density estimator should mean, what mathematical object it should expose, and what invariants the public API must guarantee.

The central issue is that a density is never an intrinsic object. A density is always defined with respect to a reference measure. Any architecture for density estimation that does not make the reference measure explicit is underspecified.

## The Core Problem

The library needs a principled abstraction for modeling the treatment distribution given covariates.

At the most fundamental level, the object of interest is the conditional law

$$
P_{T \mid X = x}
$$

of treatment `T` given covariates `X = x`.

If the API exposes a density, then it is not exposing the law itself. It is exposing a Radon-Nikodym derivative of that law with respect to a chosen reference measure `mu` on the treatment space:

$$
p_{mu}(t \mid x) = \frac{d P_{T \mid X = x}}{d mu}(t).
$$

That distinction matters.

Two implementations can represent the same conditional law while returning different numeric densities if they use different reference measures. Likewise, preprocessing that changes coordinates can change the numeric value of the density even when the underlying law is unchanged.

The design problem is therefore:

**What should `skcausal` mean by a treatment density estimator, and how should the API make the treatment space, the conditioning structure, and the reference measure explicit?**

There is a second issue intertwined with the first one.

Even after the reference measure is fixed, the library still needs to decide which treatment-related quantity is being estimated. In practice this may be:

- a conditional density `p_mu(t | x)`
- a marginal density `p_mu(t)`
- a stabilized ratio `p_mu(t | x) / p_mu(t)`
- an inverse ratio `p_mu(t) / p_mu(t | x)`
- potentially other derived scoring functionals of the conditional treatment law

These are not the same object. In particular, `p_mu(t | x) / p_mu(t)` is generally not itself a density. It is a density ratio. The architecture should therefore distinguish:

- the reference measure with respect to which derivatives are defined, and
- the target functional the estimator is supposed to return

Those are separate axes of the design.

## Why The Reference Measure Must Be First-Class

### A density is always relative to something

For continuous treatment in Euclidean space, the default reference measure is often Lebesgue measure.

For categorical treatment on a finite set, the natural reference measure is counting measure.

For mixed treatment, the natural object is usually a product measure, for example:

$$
mu = lambda \otimes nu,
$$

where `lambda` is Lebesgue measure on the continuous coordinates and `nu` is counting measure on the categorical coordinates.

The architecture must decide whether these choices are:

- implicit conventions
- explicit metadata on the estimator
- explicit user-provided objects

### The same law can have different densities

If `T` is continuous and the estimator internally rescales `T`, standardizes it, or applies another invertible transform, then the density in the transformed coordinates is not the same numeric object as the density with respect to the original coordinates.

If the API claims to return a density for the original treatment variable, then preprocessing must either:

- preserve the reference measure, or
- apply the correct change-of-variables correction

For a one-dimensional invertible transformation `z = g(t)`, this means accounting for the Jacobian:

$$
p_T(t \mid x) = p_Z(g(t) \mid x) \left| \frac{d g}{d t}(t) \right|.
$$

This is not an implementation detail. It is part of the mathematical contract of the API.

### Density ratios only make sense when the reference measure is aligned

If the library later uses densities to form objects such as:

$$
\frac{p(t \mid x)}{p(t)},
$$

then the numerator and denominator must be densities with respect to the same reference measure. Otherwise the ratio is ill-defined.

This is one reason the reference measure should not be hidden.

It also means that a stabilized quantity such as

$$
\frac{p_{mu}(t \mid x)}{p_{mu}(t)}
$$

inherits its meaning from two different derivatives defined with respect to the same `mu`. If the numerator and denominator are not defined relative to the same underlying measure, then the stabilized quantity is not mathematically coherent.

### Some treatment laws may not admit a density under the chosen measure

If the treatment distribution lives on a lower-dimensional manifold, has atoms plus an absolutely continuous component, or is otherwise singular with respect to the chosen measure, a density may fail to exist or may only exist after changing the underlying measure.

The architecture must decide whether it is trying to represent:

- a conditional law in full generality, or
- only those conditional laws that admit a density under a supported family of reference measures

## The Problem To Solve

The staff engineer or agent should focus on the density estimator itself, before considering concrete model families.

The problem is to define a sound interface for a component that answers questions of the form:

- what is the conditional treatment law given `X`?
- does that law admit a density under the chosen reference measure?
- if so, what is the value of that density or log-density at observed treatment values?
- is the requested output actually a density, a marginal density, a density ratio, or some other derived functional?
- what additional functionals of that law should the interface expose?

This leads to several design questions.

## Questions The Architecture Must Answer

### 1. What is the primary object: law, density, ratio, or treatment score?

Possible positions:

- the primary abstraction is the conditional law `P_{T | X}`
- the primary abstraction is a density with respect to an explicit measure `mu`
- the primary abstraction is a broader treatment-model object that can expose a law, a density, and selected downstream functionals such as density ratios

This is the most important design decision.

If density is primary, the API must carry the reference measure explicitly.

If law is primary, density evaluation becomes a capability that may or may not be available depending on the law and the chosen measure.

If ratio-like or stabilized quantities are also first-class outputs, the API must be explicit that these are not densities. They are derived functionals built from one or more densities, all of which must share a compatible reference measure.

### 1a. What is the target functional?

Independent of how the treatment space and reference measure are represented, the architecture must decide what family of outputs is in scope.

Examples include:

- conditional density: `p_mu(t | x)`
- conditional log-density: `log p_mu(t | x)`
- marginal density: `p_mu(t)`
- marginal log-density: `log p_mu(t)`
- stabilized density ratio: `p_mu(t | x) / p_mu(t)`
- inverse stabilized ratio: `p_mu(t) / p_mu(t | x)`

Potentially the library may later want other functionals, for example quantities corresponding to alternative balancing objectives or target measures.

The design should decide whether one abstraction is responsible for all of these, or whether:

- one abstraction represents the treatment law or density itself, and
- other abstractions derive ratios or weights from that foundation

### 2. What is the treatment space?

Before choosing algorithms, the architecture must specify what kinds of treatment spaces it intends to support.

At minimum, the space may need to cover:

- finite categorical spaces
- real-valued Euclidean spaces
- mixed product spaces with both continuous and categorical coordinates

Potentially, the space may later need to cover:

- constrained continuous supports such as positive reals or simplices
- structured discrete spaces
- lower-dimensional or singular supports

The treatment space and the reference measure are linked. They should not be designed independently.

### 3. How should treatment semantics be represented?

Raw storage dtype is not enough.

An integer-valued treatment can mean very different things:

- a continuous dose recorded as an integer
- an ordered category
- an unordered category ID
- a count-valued treatment with its own natural support

The architecture must decide whether treatment semantics are represented by:

- inferred type
- explicit metadata
- a declared treatment-space object

The important point is that semantic treatment type should derive the reference measure and preprocessing rules, not the other way around.

### 4. What preprocessing is legitimate?

The architecture must define which transformations are merely internal representations and which transformations change the meaning of the density.

Examples:

- label encoding a categorical variable is usually a representation choice, not a change of measure, as long as the estimator still models the original finite state space
- one-hot encoding can be safe as an internal feature representation, but should not be confused with changing the treatment space to a continuous vector space
- rescaling or standardizing continuous treatment changes coordinates and therefore affects density values unless the Jacobian is accounted for

This implies a strong requirement:

**The public contract must specify whether returned densities are with respect to the original treatment representation or an internal transformed representation.**

### 5. What should the public contract return?

The API must be precise about outputs.

Open questions include:

- should the fundamental scoring method be `density`, `log_density`, or both?
- should the API distinguish between returning a density and returning a density-derived score?
- if stabilized or inverse-stabilized quantities are exposed, should they be separate methods or separate estimator types?
- should output always be one scalar per observation corresponding to the joint density at the observed treatment?
- should the interface expose `sample`?
- should the interface expose `cdf`, and if so only for ordered one-dimensional settings?
- should the interface expose probabilities of sets, or is pointwise density evaluation enough?

The answer should be driven by the mathematical object being modeled, not by a particular model implementation.

### 6. How should multivariate treatment be modeled?

If treatment has multiple coordinates, the design must treat it as a joint object.

The reference measure then lives on the product space, typically as a product measure.

Examples:

- continuous-continuous treatment -> product of Lebesgue measures
- categorical-categorical treatment -> product of counting measures
- mixed treatment -> mixed product measure

The key architectural question is not which factorization strategy to use internally. It is whether the abstraction itself is defined around:

- a full joint law on a product space, or
- a collection of per-coordinate objects with optional composition rules

Those are different contracts.

### 7. What assumptions must be explicit?

The interface should make it obvious when any of the following are being assumed:

- absolute continuity with respect to a chosen measure
- conditional independence across treatment coordinates
- autoregressive factorization of the joint law
- support restrictions
- availability of a log-density but not a density, or vice versa

These assumptions should not be hidden in implementation details or passive tags.

### 8. How does the density abstraction relate to downstream causal objects?

Density evaluation is not the same thing as propensity weighting or density-ratio estimation.

Those downstream quantities may be derived from the treatment law or from densities under a shared reference measure, but they are not the same abstraction.

This matters directly for stabilized quantities. A quantity such as

$$
\frac{p_{mu}(t \mid x)}{p_{mu}(t)}
$$

is neither a conditional law nor a density. It is a derived ratio that may be useful for causal estimation, balancing, or diagnostics.

The design should clarify whether:

- the density estimator is foundational and other objects are derived from it, or
- the density estimator is only one specialized tool among several treatment-model abstractions, some of which may target ratios or weights directly

This question should be answered only after the density-estimation contract itself is clear.

## Design Constraints Implied By The Math

Any acceptable architecture should satisfy the following constraints.

### 1. The reference measure must be explicit in the conceptual model

It may be encoded as metadata, a typed object, or a well-defined convention, but it cannot remain implicit or ambiguous.

### 2. Public semantics must be defined in the original treatment space

Internal preprocessing is allowed, but the API must make clear whether outputs are densities with respect to the original treatment variable or a transformed variable.

Silent change of measure is not acceptable.

### 3. Treatment semantics must not be inferred from dtype alone

The architecture needs a principled way to represent treatment meaning, because treatment meaning determines the support and the natural reference measure.

### 4. The abstraction must separate mathematical contract from numerical strategy

Choices like factorization, autoregressive decomposition, kernel models, probabilistic regressors, classifiers, or flows are implementation strategies.

The interface should first define the object being estimated and only then allow different strategies to implement it.

### 5. The API must distinguish existence of a law from existence of a density

The system should not force every treatment model into a density API if the underlying conditional law does not admit the required derivative under the chosen reference measure.

Likewise, the system should not treat every density-derived quantity as if it were itself a density.

## Non-Goals For This Design Exercise

The point of this brief is not to choose a particular estimator family.

It is not about:

- whether to use skpro, KDEs, classifiers, flows, mixtures, or trees
- preserving the current tag set
- preserving the current public API
- choosing the fastest implementation

Those decisions come after the abstraction is defined.

## Desired Outcome

The proposed design should make it possible to answer, unambiguously, questions like these:

- what object is being estimated?
- on what treatment space is it defined?
- with respect to what reference measure is density evaluated?
- is the returned quantity a density, a marginal density, a ratio, or another treatment score?
- what transformations are allowed internally without changing the public meaning of the result?
- what outputs are guaranteed by the interface?
- what assumptions are explicit versus optional?

If the final design cannot answer those questions cleanly, the abstraction is still too loose.

## Requested Deliverable From A Staff Engineer Or Agent

Please propose:

1. the primary mathematical abstraction to expose in the library
2. how the treatment space should be represented in the API
3. how the reference measure should be represented, selected, and validated
4. what target functionals are first-class in scope: conditional density, marginal density, stabilized ratio, inverse ratio, or others
5. the exact semantics of `density` and `log_density`, if both exist
6. what preprocessing is allowed before private fit / predict logic, and what measure corrections are required
7. the contract for multivariate and mixed treatment
8. the assumptions that must be explicit in the public API
9. the relationship between this abstraction and downstream causal quantities such as weighting or density ratios

Please optimize for conceptual correctness and long-term maintainability rather than compatibility with the current code.

## Short Version

The real design problem is not how to improve one existing base class. It is how to define a mathematically sound treatment density abstraction.

The key issues are:

- a density only exists relative to a reference measure, and
- the library may need to estimate not only densities but also density-derived functionals such as stabilized ratios

The reference measure determines what the estimator means, how preprocessing affects outputs, how mixed treatment should be handled, and when downstream ratios are well-defined.

The architecture should therefore start by making the conditional treatment law, the treatment space, and the reference measure explicit, and only then decide what APIs and implementations should sit on top of that foundation.