#import "/typ/templates/blog.typ": *
#import "@preview/calloutly:1.2.0": (
  callout, callout-style, calloutly, caution, code-block-style, error, important, note, success, tip, warning,
)
#import "@preview/merman:0.4.0": mermaid
#show: main.with(
  title: "Mathematics in VAE for Engineers.(WIP)",
  desc: [You might have heard of VAE. Maybe even used it in those generative models. But have you ever thought about the math behind it? What's the problem VAE originally aim to solve?],
  date: "2026-10-22",
  tags: (
    blog-tags.dl,
    blog-tags.math,
  ),
)

#let qPhi = math-color(green, $q_Phi$)
#let ptheta = math-color(blue, $p_theta$)
#show math.equation: it => {
  show sym.theta: s => math-color(blue, s)
  show sym.Phi: s => math-color(green, s)
  it
}

= Background Knowledge
== Bayesian Statistics Terminologies
// TODO: explain the difference between how we represent a model in probility form and mapping form.
#table(
  columns: 3,
  stroke: none,
  table.header([*Name*], [*Notation*], [*Description*]),
  [*Prior*], [$p(z)$], [The overall distribution of the latent variable $z$. For example, $p(z) = cal(N)(0, I)$],
  [*likelihood*],
  [$p(x|z)$],
  [The probability of the observed variable $x$ given the latent variable $z$. It also describes the generative process, so it is also called probabilistic decoder.],

  [*posterior*], [$p(z|x)$], [The probability of the latent variable $z$ given the observed variable $x$.],

  [*Evidence*],
  [$p(x)$],
  [You can think of Evidence as how likely a generative model is to generate a particular image or how reasonable the model considers this image to be. Also called Marginal Likelihood in original VAE paper],
)
These four distributions are related by Bayes' theorem:
$ p(z|x) = (p(x|z)p(z))/p(x) $

= Research Background
Since Encoder-Decoder architecture has become super common in deep learning, we won't discuss the application of VAE in detail. Instead, we will focus on the math behind VAE and try to restore the history background of when VAE was proposed.
== Problem statement
A common misconception is that VAE was originally motivated purely by the task of generating new data. In fact, one of its central motivations was a classical problem in probabilistic modeling: when the posterior over the latent variables is intractable, how can we efficiently infer the latent variables and learn the parameters of a generative model? You might ask, why is the posterior inference problem important? Generative models seem to be much more popular. In statistics, a classical scenario is that we observed some data $x$, which can be huge, and we want to know what process produced them.
#footnote[For example, we observed a lot of symptoms of your body $x$, then you might want to know what factors $z$ cause these symptoms, such as disease.]

Since we are interested in the posterior inference problem, we can first assume that the decoder $p(x|z)$ is a neural network, while its parameters remain unknown. We denote it as $ptheta (x|z)$, where $theta$ represents the parameters of the neural network. And, $p(z)$ as the distribution over latent space, is determined by Encoder.
#footnote[In inference perspective, if you integrate $p(z|x) p(x)$ over all the data $x$, you do get the distribution of $p(z)$.]
Since both encoder and decoder are something we are going to find, we can simply impose a prior distribution(e.g., a standard isotropic Gaussian) to act as a "constraint" on both models.
Then according to Bayes' theorem, we can compute the posterior $p(z|x) = (ptheta (x|z)p(z))/p(x)$.
The only problem is that the evidence $p(x) = integral ptheta (x|z)p(z)d z$ is often intractable when the likelihood is parameterized by a nonlinear neural network, especially when the latent space is high-dimensional.

Wait, why can you just assume that the generative model $ptheta (x|z)$ is known? How about we assume that the posterior $p(z|x)$ is known? Just use a neural network to fit it, then the problem is solved! Of course you can do that, but then the Bayes' theorem would become $p(x|z) = (p(z|x)p(x))/p(z)$, then $p(x) = integral p(z|x)p(x)d z$ is still intractable.
In other words, once you know side of the Bayes' theorem, you already have a theoretical solution provided by it; we just cannot obtain its explicit form.

What if we use two neural networks to model the generative process and the posterior inference process simultaneously?
This is the central idea behind VAE. We use a decoder $ptheta (x|z)$ to model how data are generated from latent variables, and an encoder $qPhi (z|x)$ to approximate the intractable posterior ($ptheta (z|x)$).
However, this introduces a new challenge: how can we jointly optimize the two networks when the true posterior and the marginal likelihood are intractable?

= Design: VAE Derivation
Remember that we now have a Decoder $ptheta (x|z)$ and a theoretical Encoder $ptheta (z|x)$ derived from our decoder via Bayes' theorem. Since we can't obtain the explicit form of this theoretical Encoder, we are now using another neural network, with its parameters denoted as $Phi$, to approximate it. This approximated Encoder is denoted as $qPhi (z|x)$.

== Evidence Lower Bound (ELBO)
#note[
  Watch #link("https://www.youtube.com/watch?v=KHVR587oW8I")[this video] to learn KL divergence.
]
For simplicity's sake, I will use $qPhi$ to represent $qPhi (z|x)$ and $ptheta$ to represent $ptheta (z|x)$ in this section.

Since one of our goals is to make $qPhi$ to be as close to $ptheta$ as possible, we can simply minimize its KL divergence.
$
  D_"KL"(qPhi || ptheta) = integral_z qPhi dot log(qPhi / ptheta) d z
$
And don't forget that we still don't know the Decoder's parameters $theta$ at this point. So another goal is naturally optimize our Decoder, which means to maximize the probability of generating data in our datasets $ptheta(x)$. We will start with only KL divergence, in later steps you will see $ptheta(x)$ emerges by itself.
#tip[
  You might wonder how can we minimize the KL divergence between two distributions we don't know its form? Something called *Monte Carlo sampling* will be used in later steps to estimate certain distributions.
  #footnote[Strictly speaking, Monte Carlo sampling is used to estimate the expectation of a distribution.]
  For now, let's treat them as ordinary distributions and later reformulate them in a form amenable to estimation.
]

Because of the definition of expected value, we can rewrite this integral as
$
  D_"KL"(qPhi || ptheta) & = integral_z qPhi dot log qPhi / ptheta d z \
                         & = bb(E)_(qPhi) [log qPhi / ptheta] \
                         & = bb(E)_(qPhi) [log qPhi] - bb(E)_(qPhi) [log ptheta]
$
We know that $ptheta (z|x) = (ptheta (z,x)) / (ptheta (x))$, thus
$
  D_"KL"(qPhi || ptheta) & = bb(E)_(qPhi) [log qPhi] - bb(E)_(qPhi) [log(ptheta (z,x)) / (ptheta (x))] \
                         & = bb(E)_(qPhi) [log qPhi ] - bb(E)_(qPhi)[log ptheta (z,x)] + bb(E)_(qPhi)[log ptheta (x)]
$
$ptheta (x)$ is not related to $z$, thus $bb(E)_(qPhi)[log ptheta (x)] = log ptheta (x)$.
$
  D_"KL"(qPhi || ptheta) & = log ptheta (x) - bb(E)_(qPhi)[log ptheta (z,x) - log qPhi]
$
Here we get both variables we want to optimize in a single equation. And notice that
$
  bb(E)_(qPhi)[log ptheta (z,x) - log qPhi] = log ptheta (x) - D_"KL"(qPhi || ptheta)
$
meaning by maximizing $bb(E)_(qPhi)[log ptheta (z,x) - log qPhi]$ we can minimize the KL divergence and maximize $ptheta (x)$ at the same time! We now call this useful term Evidence Lower Bound or ELBO for short
$
  "ELBO" = bb(E)_(qPhi)[log ptheta (z,x) - log qPhi]
$
, since
$
  log ptheta (x) = D_"KL"(qPhi || ptheta) + bb(E)_(qPhi)[log ptheta (z,x) - log qPhi]
$
and $D_"KL"(qPhi || ptheta) >= 0$, thus $bb(E)_(qPhi)[log ptheta (z,x) - log qPhi]$ is the lower bound of log evidence $log ptheta (x)$.

== Find the Gradient of ELBO
Now the issue becomes how do we maximize ELBO. A natural solution is to find its gradient, and then we can perform gradient descent to $-"ELBO"$. Let's first try to find the gradient w.r.t #footnote[with respect to] $theta$.
$
  nabla_theta "ELBO" & = nabla_theta (bb(E)_(qPhi(z|x))[log ptheta (z,x) - log qPhi(z|x)]) \
                     & = nabla_theta (integral_z qPhi(z|x)[log ptheta (z,x) - log qPhi(z|x)] d z) \
                     & = integral_z qPhi(z|x) space nabla_theta [log ptheta (z,x) - log qPhi(z|x)] d z
$
Since the second term $log qPhi(z|x)$ is not related to $theta$, its gradient w.r.t $theta$ is just zero.
$
  & nabla_theta "ELBO" \
  & = integral_z qPhi(z|x) space nabla_theta [log ptheta (z,x)] d z \
  & = bb(E)_(qPhi(z|x)) nabla_theta [log ptheta (z,x)]
$
After #link(<Reparameterization>)[_The Reparameterization Trick_], we can perform Monte Carlo sampling to estimate the value of this expectation. The basic idea is just sampling bunch of latents $z$ out of the distribution $qPhi(z|x)$, which can be done after _The Reparameterization Trick_, then calculate the average value of the gradient given those z.
$
  & bb(E)_(qPhi(z|x)) nabla_theta [log ptheta (z,x)] \
  & approx 1/L sum^L_(l=1) nabla_theta [log ptheta(z, x)]
$
Where $L$ means the number of Monte Carlo samples.

To sum up, so far there is nothing blocking us to find the gradient of the ELBO w.r.t $theta$. But we won't be that lucky when we try to find the gradient w.r.t $Phi$.
$
  nabla_Phi "ELBO" & = nabla_Phi (bb(E)_(qPhi(z|x))[log ptheta (z,x) - log qPhi(z|x)]) \
  & = nabla_Phi integral_z qPhi(z|x)[log ptheta (z,x) - log qPhi(z|x)] d z \
  & = integral_z nabla_Phi [qPhi(z|x)[log ptheta (z,x) - log qPhi(z|x)]] d z \
  & = integral_z qPhi(z|x) nabla_Phi [log ptheta (z,x) - log qPhi(z|x)] d z + integral_z nabla_Phi qPhi(z|x) space [log ptheta (z,x) - log qPhi(z|x)] d z
$
Once again, we can do Monte Carlo sampling to the first term since it is an expectation. But we can't do so to the second term. That's exactly where _The Reparameterization Trick_ comes in.
== The Reparameterization Trick <Reparameterization>
The origin of the problem is that $qPhi(z|x)$, which is the distribution we're sampling from, is related to the value we are finding gradient with respect to ($Phi$).
So the solution here is to replace $qPhi(z|x)$ with an unrelated distribution, while keeping the value of ELBO unchanged. That sounds impossible first, but a special design is applied here to make this possible.

The encoder network doesn't directly output a latent variable $z$, instead, it gives you mean and variance according to $x$ *deterministically*. Later we sample an $z$ from the Gaussian distribution given those parameters.

#mermaid(
  "flowchart LR
      X((x)) --> E[[Encoder]]

      E --> MU[Mean μ]
      E --> VAR[Variance σ²]

      MU --> G[(Gaussian Distribution)]
      VAR --> G

      G --> S{{Sampling}}
      S --> Z((z))

      classDef input fill:#F1F5F9,stroke:#64748B,stroke-width:2.5px,color:#334155
      classDef encoder fill:#DBEAFE,stroke:#2563EB,stroke-width:3px,color:#1D4ED8
      classDef params fill:#EDE9FE,stroke:#7C3AED,stroke-width:2px,color:#5B21B6
      classDef gaussian fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#075985
      classDef sampling fill:#FEF3C7,stroke:#D97706,stroke-width:3px,color:#92400E
      classDef output fill:#DCFCE7,stroke:#16A34A,stroke-width:2.5px,color:#166534

      class X input
      class E encoder
      class MU,VAR params
      class G gaussian
      class S sampling
      class Z output
    ",
  width: 100%,
  site-config: (
    layout: "dagre",
    theme: "base",
    look: "classic",
  ),
)

We denote the mean and variance as $mu_Phi (x)$ and $sigma_Phi^2 (x)$, thus we have
$
  qPhi(z|x) = cal(N) (mu_Phi (x), sigma_Phi^2 (x))
$

This is also true for the decoder. (In some cases, the variance is fixed.)
$
  ptheta(x|z) = cal(N)(mu_theta (z), sigma_theta^2 (z))
$
We will use this feature later on.
<decoder_samp>

Now we know that $z$ is sampled from an Gaussian distribution, thus
$
   qPhi(z|x) & = 1/(sigma_Phi (x) sqrt(2pi)) exp(-(z-mu_Phi (x))^2/(2 sigma_Phi^2 (x))) \
  "let" f(z) & =^"def" log ptheta (z,x) - log qPhi \
      "ELBO" & = integral qPhi(z|x) f(z) d z \
             & = integral 1/(sigma_Phi (x) sqrt(2pi)) exp(-(z-mu_Phi (x))^2/(2 sigma_Phi^2 (x))) f(z) d z
$
With that in mind, now let's define a new variable $epsilon$, sampled from Standard Normal distribution $cal(N)(0,1)$.
Then there is the key step in The Reparameterization Trick: we can use linear transformation(denoted as $g_Phi (epsilon, x)$) to convert $epsilon$ to $z$.
$
  "If" epsilon ~ cal(N)(0,1) "and" z ~ cal(N) (mu_Phi (x), sigma_Phi^2 (x)) \
  "Then" z = g_Phi (epsilon, x) = mu + sigma epsilon \ d z = sigma d epsilon
$
Let's now replace all the $z$ with $epsilon$
$
  "ELBO" & = integral 1/(sigma sqrt(2pi)) exp(-(sigma epsilon)^2/(2 sigma^2)) f(mu + sigma epsilon) sigma d epsilon \
         & = integral 1/(sqrt(2pi)) exp(-epsilon^2/2) f(mu + sigma epsilon) d epsilon \
         & = bb(E)_(epsilon ~ cal(N)(0,1)) [f(mu + sigma epsilon)]
$
At this point, the distribution used for sampling the expectation has been changed from the problematic $qPhi$ to $epsilon$, which is independent of the optimizable parameters $Phi$. Thus, finding gradient is not a problem anymore:
$
  nabla_(theta, Phi) "ELBO" approx 1/L sum^L_(i=1) nabla_(theta, Phi) [log ptheta (x, g_Phi (epsilon, x)) - log qPhi (g_Phi (epsilon, x) | x)]
$

This optimization method is called _Stochastic Gradient Variational Bayes_ or SGVB. To sum up, SGVB did 2 things:
+ Monte Carlo estimates the expectation.
+ Reparameterization separates the randomness from the optimizable parameters, making gradient can be propagated to $Phi$.
== Separate the Reconstruction Term and Regularization Term
Having gone through the journey above, we are now ready to optimize the ELBO. But we can do even better.

For simplicity, I will use the letter $z$ below to denote $g_Phi (epsilon, x)$, but please keep in mind that this is the result after reparameterization.
$
  "ELBO" & = bb(E)_(epsilon ~ cal(N)(0,1)) [
             underbrace(
               log ptheta (x, z),
               &= log (ptheta (x|z) dot ptheta(z)) \
               &= log ptheta (x|z) + log ptheta(z)
             )
             - log qPhi (z | x)] \
         & = bb(E)_(epsilon ~ cal(N)(0,1))[log ptheta (x|z)]
           - underbrace(
             bb(E)_(epsilon ~ cal(N)(0,1))[log (qPhi (z | x))/ptheta(z)],
             D_"KL" (qPhi (z | x) || ptheta(z))
           ) \
         & approx underbrace(1/L sum^L_(i=1) log ptheta(x|z^((i))), "Reconstruction Term") -
           underbrace(D_"KL" (qPhi (z^((i)) | x) || ptheta(z^((i)))), "Regularization Term")
$
This way, the Reconstruction Term can keep being estimated using Monte Carlo sampling, while the KL Divergence can be computed analytically when both distributions are Gaussian.($ptheta(z) = cal(N)(0,1)$ because we want our latent space to be Gaussian.)

As the same trick played in #link(<Reparameterization>)[_The Reparameterization Trick_], $qPhi(z|x)$ can be represented as
$
  qPhi(z|x) = cal(N) (mu_Phi (x), sigma_Phi^2 (x))
  = 1/(sigma_Phi (x) sqrt(2pi)) exp(-(z-mu_Phi (x))^2/(2 sigma_Phi^2 (x)))
$
Again, for simplicity, I will use the $sigma$ and $mu$ below to denote $mu_Phi (x)$ and $sigma_Phi (x)$. Then the KL Divergence can be expanded as follows:
$
  D_"KL"(qPhi(z | x) parallel
    ptheta(z)) & = integral qPhi(z | x)
                 log (qPhi(z | x) / ptheta(z)) dif z \
               & = integral qPhi(z | x)
                 log (
                   1 / sigma
                   exp(-1/2 ((z - mu)^2 / sigma^2 - z^2))
                 ) dif z \
               & = -1/2 integral qPhi(z | x)
                 [log sigma^2 - z^2 + 1/sigma^2 (z - mu)^2]
                 dif z \
               & = -1/2 [
                   log sigma^2 integral qPhi(z | x) dif z
                   - integral z^2 qPhi(z | x) dif z
                   + 1/sigma^2 integral (z - mu)^2
                   qPhi(z | x) dif z
                 ]
$

Using the properties of the Gaussian distribution:

$
             integral qPhi(z | x) dif z & = 1 \
         integral z^2 qPhi(z | x) dif z & = mu^2 + sigma^2 \
  integral (z - mu)^2 qPhi(z | x) dif z & = sigma^2
$

Substituting these results:

$
  D_"KL"(qPhi(z | x) parallel
    ptheta(z)) & = -1/2 [
                   log sigma^2 - (mu^2 + sigma^2)
                   + 1/sigma^2 sigma^2
                 ] \
               & = -1/2 [
                   log sigma^2 - mu^2 - sigma^2 + 1
                 ] \
               & = 1/2 [
                   mu^2 + sigma^2 - 1 - log sigma^2
                 ]
$

Substituting the closed-form KL Divergence into the ELBO:
$
  "ELBO" & = bb(E)_(qPhi(z | x))
           [log ptheta(x | z)]
           - D_"KL"(qPhi(z | x) parallel ptheta(z)) \
         & approx underbrace(
             1/L sum_(i=1)^L
             log ptheta(x | z^((i))), "Reconstruction Term"
           )
           + underbrace(
             1/2 [
               log sigma^2 - mu^2 - sigma^2 + 1
             ], "Regularization Term"
           )
$

where $z^((i)) = mu + sigma epsilon^((i))$ and
$epsilon^((i)) tilde cal(N)(0, 1)$.

== From Theory to Loss Function
As we mentioned #link(<decoder_samp>)[earlier], the network gives you means and variances, thus $ptheta(x|z)$ is also Gaussian.
$
  ptheta(x|z) & = cal(N)(mu_theta (z), sigma_theta^2 (z)) \
              & = 1/(sigma_theta (z) sqrt(2pi)) exp(-(x-mu_theta (z))^2 / (2 sigma_theta^2 (z))) \
$
$
  cal(L)_"recon" & = - 1/L sum_(i=1)^L log ptheta(x | z^((i))) \
                 & = 1/L sum_(i=1)^L (log sigma_theta (z) + 1/2 log(2pi) + (x-mu_theta (z))^2 / (2 sigma_theta^2 (z)))
$
In the paper's experiment they found that the number of samples $L$ per data point can be set to 1 as long as the minibatch size was large enough. Thus $cal(L)_"recon"$ can be further simplified as
$
  cal(L)_"recon" = log sigma_theta (z) + 1/2 log(2pi) + (x-mu_theta (z))^2 / (2 sigma_theta^2 (z))
$

If we now substitute $z = g_Phi (epsilon, x)$ back and don't consider $sigma$ for now, $cal(L)_"recon"$ is proportional to $(x-mu_theta (g_Phi (epsilon, x)))^2$, which is exactly MSE loss.

For the Gaussian Decoder with a fixed unit variance, the training Loss for each data point in the end is:
$
  cal(L)_"VAE" =
  1/2 || x-mu_theta (g_Phi (epsilon, x)) ||^2 - 1/2 [
    log sigma_Phi^2 (x) - mu_Phi^2 (x) - sigma_Phi^2 (x) + 1
  ]
$
