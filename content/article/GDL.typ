#import "/typ/templates/blog.typ": *
#import "@preview/calloutly:1.2.0": (
  callout, callout-style, calloutly, caution, code-block-style, error, important, note, success, tip, warning,
)
#show: main.with(
  title: "A quick intro to Geometric Deep Learning(WIP)",
  desc: [This article aims to answer what Geometric Deep Learning is, how it works and why it is useful by providing you a quick walk-through to derive CNN from shift-equivariance.],
  date: "2026-09-20",
  tags: (
    blog-tags.dl,
    blog-tags.math,
  ),
)

Most of the time we consider deep learning an engineering problem -- we are told what works what doesn't, what
standard solution is to a specific problem.
But very few people ask why. Geometric Deep Learning gives us the math tool to understand why all those famous architectures, including CNN, LSTM, Transformer and so on, work better than, say, plain MLP.

To further explain how it works and why it is useful, we will skip all the math in the front part of #link("https://geometricdeeplearning.com/book/")[Original _Geometric Deep Learning_ book] and give you a quick walk-through to derive CNN from shift-equivariance.

#tip[
  = Before you start
  Here are some nice youtube videos can give some sense of Group Theory. It's not required to understand this article, but it definitely helps.
  - #link("https://www.youtube.com/watch?v=tGCqP2ytP14")[_Group Theory Step-by-Step: 1 - 7_] by TheGrayCuber
  - #link("https://www.youtube.com/watch?v=KufsL2VgELo")[_What is Group Theory? — Group Theory Ep. 1_] by Nemean

  No need to remember all the details, feel it.
  // TODO: understand how CNN works(what is feature map, what is kernel)
]

= Shift Equivariance
Let's start with an 1D example with 4 pixels.
$
  bold(x) = vec(x_0, x_1, x_2, x_3)
$
You can imagine, each value of x is the brightness of a pixel.

Since we want to discuss shift equivariance, we need to first define what is a shift operation. To make things simple at first, we will assume images have no boundary, which means positions wrap around in a circle. For example, if we shift the image by 1 pixel to the right, the new image will be:
$
  vec(x_0, x_1, x_2, x_3) arrow vec(x_3, x_0, x_1, x_2)
$
Let's call this shift operation $S$. Naturally we got
$S(vec(x_0, x_1, x_2, x_3)) = vec(x_3, x_0, x_1, x_2)$.

#note[
  If you were familiar with group theory, you should notice we've already formed a group $G={I, S, S^2, S^3}$.
  But I want to make some obvious things clear for those who have zero background in group theory.

  - An operation can be repeated multiple times, and we denote that as $S^n$, $n$ is the number of times we apply the operation.
    $
      S^2(vec(x_0, x_1, x_2, x_3)) = S(S(vec(x_0, x_1, x_2, x_3))) = S(vec(x_3, x_0, x_1, x_2)) = vec(x_2, x_3, x_0, x_1)
    $
  - Also an operation can be composed.
  // TODO
]

Now suppose we have an image-processing layer $F$ that takes an image as input and output a feature map with the same size as the input.
$
  F(vec(x_0, x_1, x_2, x_3)) = vec(y_0, y_1, y_2, y_3)
$
Then the shift equivariance is simply $F(S(x)) = S(F(x))$, meaning *the shift in the input will result in the same shift in the output*.

// TODO
// #callout(title: "Extra example")[

// ]

Sometimes we call properties like this *Inductive Bias*. To my understanding, Inductive Bias are some kinds of rules(Or prior knowledge in fancy wolds) we observed from our data.
For example, in the model's perspective, Shift Equivariance means that if the cat in the image moved a few pixels, the model should still be able to recognize that's a cat. By embedding those rules into loss function(e.g. by adding regularization) or architecture, we can make the model work more efficiently.
We will see exactly how that works in the next chapter.

= Deriving CNN
== Start with Representing our Layer as a Matrix
// TODO: should we start with matrix?

== Another Mathematical Way to Represent Image
To align our final derivation with the standard convolution expression you are(should be) familiar with, we need to introduce a way that represents a signal(in this case, an image) as the composition of a series of impulse functions. It takes some time to get used to, so let's take it slowly.

Another way to represent a signal rather than a vector, is to represent it as a mapping(a fancy way to say function) from the "position"(a set of integers) to the value of that position $x: Z mapsto R$.
// TODO: add more examples
This kind of representation is widely used in GDL. And you should see that it contains exactly the same information as a vector.

The simplest form of this kind of mapping is an impulse function, which is defined as:
$
  delta_0(u) = cases(
    1\, u = 0,
    0\, u!= 0
  ) space, u in Z
$
By itself, it means one bright pixel at the origin...
```
... 0 0 1 0 0...
        ^
        0
```
..., which is not very useful. But we can combine it with the shift operation we defined earlier to represent any image. To make things easier to read, we define $delta_v = S^v (delta_0)$. Then any signal $x$ can be assembled out of these one-pixel signals:
$
  x(u) = sum_(v in Z) a(v) delta_v (u)
$
For simplicity's sake $x$ is also written as $x = sum_(v in Z) a(v) delta_v$. Keep in mind that $delta_v$ is a function.

For example, an image like this:
```
position:      -1   0   1
x:              2   5  -1
```
can be represented as $x = 2 delta_(-1) + 5 delta_0 -1 delta_1$.

== Redefine the Shift Operation


== From Shift Equivariance to Convolution
Let's also start from a linear layer $F$, then we will extend to multiple layers with activation functions. For now, Shift Equivariance and Linearity are all we need to derive the convolution operation.

Plug in the representation of $x$ above to our layer $F$, we have:
$
  F(x) = F(sum_(v in Z) a(v) delta_v)
$
#note[
  Just want to remind you that $F$ takes a function $x$(signal) and outputs another function(feature map). The way we represent the feature map is also by a function that takes a position(we call it $u$) and outputs the value of that position in the feature map.

  So the complete form of $F(x)$ should be:
  $
    F(x)(u) = F(sum_(v in Z) a(v) delta_v (u))
  $
]
Because $F$ is linear, we can move it inside the summation:
$
  F(x) = sum_(v in Z) a(v) F(delta_v)
$
Because $F$ is shift equivariant, we can replace $F(delta_v)$ with $S^v (F(delta_0))$.
$
  F(x) = sum_(v in Z) a(v) S^v (F(delta_0))
$
And you can see that $F(delta_0)$ is a function that's not related neither to the input $x$ nor to $v$. That's what we call as the *kernel*. We denote it as $theta = F(delta_0)$.
$
  F(x) = sum_(v in Z) a(v) S^v theta
$
#note[
  It might look a bit confusing at first, but notice that $theta: Z mapsto R$ ($theta(u) = F(delta_0)(u)$), which, recall from the previous chapter, is exactly the same as a vector. And if we assume $u$ is a fixed position, then $S^v theta("const")$ is sim
]
