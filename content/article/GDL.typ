#import "/typ/templates/blog.typ": *
#import "@preview/calloutly:1.2.0": (
  note, tip,
)
#show: main.with(
  title: "An Introduction to Geometric Deep Learning",
  desc: [This article aims to answer what Geometric Deep Learning is, how it works and why it is useful by providing you a walk-through that derives key components of well-known architectures from equivariance.],
  date: "2026-09-20",
  tags: (
    blog-tags.dl,
    blog-tags.math,
  ),
)

= Motivation

Most of the time we consider deep learning as an empirical subject. We are told what works, what doesn't, and what the standard solution is to some specific problems, yet few people have asked for the reason behind them.

This article aims to serve as a brief introduction to Geometric Deep Learning (GDL), which introduces a different way of understanding famous deep learning architectures such as CNNs, GCNs, and Transformers. It explains why these architectural designs may take advantage of the geometric properties of the underlying data.

Without bordering on the obscure mathematics of the original #link("https://geometricdeeplearning.com/book/")[_Geometric Deep Learning_ book], this blog aims to give you a clean big picture by deriving the key components of CNN, DeepSets, GCN, and Transformer, and revealing their relations.

#tip(title: "Before you start")[
  No worries if you don't understand some terms in the introduction above. I have tried to explain them all from first principles as much as possible, but for a smooth reading experience you should at least be familiar with the following.

  - First-year level linear algebra.
  - Basic knowledge about Convolutional Neural Networks and self-attention.

  Here are some nice YouTube videos that give a sense of them.

  - #link("https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab")[_Essence of linear algebra_] by 3Blue1Brown
  - #link("https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi")[_But what is a neural network?_] by 3Blue1Brown
  - #link("https://www.youtube.com/watch?v=KuXjwB4LzSA")[_But what is a convolution?_] by 3Blue1Brown

  As a complement, there are some awesome materials on introductory group theory. To read this article they are not a must, but they definitely help if you want to dive deeper into the formal language of GDL.

  - #link("https://www.youtube.com/watch?v=tGCqP2ytP14")[_Group Theory Step-by-Step: 1 - 7_] by TheGrayCuber
  - #link("https://www.youtube.com/watch?v=KufsL2VgELo")[_What is Group Theory? --- Group Theory Ep. 1_] by Nemean
]

= Geometric Equivariance

We have talked about taking advantage of the geometric properties of the data. To make our language clearer, we are specifically referring to the ones called _equivariance_ under a certain transformation. The following example shows a simple model $f$ that estimates the prices of items on your desk.

$
  f("Laptop", "Mouse", "Mug", "Phone") = (25000¥, 300¥, 90¥, 10000¥)
$

Note that the input data comes as a _sequence_. The model should, as we expect, assign the same value to each corresponding item if we permute the order of them, despite the fact that the prices might be interdependent, as one is likely richer if they own a more expensive laptop.

$
  f("Mug", "Laptop", "Phone", "Mouse") = (90¥, 25000¥, 10000¥, 300¥)
$

We can conclude that such a model satisfies equivariance under permutation. Generally speaking, that is, for any input $bold(x)$ after some transformation $g$, the model outputs the exact same results as if we had applied the same transformation to the output instead:

$
  f(g dot bold(x)) = g dot f(bold(x)).
$

In other words, we say that the model $f$ and the transformation $g$ _commute_, and this is exactly what we mean by equivariance.

Assumptions like this are called _inductive biases_. An inductive bias is a prior belief, built into a learning algorithm before any data is seen, about which functions are likely to solve the task. A model that is equivariant by construction does not need to learn the symmetry from examples. Instead, the symmetry is already in its architecture, which usually means fewer parameters and better sample efficiency.

Many real tasks have such symmetries. A classical computer vision task is to categorise image data with a wide range of tags, and it is more than reasonable to design the model architecture around the equivariance of interest. In the next section we will take the simplest case, shift equivariance, and derive the core component of the Convolutional Neural Network (CNN).

= Convolutional Neural Networks

To keep things simple, we will represent an image as a column vector, i.e. a single row of pixels. A barcode is a good example. It is essentially a 1D image, a row of black and white bars, where each bar plays the role of one pixel and its value records the brightness of that pixel.

$
  bold(x) = mat(delim: "[", x_1; x_2; dots.v; x_n)
$

With the signal written as a column vector, we can now ask what the first layer should compute. You already know that a CNN is built from convolution layers. The question here is why convolution is the natural choice among all linear layers. To answer it, look at the first layer. Whatever we do with the signal later, this layer has to turn it into a feature map. Suppose the layer acts on the signal by left multiplication with a matrix $W$, so $bold(y) = W bold(x)$, with $bold(y)$ of the same spatial size as $bold(x)$. For now we assume a single channel and no bias. Images have no preferred origin. If a pattern appears at a different position, the layer should produce the same response, shifted along with the pattern.

To make this precise, assume the signal is periodic, so positions wrap around in a circle. This is the _periodic boundary condition_. For example, the unit shift acts on the vector as:

$
  S mat(delim: "[", x_1; x_2; dots.v; x_n)
  = mat(delim: "[", x_n; x_1; dots.v; x_(n-1)).
$

Equivalently, $S$ is the permutation matrix

$
  S = mat(
    delim: "[",
    0, 0, dots, 0, 1;
    1, 0, dots, 0, 0;
    0, 1, dots, 0, 0;
    dots.v, dots.v, dots.down, dots.v, dots.v;
    0, 0, dots, 1, 0;
  )
$

#note(title: "Aside")[
  If you are familiar with group theory, the shifts of a periodic signal form a _cyclic group_:
  $
    G = {S^0, S^1, dots, S^(n-1)}, quad S^0 = I.
  $
  A few basic facts are enough.

  - A shift can be composed with itself. For example, $S^2$ means applying the unit shift twice, which is the same as shifting by two pixels at once.
  - Shifts can also be composed with each other, and the result is still a shift, $S^a S^b = S^(a+b)$ with the exponent taken modulo $n$. This also means every shift has an inverse; for example, $S^(n-1)$ undoes $S$.
]

Since every other shift is a composition of $S$ with itself, it is enough to require that $W$ commute with $S$:

$
  W(S bold(x)) = S(W bold(x)) quad "for every " bold(x).
$

This is _shift equivariance_. Since this holds for all $bold(x)$, it becomes $W S = S W$. Pointwise nonlinearities commute with shifts, so this property survives the rest of the feature extractor. We will see that this single equation forces $W$ to be a convolution and cuts its free parameters from quadratic to linear.

Shift equivariance implies $W S = S W$, which ensures that $W$ is a circulant matrix for some parameters $theta_0, dots, theta_(n-1)$ of the form:

$
  W = mat(
    delim: "[",
    theta_0, theta_(n-1), theta_(n-2), dots, theta_1;
    theta_1, theta_0, theta_(n-1), dots, theta_2;
    theta_2, theta_1, theta_0, dots, theta_3;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    theta_(n-1), theta_(n-2), theta_(n-3), dots, theta_0;
  )
$

by Theorem 1.

The subscript of $theta$ is the cyclic offset from the diagonal, so $theta_0$ is the diagonal entry, $theta_1$ the first subdiagonal, and $theta_(n-1)$ the first superdiagonal. Observe that the layer computes exactly a discrete convolution:

$
  y_i = sum_(j=1)^n W_(i,j) x_j = (bold(x) ast bold(theta))_i.
$

$
  bold(y) = bold(x) ast bold(theta)
$

The vector

$
  bold(theta) = mat(delim: "[", theta_0; theta_1; dots.v; theta_(n-1))
$

is the convolution kernel. In this way, we naturally deduce the convolution formula from linearity and shift equivariance. Note that our kernel size is equal to the width of the image, which reduces to a smaller one, as in most CNNs, if we manually set further weights to zero.

#note(title: "Theorem 1")[
  *Statement.* Let $n in NN$ and let $S in bb(R)^(n times n)$ be the permutation matrix of the unit shift on a periodic grid of $n$ pixels, that is, $(S bold(x))_i = x_(i-1)$ for $i = 1, dots, n$, with indices read modulo $n$ in ${1, dots, n}$. Let $W in bb(R)^(n times n)$. If $W S = S W$, then there exist $theta_0, dots, theta_(n-1) in bb(R)$ such that

  $
    W_(i,j) = theta_(i-j) quad "for all " i,j in {1, dots, n}.
  $

  The subscript of $theta$ is also read modulo $n$, so it always lies in ${0, dots, n-1}$.

  Equivalently,

  $
    W = mat(
      delim: "[",
      theta_0, theta_(n-1), theta_(n-2), dots, theta_1;
      theta_1, theta_0, theta_(n-1), dots, theta_2;
      theta_2, theta_1, theta_0, dots, theta_3;
      dots.v, dots.v, dots.v, dots.down, dots.v;
      theta_(n-1), theta_(n-2), theta_(n-3), dots, theta_0;
    )
  $

  *Proof.* The unit shift is the permutation matrix

  $
    S = mat(
      delim: "[",
      0, 0, dots, 0, 1;
      1, 0, dots, 0, 0;
      0, 1, dots, 0, 0;
      dots.v, dots.v, dots.down, dots.v, dots.v;
      0, 0, dots, 1, 0;
    )
  $

  By the definition of $S$, right multiplication by $S$ shifts column $j+1$ of $W$ to column $j$, so

  $
    (W S)_(i,j) = W_(i,j+1).
  $

  Similarly, left multiplication by $S$ shifts row $i-1$ of $W$ to row $i$, so

  $
    (S W)_(i,j) = W_(i-1,j).
  $

  Since $W S = S W$, we obtain

  $
    W_(i,j+1) = W_(i-1,j) quad forall i,j.
  $

  Letting $i' = i-1$ gives

  $
    W_(i'+1,j+1) = W_(i',j) quad forall i',j.
  $

  Thus $W_(i,j)$ is constant along each cyclic diagonal and depends only on the cyclic offset $i-j$. Define $theta_k := W_(k+1,1)$ for $k = 0, dots, n-1$. Then

  $
    W_(i,j) = theta_(i-j). quad square
  $
]

To wrap up, shift equivariance forces the linear layer to be a convolution. That said, a practical CNN also has element-wise nonlinearities, pooling, and other details. The key point here is that GDL never specifies a complete, ready-to-use architecture. It identifies the core components that a symmetry forces. The mathematical results inform how those components should be designed, and the remaining details are engineering choices compatible with that symmetry.

In the following chapters we will meet architectures you may not be familiar with. You do not need much prior expertise in them. GDL informs the symmetry principle that comes before the architectural details, and that principle is enough to see why each architecture is built the way it is.

= DeepSets

From now on, we will focus on architectures that concern the so-called _permutation equivariance_. As discussed in the introduction, many kinds of data come as a set. A point cloud is a set of 3D points, and a bag of images is a set of instances. The order of the elements carries no meaning, so permuting the input should only permute the output, and a set-level prediction should not change at all.

Since the publication of Deep Sets (Zaheer et al., 2017), its simple architecture has become a standard baseline for tasks such as population statistic estimation, point cloud classification, set expansion, and outlier detection. We will derive the layer from permutation equivariance. The input is a matrix

$
  X = mat(delim: "[", bold(x)_1; bold(x)_2; dots.v; bold(x)_n) in bb(R)^(n times d),
$

where each row $bold(x)_i in bb(R)^(1 times d)$ is one set element, $n$ is the number of elements, and the $d$ columns are the channels. Let $P in bb(R)^(n times n)$ be a permutation matrix, so $P X$ permutes the rows of $X$. Assume the layer is a fixed linear map $X |-> W X$ with $W in bb(R)^(n times n)$. Permutation equivariance requires

$
  W(P X) = P(W X) quad "for every " P " and every " X.
$

Since this holds for all $X$,

$
  W P = P W.
$

By Theorem 2, since $W$ commutes with every permutation matrix, its diagonal entries are all equal and its off-diagonal entries are all equal. That is,

$
  W = alpha I + beta bold(1) bold(1)^top
  = mat(
    delim: "[",
    alpha + beta, beta, dots, beta;
    beta, alpha + beta, dots, beta;
    dots.v, dots.v, dots.down, dots.v;
    beta, beta, dots, alpha + beta;
  ).
$

Equivalently,

$
  W_(i j) = cases(
    alpha + beta & quad i = j,
    beta & quad i != j.
  )
$

The layer therefore computes

$
  bold(y)_i = alpha bold(x)_i + beta sum_(j=1)^n bold(x)_j.
$

It is worth pausing on what this layer does. The term $alpha bold(x)_i$ keeps the element itself, and the term $beta sum_(j=1)^n bold(x)_j$ merges all elements with equal weight. This makes sense for a set. With no additional structure, the only information attached to an element is its own feature vector, so every other element enters with the same weight.

#note(title: "Theorem 2")[
  *Statement.* Let $W in bb(R)^(n times n)$. If $W P = P W$ for every permutation matrix $P in bb(R)^(n times n)$, then there exist $alpha, beta in bb(R)$ such that

  $
    W = alpha I + beta bold(1) bold(1)^top.
  $

  *Proof.* For $n <= 2$ the claim follows from the first two equalities below, so assume $n >= 3$. Let $P$ be the permutation matrix that swaps $i$ and $j$.

  Right multiplication by $P$ swaps columns $i$ and $j$ of $W$, and left multiplication by $P$ swaps rows $i$ and $j$. Since $W P = P W$, the two matrices agree entry by entry.

  At $(i,i)$, the entry of $W P$ is $W_(i,j)$, while the entry of $P W$ is $W_(j,i)$. Hence

  $
    W_(i,j) = W_(j,i).
  $

  At $(i,j)$, the entry of $W P$ is $W_(i,i)$, while the entry of $P W$ is $W_(j,j)$. Hence

  $
    W_(i,i) = W_(j,j).
  $

  For $k != i$ and $k != j$, the $(i,k)$ entry of $W P$ is $W_(i,k)$, while the $(i,k)$ entry of $P W$ is $W_(j,k)$. Hence

  $
    W_(i,k) = W_(j,k).
  $

  The second equality makes all diagonal entries equal and the third makes all off-diagonal entries in a fixed column equal.

  For distinct $k,l$ and $m$ with $m != k$ and $m != l$, the same column-swap argument with the transposition swapping $k$ and $l$ gives $W_(m,k) = W_(m,l)$, so different columns share the same off-diagonal value.

  Hence all diagonal entries equal some $d$ and all off-diagonal entries equal some $beta$. Therefore $W = (d-beta) I + beta bold(1) bold(1)^top$, and setting $alpha = d-beta$ proves the statement. $square$
]

The linear layer $W$ can be read as a message aggregation matrix. Row $i$ specifies how element $i$ merges information from all elements, and $W_(i j)$ is the weight it gives to element $j$. For $W = alpha I + beta bold(1) bold(1)^top$, every off-diagonal weight is the same $beta$, so the aggregation cannot distinguish one neighbour from another. The matrix is fixed, so the model can only tell an element apart from the rest of the set.

= Graph Convolutional Networks

A set has no structure beyond its elements. The only relation available is whether two elements are the same, which is why DeepSets can only distinguish $i = j$ from $i != j$. Many data sources carry richer relations, such as citations between papers, friendships between users, or bonds between atoms.

A graph is given by an adjacency matrix $A$, with $A_(i j) = 1$ when nodes $i$ and $j$ are connected, and $A_(i j) = 0$ otherwise. Since we now consider graph-structured data, we will refer to elements as nodes from now on.

A richer layer should let each row compute its own weights from the others, according to the graph structure. Specifically, row $bold(x)_i$ can read any other row $bold(x)_j$ and assign it a weight that reflects whether they are connected, written $a_(i j)$. The same node can then aggregate its neighbours differently under different graphs. This is why $W$ becomes a function of $A$. Later, we replace $W$ by a matrix $G(A) in bb(R)^(n times n)$, a function of the adjacency matrix. The simplest choice is $G(A) = A$.

$
  A = mat(
    delim: "[",
    a_(1 1), a_(1 2), dots, a_(1 n);
    a_(2 1), a_(2 2), dots, a_(2 n);
    dots.v, dots.v, dots.down, dots.v;
    a_(n 1), a_(n 2), dots, a_(n n);
  ).
$

Additionally, we can map each element to a feature with a shared linear transform $W_V$ before aggregation:

$
  V(X) = X W_V in bb(R)^(n times d').
$

Then

$
  Y = G(A) V(X).
$

Because the same $W_V$ is applied to every row, $V(P X) = P V(X)$. The linear map therefore preserves equivariance, while giving the layer more expressive ability. The simplest choice is $W_V = I$, which gives $V(X) = X$.

To see whether $G(A) V(X)$ is still permutation equivariant, note that permuting the nodes turns the adjacency matrix into

$
  P A P^top.
$

To see why, let's write the adjacency matrix with its row and column indices:

#figure(
  table(
    columns: 5,
    align: center,
    inset: 5pt,
    stroke: none,
    table.hline(stroke: 0.5pt, y: 1),
    table.vline(stroke: 0.5pt, x: 1),
    table.header([], [$1$], [$2$], [$dots.c$], [$n$]),
    [$1$], [$a_(1 1)$], [$a_(1 2)$], [$dots.c$], [$a_(1 n)$],
    [$2$], [$a_(2 1)$], [$a_(2 2)$], [$dots.c$], [$a_(2 n)$],
    [$dots.v$], [$dots.v$], [$dots.v$], [$dots.down$], [$dots.v$],
    [$n$], [$a_(n 1)$], [$a_(n 2)$], [$dots.c$], [$a_(n n)$],
  ),
  caption: none,
)

The entry $a_(i j)$ records the edge between node $i$ and node $j$. When the nodes are relabelled by $P$, the row index and the column index have both changed. Left multiplication by $P$ permutes the rows, and right multiplication by $P^top$ permutes the columns, hence the permuted matrix is $P A P^top$.

Theorem 3 shows that replacing the fixed layer $W X$ by $G(A) V(X)$, which depends on the graph, keeps the layer permutation equivariant as long as $G$ satisfies

$
  G(P A P^top) = P G(A) P^top.
$

#note(title: "Theorem 3")[
  *Statement.* Let $G: bb(R)^(n times n) -> bb(R)^(n times n)$ and $V: bb(R)^(n times d) -> bb(R)^(n times d')$. Define $Y(X,A) = G(A) V(X)$. If

  $
    G(P A P^top) = P G(A) P^top, quad V(P X) = P V(X)
  $

  for every permutation matrix $P in bb(R)^(n times n)$, then $Y$ is permutation equivariant,

  $
    Y(P X, P A P^top) = P Y(X,A)
  $

  for every $P$, $X$, and $A$.

  *Proof.* For any permutation matrix $P$,

  $
    Y(P X, P A P^top)
    &= G(P A P^top) V(P X) \
    &= P G(A) P^top P V(X) \
    &= P G(A) V(X) \
    &= P Y(X,A),
  $

  where $P^top P = I$. $square$
]

For Graph Convolutional Networks (GCN), $G(A)$ is the symmetric normalized propagation matrix. Add an edge from each node to itself so that each node keeps its own features:

$
  tilde(A) = A + I, quad tilde(A)_(i j) = A_(i j) + delta_(i j),
$

where $delta_(i j) = 1$ when $i = j$ and $0$ otherwise. The degree of node $i$ in $tilde(A)$ is

$
  tilde(d)_i = sum_(j=1)^n tilde(A)_(i j) = d_i + 1,
$

and the degree matrix is

$
  tilde(D) = op("diag")(tilde(d)_1, dots, tilde(d)_n).
$

The symmetric normalized propagation matrix is

$
  G(A) = tilde(D)^(-1/2) tilde(A) tilde(D)^(-1/2), quad
  G_(i j)(A) = tilde(A)_(i j) / sqrt(tilde(d)_i tilde(d)_j).
$

In the framework $Y = G(A) V(X)$, the GCN layer applies a pointwise nonlinearity:

$
  Y = sigma(G(A) V(X)), quad
  bold(y)_i = sigma(sum_(j=1)^n G_(i j)(A) bold(x)_j W_V),
$

where $V(X) = X W_V$ and $W_V in bb(R)^(d times d')$ is the learnable linear transform.

Under a permutation $P$, the features and the adjacency matrix transform together:

$
  X' = P X, quad A' = P A P^top.
$

Then

$
  tilde(A)' = P tilde(A) P^top, quad tilde(D)' = P tilde(D) P^top,
$

so

$
  G(A') = P G(A) P^top.
$

Also $V(X') = P V(X)$. By Theorem 3,

$
  Y' = P Y.
$

= Self-Attention

Since the publication of _Attention is all you need_ (Vaswani et al., 2017), this very mechanism has been applied across a wide range of tasks, such as natural language processing (NLP), audio processing, and computer vision.

In this chapter, we will discuss how to design a layer that enables each row to compute its own weights from the others, according to their content. Specifically, row $bold(x)_i$ can read any other row $bold(x)_j$ and assign it a weight that reflects its relevance to $bold(x)_i$, written $g(bold(x)_i, bold(x)_j)$. The same element can then aggregate the set differently under different inputs. This is why $W$ becomes a function of $X$. Later, we replace $W$ by a matrix $G(X) in bb(R)^(n times n)$ measuring relevance between rows.

$
  G(X) = mat(
    delim: "[",
    g(bold(x)_1, bold(x)_1), g(bold(x)_1, bold(x)_2), dots, g(bold(x)_1, bold(x)_n);
    g(bold(x)_2, bold(x)_1), g(bold(x)_2, bold(x)_2), dots, g(bold(x)_2, bold(x)_n);
    dots.v, dots.v, dots.down, dots.v;
    g(bold(x)_n, bold(x)_1), g(bold(x)_n, bold(x)_2), dots, g(bold(x)_n, bold(x)_n);
  ).
$

Each element is projected into a query, a key, and a value:

$
  Q = X W_Q, quad K = X W_K, quad V(X) = X W_V,
$

with $W_Q, W_K in bb(R)^(d times d_k)$ and $W_V in bb(R)^(d times d')$.

The relevance of element $j$ to element $i$ is the scaled dot product of its query and key:

$
  S = (Q K^top) / sqrt(d_k), quad
  S_(i j) = (bold(x)_i W_Q W_K^top bold(x)_j^top) / sqrt(d_k).
$

A row-wise softmax turns these scores into aggregation weights:

$
  G(X) = op("softmax")_("row")((Q K^top) / sqrt(d_k)), quad
  G_(i j)(X) = exp(S_(i j)) / (sum_(k=1)^n exp(S_(i k))).
$

The layer output is

$
  Y = G(X) V(X), quad
  bold(y)_i = sum_(j=1)^n G_(i j)(X) bold(x)_j W_V.
$

#note(title: "Theorem 4")[
  *Statement.* Let $G: bb(R)^(n times d) -> bb(R)^(n times n)$ and $V: bb(R)^(n times d) -> bb(R)^(n times d')$. Define $Y(X) = G(X) V(X)$. If

  $
    G(P X) = P G(X) P^top, quad V(P X) = P V(X)
  $

  for every permutation matrix $P in bb(R)^(n times n)$, then $Y$ is permutation equivariant,

  $
    Y(P X) = P Y(X)
  $

  for every $P$ and $X$.

  *Proof.* For any permutation matrix $P$,

  $
    Y(P X)
    &= G(P X) V(P X) \
    &= P G(X) P^top P V(X) \
    &= P G(X) V(X) \
    &= P Y(X),
  $

  where $P^top P = I$. $square$
]

Self-attention satisfies both conditions. Substitute $P X$ into the projections:

$
  Q(P X) = (P X) W_Q = P Q, quad K(P X) = (P X) W_K = P K,
$

so the score matrix becomes

$
  S(P X)
  &= (Q(P X) K(P X)^top) / sqrt(d_k) \
  &= (P Q (P K)^top) / sqrt(d_k) \
  &= (P Q K^top P^top) / sqrt(d_k) \
  &= P ((Q K^top) / sqrt(d_k)) P^top \
  &= P S P^top.
$

The row-wise softmax commutes with the simultaneous row and column permutation, hence

$
  G(P X) = P G(X) P^top.
$

Also $V(P X) = (P X) W_V = P V(X)$. Theorem 4 then gives

$
  Y(P X) = P Y(X).
$

Softmax is a design choice. Equivariance only requires $G(P X) = P G(X) P^top$; any weight function with this property works. Softmax makes the weights nonnegative and row-normalized, which is convenient for training.

Self-attention is the case $G = G(X)$. GCN is the case $G = G(A)$, and DeepSets is the case where $G$ is constant. All three fit the same framework

$
  Y = G(X,A) V(X).
$

= Putting it all together

In the unified form

$
  Y = G(X,A) V(X),
$

the architectures discussed so far differ in the aggregation matrix $G$ and in the symmetry they respect.

CNN uses a fixed circulant aggregation:

$
  G = mat(
    delim: "[",
    theta_0, theta_(n-1), theta_(n-2), dots, theta_1;
    theta_1, theta_0, theta_(n-1), dots, theta_2;
    theta_2, theta_1, theta_0, dots, theta_3;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    theta_(n-1), theta_(n-2), theta_(n-3), dots, theta_0;
  )
$

which commutes with the cyclic shift. Its symmetry group is the cyclic group of shifts, while the other three use the full permutation group.

DeepSets uses a fixed aggregation:

$
  G = alpha I + beta bold(1) bold(1)^top.
$

GCN uses the symmetric normalized adjacency:

$
  G(A) = tilde(D)^(-1/2) tilde(A) tilde(D)^(-1/2).
$

Self-attention computes the weights from the content:

$
  G(X) = op("softmax")_("row")((X W_Q W_K^top X^top) / sqrt(d_k)).
$

So CNN depends on the cyclic offset, DeepSets depends only on identity, GCN depends on graph structure, and self-attention depends on content.

#note(title: "Aside")[
  The name of GCN deserves a final comment. Every shift is a permutation, so a CNN is a special case of a GCN on a fixed regular graph. Each pixel is a node, and the pixels covered by the kernel are its neighbours. The adjacency matrix of this graph is circulant, so the aggregation matrix is a convolution matrix. GCN extends this aggregation to an arbitrary graph, which is why the name keeps the word _convolutional_.
]

= References

- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). _Geometric deep learning: Grids, groups, graphs, geodesics, and gauges_. arXiv preprint arXiv:2104.13478.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In _Proceedings of the 5th International Conference on Learning Representations (ICLR)_.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. _Proceedings of the IEEE, 86_(11), 2278--2324.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In _Advances in Neural Information Processing Systems_ (Vol. 30, pp. 5998--6008).
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., & Smola, A. J. (2017). Deep sets. In _Advances in Neural Information Processing Systems_ (Vol. 30, pp. 3391--3401).
