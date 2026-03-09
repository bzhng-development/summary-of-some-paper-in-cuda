## 1. Executive Summary

This paper introduces **Barnes-Hut t-SNE** and **dual-tree t-SNE**, two tree-based algorithms that reduce the computational complexity of learning t-distributed Stochastic Neighbor Embeddings from $O(N^2)$ to $O(N \log N)$ while requiring only $O(N)$ memory. By approximating repulsive forces using quadtrees (or octrees) and sparsifying input similarities via vantage-point trees, these methods enable the visualization of massive datasets containing millions of objects, such as the **TIMIT** speech dataset with **1,105,455** frames, which was embedded in under four hours. Experimental results on five large-scale datasets, including **MNIST** (70,000 images) and **SVHN** (630,420 images), demonstrate that the Barnes-Hut variant notably outperforms the dual-tree approach in speed-accuracy trade-offs, making high-dimensional data exploration practically feasible where standard t-SNE would take days or be impossible.

## 2. Context and Motivation

### The Bottleneck of Visualizing High-Dimensional Data
The core problem addressed by this paper is the **computational intractability** of visualizing large, high-dimensional datasets using state-of-the-art embedding techniques. In modern data analysis, researchers often deal with objects described by thousands of variables (e.g., pixel values in images, acoustic features in speech, or gene expression levels). To make sense of such data, analysts rely on **low-dimensional embeddings**, where each high-dimensional object is mapped to a point in 2D or 3D space. The goal is to preserve the local structure: similar objects should appear close together, while dissimilar ones should be far apart.

Among the various embedding techniques, **t-Distributed Stochastic Neighbor Embedding (t-SNE)** has emerged as a particularly successful method for visualization. Unlike linear methods like Principal Component Analysis (PCA), t-SNE excels at revealing complex, non-linear manifolds and clustering structures. However, t-SNE suffers from a severe scalability limitation: its computational complexity scales **quadratically**, $O(N^2)$, with the number of data points $N$.

This quadratic scaling arises from the mathematical formulation of the t-SNE objective function. The algorithm minimizes the Kullback-Leibler (KL) divergence between two probability distributions: one representing pairwise similarities in the high-dimensional input space ($P$) and another in the low-dimensional embedding space ($Q$). The gradient of this objective function, which guides the optimization, involves summing forces between **every pair** of points $(i, j)$ in the dataset:

$$
\frac{\partial C}{\partial y_i} = 4 \sum_{j \neq i} (p_{ij} - q_{ij}) q_{ij} Z (y_i - y_j)
$$

Here, $p_{ij}$ and $q_{ij}$ are the similarity probabilities, and $Z$ is a normalization constant. Because the summation iterates over all $N(N-1)$ pairs, computing the gradient for a single iteration requires $O(N^2)$ operations.
*   **Real-world impact:** For a dataset with $N=10,000$ points, this results in roughly $10^8$ pairwise calculations per iteration, which is manageable. However, for $N=1,000,000$ (one million) points, the number of interactions explodes to $10^{12}$. As noted in the paper, standard t-SNE becomes practically impossible for datasets larger than a few thousand objects, forcing analysts to either subsample their data (losing information) or abandon the technique entirely.

### Limitations of Prior Approaches
Before this work, several strategies existed to mitigate the $O(N^2)$ bottleneck, but each carried significant drawbacks:

1.  **Landmark t-SNE:** This approach selects a small subset of "landmark" points, computes the embedding for these landmarks, and then places the remaining points based on their relationship to the landmarks.
    *   *Shortcoming:* This does not facilitate the visualization of **all** available data. The global structure might be distorted, and fine-grained details in the non-landmark points are lost.
2.  **Parametric t-SNE:** Instead of optimizing point locations directly, this method trains a neural network (a parametric function) to map input data to the embedding space using stochastic gradient descent.
    *   *Shortcoming:* This substantially complicates the learning process. It requires tuning network architecture and hyperparameters, and it is generally only applicable when the input data consists of fixed-size high-dimensional vectors, limiting its flexibility compared to the non-parametric original t-SNE.
3.  **Fast Multipole Methods (FMM):** In physics and other fields, FMMs are used to accelerate $N$-body simulations to $O(N)$ by expanding interaction forces into series (e.g., Hermite polynomials for Gaussian forces).
    *   *Shortcoming:* The paper explicitly notes that FMMs cannot be readily applied to t-SNE. The repulsive forces in t-SNE are governed by a **Student-t kernel** (specifically, a Cauchy distribution), which has heavy tails. To the authors' knowledge, no appropriate mathematical expansion exists for Student-t interactions that would allow the factorization required by FMMs. Adapting FMM would require replacing the Student-t kernel with a Gaussian approximation, fundamentally altering the algorithm's ability to handle dissimilar points correctly.

### Positioning Relative to Existing Work
This paper positions itself by adapting **tree-based approximation algorithms** from computational physics and statistics to the specific constraints of t-SNE. The authors draw inspiration from two distinct lines of prior work:

*   **Space-Partitioning Trees for Similarity:** To handle the input similarity matrix $P$, the paper leverages **vantage-point trees** (Yianilos, 1993). Prior work in nearest-neighbor search demonstrated that metric trees could efficiently find neighbors in $O(N \log N)$ time. The authors adapt this to create a **sparse approximation** of $P$, recognizing that for dissimilar points, $p_{ij}$ is infinitesimally small and can be treated as zero without affecting the embedding quality.
*   **Tree-Based N-Body Solvers:** To handle the expensive gradient computation (specifically the repulsive forces), the paper adapts two classic $O(N \log N)$ algorithms:
    1.  **The Barnes-Hut Algorithm (1986):** Originally developed for astrophysical simulations to calculate gravitational forces. It approximates the force exerted by a distant cluster of stars as a single force from their center of mass.
    2.  **The Dual-Tree Algorithm (Gray and Moore, 2001):** A more advanced technique that traverses two trees simultaneously to approximate interactions between groups of points (cell-cell interactions) rather than just point-group interactions.

While these algorithms were well-established in astronomy and density estimation, they had not been successfully integrated into the t-SNE framework. The paper's unique contribution lies in modifying these algorithms to work with the **Student-t kernel** and the specific gradient structure of t-SNE.

Crucially, the paper distinguishes itself from concurrent work (such as Yang et al., 2013) by not only implementing the Barnes-Hut approximation but also investigating the **dual-tree algorithm** for t-SNE. Furthermore, it provides a rigorous empirical comparison of both methods across five massive datasets, challenging the assumption that the more complex dual-tree approach would necessarily outperform the simpler Barnes-Hut variant in this specific context. The result is a practical solution that maintains the theoretical properties of t-SNE while unlocking the ability to visualize datasets with **millions** of objects.

## 3. Technical Approach

This section details the mechanical innovations that allow t-SNE to scale from thousands to millions of data points. The core idea is to replace the exact, quadratic-cost calculation of pairwise forces with two distinct approximations: a **sparse approximation** for the input similarities (attractive forces) and a **tree-based approximation** for the embedding interactions (repulsive forces).

### 3.1 Reader orientation (approachable technical breakdown)
The system is a modified optimization engine for t-SNE that replaces the brute-force calculation of every pairwise interaction with a hierarchical "summary" system based on spatial trees. It solves the $O(N^2)$ bottleneck by observing that distant points exert nearly identical repulsive forces, allowing the algorithm to group them into single computational units, while simultaneously ignoring negligible attractive forces between dissimilar items.

### 3.2 Big-picture architecture (diagram in words)
The accelerated t-SNE pipeline consists of three sequential processing stages that transform raw high-dimensional data into a low-dimensional map:
1.  **Sparse Input Similarity Module:** Takes the raw high-dimensional dataset $D$ and constructs a **vantage-point tree** to efficiently find nearest neighbors, outputting a sparse probability matrix $P$ where only local connections are non-zero.
2.  **Embedding Tree Constructor:** At every iteration of the optimization, takes the current 2D/3D point locations $E$ and builds a **quadtree** (or octree), storing the count and center-of-mass for every spatial cell.
3.  **Approximate Gradient Engine:** Traverses the embedding tree to compute the gradient; it calculates exact attractive forces using the sparse $P$ matrix and approximates repulsive forces by summing interactions between points and tree cells (Barnes-Hut) or between pairs of tree cells (Dual-Tree), outputting the update vector for each point.

### 3.3 Roadmap for the deep dive
*   **Approximating Input Similarities:** We first explain how the algorithm sparsifies the input data using vantage-point trees, reducing the cost of computing attractive forces from $O(N^2)$ to $O(N \log N)$.
*   **The Barnes-Hut Approximation:** We then detail the construction of the quadtree on the embedding space and the specific "summary condition" used to approximate repulsive forces between points and clusters.
*   **The Dual-Tree Approximation:** We examine the more complex dual-tree traversal that attempts to summarize interactions between two clusters simultaneously, and why it introduces unique bookkeeping challenges.
*   **Optimization and Hyperparameters:** Finally, we describe the specific gradient descent settings, including the early exaggeration factor and the trade-off parameter $\theta$, which control the balance between speed and accuracy.

### 3.4 Detailed, sentence-based technical breakdown

#### Approximating Input Similarities via Vantage-Point Trees
The first major optimization targets the computation of the **attractive forces**, which depend on the input similarity probabilities $p_{ij}$. In standard t-SNE, computing these probabilities requires evaluating the Gaussian kernel for all $N(N-1)$ pairs, which is prohibitively expensive. However, the paper observes that for dissimilar objects, the probability $p_{ij}$ is infinitesimally small and contributes negligibly to the gradient. Therefore, the authors propose constructing a **sparse approximation** of the input distribution $P$ by considering only the nearest neighbors of each point.

To efficiently find these neighbors without checking every pair, the algorithm constructs a **vantage-point tree** on the high-dimensional input data. A vantage-point tree is a metric space partitioning data structure where each node stores a specific data object (the "vantage point") and a radius defining a hypersphere centered on that object. During construction, all objects inside the hypersphere are assigned to the left child, and those outside are assigned to the right child; the radius is typically set to the median distance to ensure a balanced tree. This structure allows the algorithm to perform exact nearest-neighbor searches in $O(u N \log N)$ time, where $u$ is the perplexity parameter.

For each object $x_i$, the algorithm performs a depth-first search on the vantage-point tree to identify its set of nearest neighbors, denoted as $\mathcal{N}_i$. The size of this set is fixed to $\lfloor 3u \rfloor$, where $u$ is the user-defined perplexity (fixed at $u=50$ in all experiments). Once the neighbor sets are identified, the conditional probabilities $p_{j|i}$ are recomputed using only the neighbors in $\mathcal{N}_i$:

$$
p_{j|i} = \begin{cases} 
\frac{\exp(-d(x_i, x_j)^2 / 2\sigma_i^2)}{\sum_{k \in \mathcal{N}_i} \exp(-d(x_i, x_k)^2 / 2\sigma_i^2)} & \text{if } j \in \mathcal{N}_i \\
0 & \text{otherwise}
\end{cases}
$$

Here, $d(x_i, x_j)$ is the Euclidean distance, and $\sigma_i$ is the bandwidth determined via binary search to match the perplexity $u$. The joint probabilities are then symmetrized as $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$. By zeroing out non-neighbor entries, the resulting matrix $P$ contains only $O(uN)$ non-zero elements. Consequently, the sum of attractive forces, which iterates only over non-zero $p_{ij}$, can be computed in linear time $O(uN)$ rather than quadratic time.

#### The Barnes-Hut Approximation for Repulsive Forces
While the attractive forces are now efficient, the **repulsive forces** in the t-SNE gradient still theoretically involve all pairs of points because the Student-t kernel has heavy tails, meaning even distant points exert a small repulsive push. The gradient component for repulsive forces is given by:

$$
F_{\text{rep}} = - \sum_{j \neq i} q_{ij}^2 Z (y_i - y_j)
$$

where $q_{ij} = (1 + \|y_i - y_j\|^2)^{-1} / Z$ is the similarity in the embedding space, and $Z = \sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}$ is the normalization constant. Computing this naively is $O(N^2)$. The Barnes-Hut algorithm accelerates this by exploiting the geometric intuition that if a group of points is sufficiently far away from a target point $y_i$, they all exert roughly the same force direction and magnitude. Thus, the group can be treated as a single "summary" point located at their center of mass.

To implement this, the algorithm constructs a **quadtree** (for 2D embeddings) or an **octree** (for 3D embeddings) on the current positions of the points $E$ at every iteration. The root node represents the entire embedding space. Non-leaf nodes recursively split their spatial cell into four (or eight) equal-sized quadrants. Each node in the tree stores two critical aggregates:
1.  $N_{\text{cell}}$: The total number of points contained within that cell.
2.  $y_{\text{cell}}$: The center-of-mass of all points in that cell, calculated as the weighted average of their coordinates.

Constructing this tree takes $O(N \log N)$ time (or $O(N)$ with careful implementation), and the tree depth adapts to the data density, becoming deeper in crowded regions and shallower in sparse regions.

The algorithm approximates the repulsive force on a point $y_i$ by traversing the quadtree starting from the root. At each node, it evaluates the **Barnes-Hut opening condition** to decide whether to treat the node as a summary or to recurse into its children. The condition compares the size of the cell to its distance from the target point:

$$
\frac{r_{\text{cell}}}{\|y_i - y_{\text{cell}}\|^2} &lt; \theta
$$

Here, $r_{\text{cell}}$ is the width (or diagonal length) of the cell, $\|y_i - y_{\text{cell}}\|$ is the distance between the target point and the cell's center of mass, and $\theta$ is a user-defined threshold controlling the trade-off between speed and accuracy.
*   If the condition holds (the cell is small and far away), the algorithm approximates the force exerted by all $N_{\text{cell}}$ points in that node as a single force from $y_{\text{cell}}$ scaled by $N_{\text{cell}}$. The traversal stops for this branch (pruning the children).
*   If the condition fails (the cell is too large or too close), the algorithm recurses into the children to compute interactions more precisely.

Additionally, the normalization constant $Z$ is approximated during the same traversal by summing the contributions of the accepted cells. The final repulsive force is then computed as the ratio of the approximated unnormalized force to the approximated $Z$. This reduces the complexity of computing repulsive forces to $O(N \log N)$.

#### The Dual-Tree Approximation
The **dual-tree algorithm** attempts to improve upon Barnes-Hut by summarizing interactions between **two groups of points** simultaneously, rather than just one group and a single point. Instead of traversing one tree for a specific target point, the algorithm simultaneously traverses two identical quadtrees (Tree A and Tree B) in a depth-first manner.

For every pair of nodes (one from Tree A, one from Tree B), the algorithm checks a dual-tree opening condition:

$$
\frac{\max(r_{\text{cell-A}}, r_{\text{cell-B}})}{\|y_{\text{cell-A}} - y_{\text{cell-B}}\|^2} &lt; \theta
$$

If this condition is satisfied, the interaction between the two cells is deemed uniform enough to be summarized. The algorithm computes a single force vector based on the centers of mass $y_{\text{cell-A}}$ and $y_{\text{cell-B}}$. This force is then distributed to the points within the cells:
1.  The force is multiplied by $N_{\text{cell-B}}$ and added to the accumulated gradient for all points in Cell A.
2.  The force is multiplied by $N_{\text{cell-A}}$ and added to the accumulated gradient for all points in Cell B (due to Newton's third law, forces are equal and opposite).

Once a pair of nodes satisfies the condition, their children are pruned from the traversal, theoretically reducing the number of computations further than Barnes-Hut. However, the paper identifies a critical implementation bottleneck: determining **which specific points** belong to a cell to update their gradients is non-trivial. Unlike Barnes-Hut, where the target point is known explicitly, the dual-tree method must efficiently locate all leaf points under a pruned node to apply the accumulated force. Storing explicit lists of point indices for every node would require excessive memory ($O(N^2)$ in the worst case) or complex pointer structures, while reconstructing the list on the fly negates the speed gains. Consequently, despite the theoretical advantage, the overhead of this bookkeeping often makes the dual-tree variant slower or comparable to the simpler Barnes-Hut approach in practice.

#### Optimization Dynamics and Hyperparameters
The success of these approximations relies on specific tuning of the gradient descent procedure to ensure convergence despite the introduced errors. The authors adopt the optimization scheme from van der Maaten and Hinton (2008) with modified parameters suited for large datasets.

*   **Early Exaggeration:** To help the optimizer find a good global structure before focusing on local details, the input probabilities $p_{ij}$ are multiplied by a constant $\alpha$ during the first 250 iterations. While the original t-SNE used $\alpha=4$, this paper finds that for large datasets, a higher value of **$\alpha=12$** is necessary. This creates tighter initial clusters that can move more freely in the embedding space, preventing the formation of fragmented maps.
*   **Momentum Schedule:** The optimization uses momentum to accelerate convergence. The momentum weight is set to **0.5** for the first 250 iterations and increased to **0.8** thereafter.
*   **Step Size:** The initial learning rate (step size) is set to **200**. The step size is adapted dynamically during optimization using the scheme proposed by Jacobs (1988), which increases the step size if the gradient direction remains consistent and decreases it if the direction oscillates.
*   **Trade-off Parameter ($\theta$):** The parameter $\theta$ in the opening conditions controls the approximation accuracy.
    *   For **Barnes-Hut t-SNE**, experiments show that **$\theta = 0.5$** provides an excellent balance, yielding embeddings nearly identical to exact t-SNE (measured by 1-nearest neighbor error) while achieving massive speedups.
    *   For **dual-tree t-SNE**, a stricter threshold of **$\theta = 0.2$** is required to achieve comparable quality, likely because the cell-cell approximation introduces different error characteristics.
    *   Setting $\theta = 0$ disables the approximation entirely, reverting to the exact $O(N^2)$ algorithm.

The initialization of the embedding points $y_i$ is done by sampling from a Gaussian distribution with a very small variance ($10^{-4}$) to keep points clustered near the origin initially. The algorithm runs for a fixed **1,000 iterations**, which the authors found sufficient for convergence in most large-scale scenarios.

#### Design Choices and Rationale
The decision to use **vantage-point trees** for input similarities rather than other spatial trees (like kd-trees) is driven by the requirement to work with arbitrary metric spaces. Vantage-point trees only require a distance function $d(x_i, x_j)$, making the approach applicable even if the input data are not simple vectors (e.g., strings or graphs), provided a metric exists.

The choice to approximate **only the repulsive forces** with trees while keeping attractive forces exact (but sparse) is dictated by the nature of the t-SNE objective. The attractive forces depend on the high-dimensional input distances, which are static; sparsifying them via nearest neighbors is safe because $p_{ij}$ decays exponentially with distance. In contrast, the repulsive forces depend on the evolving low-dimensional embedding distances and the heavy-tailed Student-t kernel, which decays polynomially. This slow decay means distant points still matter, necessitating the sophisticated Barnes-Hut or dual-tree summation rather than simple truncation.

Finally, the paper explicitly rejects **Fast Multipole Methods (FMM)** despite their $O(N)$ theoretical complexity. FMM relies on expanding the interaction kernel into a series of orthogonal polynomials (e.g., Hermite polynomials for Gaussians). The t-SNE repulsive force is governed by the Student-t kernel $(1 + \|y_i - y_j\|^2)^{-1}$, for which no such efficient, accurate expansion exists. Adapting FMM would require replacing the Student-t kernel with a Gaussian, which would destroy the "crowding problem" solution that makes t-SNE effective (i.e., the ability to map distant high-dimensional points to far-apart low-dimensional points). Thus, the $O(N \log N)$ tree-based approach represents the optimal practical compromise between mathematical fidelity and computational efficiency.

## 4. Key Insights and Innovations

The primary contribution of this work is not merely the application of existing tree algorithms to a new domain, but the specific adaptation of these algorithms to handle the unique mathematical properties of the t-SNE objective function. The following insights distinguish this work from prior attempts at scaling dimensionality reduction.

### 4.1 Asymmetric Treatment of Attractive and Repulsive Forces
A fundamental innovation in this paper is the recognition that the two components of the t-SNE gradient require fundamentally different approximation strategies. Prior approaches to accelerating N-body problems often treated all interactions uniformly. However, van der Maaten observes that the **attractive forces** (governed by the input distribution $P$) and **repulsive forces** (governed by the embedding distribution $Q$) have distinct structural properties that allow for asymmetric optimization:

*   **Sparsification vs. Summarization:** The attractive forces rely on Gaussian kernels in the high-dimensional space, which decay exponentially. This allows the algorithm to safely **truncate** these forces by computing them only for nearest neighbors (using vantage-point trees), effectively setting distant interactions to zero without loss of fidelity. In contrast, the repulsive forces rely on the Student-t kernel, which has heavy tails and decays polynomially. Truncating these forces would destroy the global structure of the embedding. Instead, the paper introduces a **summarization** strategy (Barnes-Hut/Dual-Tree) that approximates distant repulsive forces rather than ignoring them.
*   **Significance:** This asymmetric approach is critical. If one were to simply sparsify all forces (as in some landmark methods), the "crowding problem" would re-emerge, causing distinct clusters to collapse into a single dense blob. By preserving the long-range repulsive interactions via tree summaries while sparsifying the short-range attractive interactions, the algorithm maintains the topological integrity of the t-SNE solution while achieving $O(N \log N)$ complexity.

### 4.2 The Counter-Intuitive Superiority of Barnes-Hut over Dual-Tree
Perhaps the most surprising empirical finding of this paper is that the **Barnes-Hut algorithm** consistently outperforms the **dual-tree algorithm** in the context of t-SNE, despite the latter being theoretically more advanced.

*   **The Bookkeeping Bottleneck:** In standard applications (like density estimation), dual-tree algorithms excel because they summarize interactions between two groups of points simultaneously, reducing the number of force calculations. However, t-SNE requires updating the position of *every individual point* based on the resultant force. As detailed in Section 4.3, when a dual-tree node pair satisfies the opening condition, the computed force must be distributed to every leaf node (data point) contained within those cells.
*   **Implementation Reality:** To do this efficiently, the algorithm would need to maintain explicit lists of point indices for every node in the tree, leading to prohibitive memory usage ($O(N^2)$ in worst-case scenarios) or expensive traversal overhead to locate the leaves. The Barnes-Hut algorithm avoids this by treating one side of the interaction as a single target point, eliminating the need to distribute forces to a group of targets.
*   **Significance:** This insight challenges the assumption that "more complex approximation equals better performance." It demonstrates that for iterative optimization tasks where gradients must be back-propagated to individual entities, the simpler point-cell interaction model of Barnes-Hut offers a superior speed-accuracy trade-off compared to the cell-cell model of dual-tree methods. Figure 3 and Figure 4 empirically validate that Barnes-Hut achieves lower nearest-neighbor error for the same computational cost.

### 4.3 Enabling Million-Scale Visualization via Early Exaggeration Scaling
While the tree algorithms provide the computational speedup, the paper identifies a crucial stability issue that arises when applying t-SNE to massive datasets: the difficulty of organizing global structure when $N$ is very large.

*   **The Scaling Problem:** In large datasets, the embedding space becomes crowded, making it difficult for clusters to separate and move into their correct global positions during the early stages of optimization. The standard "early exaggeration" technique (multiplying $p_{ij}$ by a constant $\alpha$) used in original t-SNE ($\alpha=4$) proves insufficient for $N > 100,000$.
*   **The Innovation:** The authors propose scaling the early exaggeration factor significantly, setting **$\alpha = 12$** for large-scale experiments (Section 5.2). This creates much tighter initial clusters with stronger attractive forces, allowing them to behave as rigid bodies that can easily slide past one another to find the correct global topology before the algorithm settles into local refinement.
*   **Significance:** This is a non-obvious hyperparameter adjustment that is essential for the practical success of the accelerated algorithms. Without this increase in $\alpha$, the $O(N \log N)$ speedup would be moot because the resulting embeddings would fail to capture the global manifold structure, yielding fragmented or tangled visualizations. This adjustment enables the successful embedding of the **TIMIT dataset (1.1 million points)** shown in Figure 7, a scale previously unattainable.

### 4.4 Practical Viability of Student-t Interactions Without Series Expansion
The paper solidifies the position of tree-based methods as the *only* viable path for accelerating t-SNE, explicitly ruling out Fast Multipole Methods (FMM) for this specific kernel.

*   **The Mathematical Barrier:** FMM achieves $O(N)$ complexity by expanding interaction kernels into series of orthogonal polynomials (e.g., Hermite polynomials for Gaussians). The authors note that no such efficient expansion exists for the **Student-t kernel** $(1 + \|y_i - y_j\|^2)^{-1}$. Approximating the Student-t kernel with a Gaussian to use FMM would fundamentally alter the algorithm's behavior, removing the heavy tails that prevent the crowding problem.
*   **Significance:** By demonstrating that Barnes-Hut approximations work effectively with the raw Student-t kernel (using the simple geometric opening condition in Equation 4), the paper validates that $O(N \log N)$ is the practical lower bound for exact t-SNE. It shifts the field's focus away from seeking $O(N)$ solutions via kernel approximation (which would compromise quality) toward optimizing $O(N \log N)$ tree traversals. This ensures that the visualizations produced retain the theoretical guarantees of the original t-SNE formulation.

## 5. Experimental Analysis

This section dissects the empirical evaluation conducted by van der Maaten to validate the theoretical claims of Barnes-Hut and dual-tree t-SNE. The experiments are designed not merely to show that the algorithms are faster, but to rigorously quantify the **trade-off between computational speed and embedding quality**, and to demonstrate scalability on datasets where standard t-SNE is computationally infeasible.

### 5.1 Evaluation Methodology: Datasets, Metrics, and Baselines

To ensure the results are robust across different data modalities and scales, the author evaluates the algorithms on **five distinct large-scale datasets**. The selection covers image recognition, object classification, and speech processing, with sizes ranging from tens of thousands to over a million points.

*   **MNIST:** $N=70,000$ handwritten digit images ($D=784$ pixels). This serves as the primary benchmark for tuning hyperparameters due to its manageable size and clear class structure.
*   **CIFAR-10:** $N=70,000$ tiny color images. Crucially, the input features are not raw pixels but **1,024-dimensional activations** from the final convolutional layer of a trained deep neural network. This tests the algorithm's ability to handle high-dimensional, non-linear feature spaces.
*   **NORB:** $N=48,600$ images of 3D toys under varying lighting and angles ($D=9,216$ pixels). This dataset is specifically chosen to test if the embedding can recover continuous **manifolds** (rotation and elevation) rather than just discrete clusters.
*   **Street View House Numbers (SVHN):** $N=630,420$ cropped house number images. Features are **64-dimensional activations** from a convolutional network. This dataset pushes the scale an order of magnitude beyond MNIST.
*   **TIMIT:** $N=1,105,455$ speech frames ($D=273$ MFCC features). This is the stress test, exceeding one million data points to prove the algorithm's capability to handle "big data" scenarios.

**Preprocessing and Setup:**
All datasets undergo Principal Component Analysis (PCA) to reduce dimensionality to **$D=50$** before t-SNE is applied. This is a standard practice to remove noise and accelerate the initial nearest-neighbor search. The experimental setup strictly follows the optimization protocol of van der Maaten and Hinton (2008) but with a critical modification for large $N$: the **early exaggeration factor** $\alpha$ is set to **12** (instead of the standard 4) for the first 250 iterations. As noted in Section 5.2, this higher value is essential for large datasets to prevent clusters from fragmenting during the initial global organization phase. The perplexity $u$ is fixed at **50** for all experiments.

**Metrics and Baselines:**
The evaluation relies on two primary metrics:
1.  **Computation Time:** Measured in seconds on a standard laptop (Intel Core i5 2.6GHz). This directly quantifies the scalability claim.
2.  **1-Nearest Neighbor (1-NN) Error:** This serves as the proxy for **embedding quality**. It measures the percentage of points in the 2D embedding whose nearest neighbor belongs to a different class than the original high-dimensional nearest neighbor. A low error indicates that the local structure (the most important aspect of t-SNE) is preserved.

The baselines for comparison are:
*   **Standard t-SNE:** The exact $O(N^2)$ implementation. This is only feasible for small subsets of the data or for theoretical reference ($\theta=0$).
*   **Barnes-Hut t-SNE:** The proposed point-cell approximation.
*   **Dual-Tree t-SNE:** The proposed cell-cell approximation.

### 5.2 Quantitative Results: The Speed-Accuracy Trade-off

The core of the experimental analysis lies in **Figure 3** and **Figure 4**, which systematically explore the impact of the trade-off parameter $\theta$ and the dataset size $N$.

**Experiment 1: Tuning the Trade-off Parameter ($\theta$)**
Figure 3 plots computation time and 1-NN error against $\theta$ for the full MNIST dataset ($N=70,000$).
*   **The Finding:** There is a clear "knee" in the curve where accuracy stabilizes while speed continues to improve.
    *   For **Barnes-Hut t-SNE**, setting **$\theta = 0.5$** yields an embedding with nearly identical 1-NN error to the theoretical exact solution, yet it completes in just **751 seconds** (~12.5 minutes).
    *   For **Dual-Tree t-SNE**, a stricter threshold of **$\theta = 0.2$** is required to achieve comparable accuracy. Even at this stricter setting, the dual-tree variant is slightly slower and yields a marginally higher error rate than Barnes-Hut at $\theta=0.5$.
*   **The Counter-Intuitive Result:** The graph explicitly shows that the red line (Dual-Tree) lies above and to the right of the green line (Barnes-Hut) in the speed-accuracy plane. This empirically validates the paper's claim that the additional bookkeeping overhead of the dual-tree approach negates its theoretical advantage in reducing force calculations.
*   **Baseline Context:** The point $\theta=0$ represents standard t-SNE. The author notes that running standard t-SNE on $N=70,000$ would take **many days**, making the ~12-minute runtime of Barnes-Hut a speedup of roughly two orders of magnitude.

**Experiment 2: Scaling with Dataset Size ($N$)**
Figure 4 illustrates how computation time scales as $N$ increases from small subsets up to 70,000.
*   **Complexity Verification:** The y-axis (time) is logarithmic.
    *   **Standard t-SNE (Blue):** Shows a steep linear slope on the log-linear plot, confirming **$O(N^2)$** quadratic growth. The curve shoots up vertically, becoming impractical beyond $N \approx 10,000$.
    *   **Barnes-Hut (Green) & Dual-Tree (Red):** Both exhibit a much flatter slope, consistent with **$O(N \log N)$** complexity.
*   **Magnitude of Speedup:** At $N=70,000$, the gap is massive. While standard t-SNE is effectively infinite (in practical terms), Barnes-Hut completes in minutes. The figure demonstrates that the relative benefit of the tree-based methods **increases** as $N$ grows; the larger the dataset, the more critical the approximation becomes.
*   **Quality Stability:** The right panel of Figure 4 shows that the 1-NN error for both tree-based methods remains flat and low across all values of $N$, indicating that the approximation does not degrade as the dataset scales up.

**Experiment 3: Large-Scale Visualizations and Timing**
The paper provides concrete timing results for the largest datasets in **Figures 5, 6, and 7**, demonstrating the practical utility of the method. All visualizations use Barnes-Hut t-SNE with $\theta=0.5$.

| Dataset | Size ($N$) | Computation Time | Key Observation |
| :--- | :--- | :--- | :--- |
| **MNIST** | 70,000 | **12m 31s** | Clear separation of all 10 digit classes without supervision. |
| **CIFAR-10** | 70,000 | **13m 20s** | Good separation of distinct classes (e.g., trucks vs. ships). An 11-NN classifier on the 2D embedding achieves a generalization error of **0.2467**, competitive with logistic regression on the original 1,024-D features. |
| **NORB** | 48,600 | **6m 30s** | Successfully recovers continuous **rotation manifolds**; points arrange themselves in circular structures corresponding to object azimuth. |
| **SVHN** | 630,420 | **2h 57m 15s** | Handles nearly 1 million points. Classes are well-separated, except for a central cluster of ambiguous/unrecognizable numbers, which correlates with errors made by the upstream CNN. |
| **TIMIT** | 1,105,455 | **3h 48m 12s** | The first successful t-SNE embedding of over **1 million** points. |

A critical insight from the TIMIT result (Figure 7) is the limitation of scatter plots at this scale. While the scatter plot looks like a uniform cloud, a **Parzen density estimate** (shown in the right panel of Figure 7) reveals that the data actually forms small, dense, class-specific clusters. This highlights that while the *computation* is solved, the *visualization* of millions of points requires density estimation techniques to be interpretable.

### 5.3 Critical Assessment and Robustness

**Do the experiments support the claims?**
Yes, convincingly. The experiments provide direct evidence for the three main claims:
1.  **Scalability:** The transition from days (estimated) to hours/minutes for $N > 100,000$ is undeniable. The TIMIT experiment ($N > 10^6$) is the definitive proof point.
2.  **Accuracy:** The use of 1-NN error as a metric is appropriate for t-SNE, which prioritizes local structure. The fact that the error rates for $\theta=0.5$ are indistinguishable from the theoretical exact solution (where computable) confirms that the approximation introduces negligible distortion.
3.  **Barnes-Hut Superiority:** The side-by-side comparison in Figure 3 is rigorous. It explicitly debunks the assumption that the more complex dual-tree algorithm must be better, attributing the failure to the "bookkeeping bottleneck" of distributing forces to leaf nodes.

**Ablation Studies and Sensitivity:**
The variation of $\theta$ in Figure 3 acts as an ablation study on the approximation strictness.
*   **Robustness:** The results show that the algorithm is robust to the choice of $\theta$ within a reasonable range. Values around $0.5$ (for Barnes-Hut) work consistently well across different datasets (MNIST, CIFAR, SVHN), suggesting that extensive per-dataset tuning is not required.
*   **Failure Modes:** The paper implicitly addresses failure modes through the SVHN analysis. The "central cluster" of misclassified images suggests that if the input features (from the CNN) are ambiguous, t-SNE correctly groups them together rather than forcing a separation. This is a feature, not a bug, but it highlights that the quality of the embedding is bounded by the quality of the input representation.

**Limitations and Conditions:**
*   **Dimensionality Constraint:** The experiments are restricted to 2D embeddings (using quadtrees). The paper notes in Section 6 that extending to higher dimensions requires octrees or higher-order trees, where the tree size grows exponentially with dimension ($2^d$). Thus, while the method solves the $N$ scalability issue, it remains practically limited to visualization dimensions ($d=2, 3$).
*   **No Formal Error Bounds:** The author acknowledges in Section 6 that unlike Fast Multipole Methods, Barnes-Hut provides no formal error bounds on the gradient approximation. The validation is purely empirical. While the empirical results are strong, this leaves a theoretical gap regarding worst-case scenarios where the geometric assumptions of the tree might fail (though such cases are rare in typical data distributions).

In summary, the experimental analysis is thorough and decisive. It moves beyond simple timing benchmarks to analyze the structural fidelity of the embeddings, successfully demonstrating that tree-based approximations make t-SNE a viable tool for modern, massive datasets without sacrificing the local structure preservation that defines the algorithm.

## 6. Limitations and Trade-offs

While the proposed tree-based algorithms successfully break the $O(N^2)$ barrier, enabling t-SNE to scale to millions of data points, the approach is not without significant constraints. The acceleration comes at the cost of theoretical guarantees, dimensional flexibility, and specific geometric assumptions. Understanding these limitations is crucial for correctly interpreting the resulting visualizations and knowing when the method might fail.

### 6.1 Absence of Formal Error Bounds
The most significant theoretical limitation of the Barnes-Hut variant is the **lack of formal error bounds** on the gradient approximation.
*   **The Issue:** Unlike Fast Multipole Methods (FMM), which provide rigorous mathematical bounds on the error introduced by series expansions (e.g., Warren and Salmon, 1993), the Barnes-Hut algorithm relies on a heuristic geometric condition (Equation 4). As noted in Section 6, the error introduced by approximating a cluster of points with their center of mass can technically be **unbounded** (Salmon and Warren, 1994). If a point lies very close to a large cell that barely fails the opening condition, the approximation error could be substantial.
*   **Why it matters:** In iterative optimization like t-SNE, errors in the gradient estimate at one step can propagate and accumulate, potentially leading the optimizer to a poor local minimum or causing instability.
*   **The Paper's Defense:** The author argues that this lack of bounds is acceptable in practice for two reasons:
    1.  **Non-Convexity:** The t-SNE objective function is inherently non-convex. Even with the exact gradient, there is no guarantee of finding the global minimum.
    2.  **Descent Direction:** Convergence to *a* local minimum is theoretically guaranteed as long as the angle between the approximate gradient and the true gradient remains less than 90 degrees (i.e., their inner product is positive). The empirical results (Figure 3 and Figure 4) suggest that for typical data distributions and $\theta=0.5$, this condition holds sufficiently well to produce high-quality embeddings.
*   **Open Question:** While empirical stability is demonstrated, there remains a theoretical gap. We do not know the worst-case data distribution that would cause the Barnes-Hut approximation to diverge or produce a topologically incorrect map.

### 6.2 The Dimensionality Curse of Space-Partitioning Trees
The accelerated algorithms are effectively restricted to **low-dimensional embeddings** (specifically $s=2$ or $s=3$).
*   **The Mechanism:** The efficiency of the Barnes-Hut and dual-tree algorithms depends on the construction of spatial trees (quadtrees for 2D, octrees for 3D). These trees partition the embedding space into hyper-rectangular cells.
*   **The Constraint:** The number of children per node in such a tree is $2^s$, where $s$ is the dimensionality of the embedding.
    *   For $s=2$ (quadtree), each node has 4 children.
    *   For $s=3$ (octree), each node has 8 children.
    *   For $s=10$, each node would have $2^{10} = 1,024$ children.
*   **The Consequence:** As the embedding dimensionality increases, the tree becomes exponentially larger and sparser. The "curse of dimensionality" implies that in high-dimensional spaces, the concept of "nearby" vs. "distant" clusters breaks down, and the tree traversal loses its pruning efficiency. The memory overhead to store the tree structure and the computational cost to traverse it would explode, negating the $O(N \log N)$ speedup.
*   **Practical Impact:** This limits the utility of the method strictly to **visualization**. It cannot be used if the goal is to learn a intermediate high-dimensional representation (e.g., reducing 1000-D data to 50-D) for downstream tasks, unless one replaces the quadtree with a different metric tree structure (like a vantage-point tree) that scales better with dimension, a modification the paper suggests but does not implement or evaluate.

### 6.3 Visualization Artifacts at Massive Scales
Even though the *computation* scales to millions of points, the *visualization* of such massive embeddings introduces new interpretability challenges that the algorithm itself does not solve.
*   **The Overplotting Problem:** In Figure 7 (TIMIT dataset, $N \approx 1.1$ million), the standard scatter plot appears as a diffuse, somewhat uniform cloud of points. To the naked eye, it is impossible to discern the fine-grained cluster structure or the density variations.
*   **The Misleading Uniformity:** The paper explicitly demonstrates that this visual uniformity is an illusion. The right panel of Figure 7 shows a **Parzen density estimate** of the same embedding, which reveals that the data is actually concentrated in small, dense, class-specific clusters separated by large empty spaces.
*   **Implication for Users:** Simply running Barnes-Hut t-SNE on a million points and plotting the result as a scatter plot is insufficient. Users must employ additional visualization techniques, such as **class-conditional density maps** (van Eck and Waltman, 2010) or density estimation, to correctly interpret the structure. The algorithm solves the computation, but the human perceptual bottleneck remains.

### 6.4 Sensitivity to Hyperparameters and Data Quality
The success of the acceleration relies on specific hyperparameter tuning and the quality of the input features, which introduces potential failure modes.
*   **Early Exaggeration Scaling:** The paper finds that the standard early exaggeration factor ($\alpha=4$) used in original t-SNE is insufficient for large datasets. For $N > 100,000$, the author must increase this to **$\alpha=12$** (Section 5.2).
    *   *Risk:* If a user applies the accelerated algorithm to a new massive dataset without adjusting $\alpha$, the optimization may fail to organize the global structure, resulting in fragmented clusters or a "hairball" visualization. This adds a layer of manual tuning that was less critical for smaller datasets.
*   **Dependence on Input Features:** The quality of the embedding is strictly bounded by the quality of the input representation.
    *   *Evidence:* In the SVHN experiment (Figure 6), the algorithm correctly groups a central cluster of images that the upstream Convolutional Neural Network (CNN) found difficult to classify.
    *   *Interpretation:* While this shows the algorithm is working correctly (preserving similarity), it highlights that t-SNE cannot "fix" bad features. If the input features (e.g., raw pixels or poor CNN activations) do not separate the classes, the accelerated t-SNE will faithfully visualize that failure. The speedup allows us to see these failures faster, but it does not improve the separability of the data itself.

### 6.5 The Dual-Tree Bookkeeping Bottleneck
Finally, the paper identifies a specific architectural limitation of the **dual-tree approach** in the context of gradient-based optimization.
*   **The Paradox:** Theoretically, the dual-tree algorithm should be faster than Barnes-Hut because it summarizes interactions between two groups of points (cell-cell) rather than a point and a group (point-cell).
*   **The Reality:** As detailed in Section 4.3 and confirmed in Figure 3, the dual-tree variant is often slower or offers a worse speed-accuracy trade-off.
*   **The Cause:** The bottleneck is **bookkeeping**. When a cell-cell interaction is summarized, the resulting force must be distributed to *every individual point* contained within those cells. Efficiently identifying and updating these points requires maintaining explicit lists of indices for every node or performing expensive traversals to find leaves.
*   **Conclusion:** This suggests that for iterative point-movement problems like t-SNE, the simpler Barnes-Hut model is architecturally superior. The dual-tree approach may remain viable for problems where only the *total* energy or density needs to be computed (where distribution to individual points is unnecessary), but it is ill-suited for updating individual coordinates in an embedding.

In summary, while Barnes-Hut t-SNE is a breakthrough for scalability, it trades theoretical error guarantees for speed, restricts users to 2D/3D visualization, requires careful hyperparameter tuning for large $N$, and demands advanced density-based visualization techniques to interpret million-point outputs.

## 7. Implications and Future Directions

The introduction of Barnes-Hut and dual-tree t-SNE fundamentally alters the landscape of high-dimensional data visualization, shifting the paradigm from "sampling to visualize" to "visualizing the whole." By reducing the computational complexity from $O(N^2)$ to $O(N \log N)$, this work removes the primary barrier that previously restricted t-SNE to datasets with only a few thousand points. The implications extend beyond mere speed; they enable new scientific workflows, redefine best practices for visual analytics, and open specific avenues for algorithmic refinement.

### 7.1 Transforming the Data Analysis Workflow
Prior to this work, analyzing massive datasets (e.g., millions of gene expressions or image patches) required analysts to rely on **landmark t-SNE** or random subsampling. This introduced a critical blind spot: rare but significant patterns (outliers, small distinct clusters, or long-tail distributions) were often lost because they were unlikely to be included in a small random sample.
*   **From Hypothesis Generation to Verification:** With the ability to embed millions of points in hours rather than days, t-SNE transitions from a tool for generating initial hypotheses on small subsets to a verification tool for global data structure. Researchers can now inspect the *entire* distribution of a dataset, ensuring that observed clusters are not artifacts of sampling bias.
*   **Democratization of Non-Linear Visualization:** The reduction in runtime (e.g., embedding 70,000 MNIST digits in ~12 minutes on a standard laptop, as shown in **Figure 3**) makes non-linear dimensionality reduction accessible on commodity hardware. This eliminates the need for specialized high-performance computing clusters for standard visualization tasks, lowering the barrier to entry for exploratory data analysis in fields like bioinformatics and digital humanities.

### 7.2 Enabled Follow-Up Research Directions
The success of tree-based approximations in t-SNE suggests several concrete paths for future algorithmic development:

*   **Extension to Other SNE Variants:** The paper explicitly notes that the tree-based framework is agnostic to the specific kernel used, provided the force calculation can be approximated. This opens the door to accelerating related algorithms such as **Neighborhood Retrieval Visualizer (NeRV)** (Venna et al., 2010) and **Elastic Embedding** (Carreira-Perpiñán, 2010). Future work could adapt the Barnes-Hut opening condition (Equation 4) to the specific objective functions of these methods, potentially unlocking $O(N \log N)$ performance for a whole family of stochastic neighbor embedding techniques.
*   **Hybrid Tree Structures for Higher Dimensions:** A key limitation identified in **Section 6** is the exponential growth of quadtree/octree nodes in embedding dimensions $s > 3$. Future research could replace the axis-aligned spatial trees with metric trees that scale better with dimensionality, such as **cover trees** (Beygelzimer et al., 2006) or **vantage-point trees** (Yianilos, 1993), for the embedding space itself. This would allow t-SNE to be used for intermediate dimensionality reduction (e.g., 1000-D $\to$ 50-D) for downstream machine learning tasks, not just 2D/3D visualization.
*   **Formal Error Bounds for Iterative Optimization:** While the paper empirically validates the stability of Barnes-Hut t-SNE, it acknowledges the lack of formal error bounds (**Section 6**). A promising theoretical direction is to derive bounds specifically for *iterative* gradient descent where errors propagate over time, rather than static N-body simulations. Adapting the stability results from Krylov subspace iterations (de Freitas et al., 2006) to the Barnes-Hut context could provide theoretical guarantees on convergence rates and maximum trajectory deviation.
*   **Dynamic Early Exaggeration Schedules:** The finding that large datasets require a higher early exaggeration factor ($\alpha=12$ vs. the standard $\alpha=4$) suggests that static hyperparameters are suboptimal. Future work could develop adaptive schedules where $\alpha$ decays dynamically based on the current cluster separation or the number of points $N$, automating the tuning process for varying dataset scales.

### 7.3 Practical Applications and Downstream Use Cases
The ability to process millions of objects enables specific high-impact applications that were previously infeasible:

*   **Single-Cell Genomics:** In biology, modern sequencing technologies generate expression profiles for hundreds of thousands to millions of individual cells. Barnes-Hut t-SNE allows researchers to visualize the entire cellular landscape of an organism or tissue in a single map, identifying rare cell types and continuous differentiation trajectories without subsampling. The paper cites early successes in mouse brain data (Ji, 2013), a field that has since exploded in scale.
*   **Large-Scale Image Repository Exploration:** For computer vision, the method enables the visualization of massive image repositories (like the **SVHN** dataset with 630,420 images or the **80 Million Tiny Images** dataset). As demonstrated in **Figure 6**, this helps diagnose failure modes of deep learning models by visually clustering misclassified examples (e.g., the ambiguous central cluster in SVHN), providing intuitive feedback for model improvement.
*   **Metagenomics and Microbiome Analysis:** The analysis of microbial communities involves sequencing millions of DNA fragments. The paper highlights applications in metagenomic data (Laczny et al., 2014), where visualizing the similarity of genetic sequences helps identify novel species and functional groups within complex environmental samples.
*   **Natural Language Processing:** With the rise of word embeddings, visualizing the semantic space of entire vocabularies (millions of words) becomes possible. The paper references word embedding visualizations (Cho et al., 2014), allowing linguists to explore semantic shifts and relationships across massive corpora.

### 7.4 Reproducibility and Integration Guidance
For practitioners looking to integrate these methods, the following guidelines ensure optimal results based on the paper's findings:

*   **When to Prefer Barnes-Hut t-SNE:**
    *   **Dataset Size:** Use this method whenever $N > 5,000$. For smaller datasets, exact t-SNE is fast enough and provides exact gradients. For $N > 10,000$, Barnes-Hut is essential.
    *   **Dimensionality:** Strictly limit the target embedding to **2D or 3D**. Do not use the standard quadtree/octree implementation for embeddings higher than 3D due to memory and traversal inefficiencies.
    *   **Algorithm Choice:** Prefer **Barnes-Hut** over dual-tree. Despite the latter's theoretical elegance, the empirical results in **Figure 3** and **Figure 4** consistently show Barnes-Hut offers a better speed-accuracy trade-off due to lower bookkeeping overhead.

*   **Critical Hyperparameter Adjustments:**
    *   **Early Exaggeration ($\alpha$):** If $N > 100,000$, increase the early exaggeration factor from the standard 4 to **12** (as done in **Section 5.2**). Failing to do so may result in fragmented clusters and poor global structure.
    *   **Trade-off Parameter ($\theta$):** Set $\theta = 0.5$ for Barnes-Hut t-SNE. This value provides near-exact accuracy with maximal speed. Lower values (e.g., 0.2) yield diminishing returns in accuracy for significant computational cost.
    *   **Perplexity ($u$):** While the paper fixes $u=50$, users should treat this as a local neighborhood size parameter. For extremely large datasets, slightly higher perplexities (e.g., 50–100) may be necessary to capture broader local structures, but the sparse approximation logic ($\lfloor 3u \rfloor$ neighbors) remains valid.

*   **Visualization Best Practices:**
    *   **Avoid Raw Scatter Plots for $N > 100,000$:** As shown in the TIMIT experiment (**Figure 7**), raw scatter plots of millions of points can be misleading, appearing as uniform clouds. Always supplement scatter plots with **density estimates** (e.g., Parzen windows) or class-conditional heatmaps to reveal the true underlying cluster structure.
    *   **Software Availability:** The authors provide open-source code (referenced in the Abstract and **Section 5**), which has become the de facto standard implementation in libraries like `scikit-learn` (as `method='barnes_hut'`). Practitioners should utilize these optimized implementations rather than attempting to code the tree traversal from scratch, given the subtle details in the opening condition and center-of-mass updates.

In conclusion, this work does not merely optimize an existing algorithm; it redefines the scale at which human intuition can be applied to data. By making the visualization of millions of high-dimensional objects computationally tractable, it empowers researchers to see the "forest" and the "trees" simultaneously, fostering deeper insights into the complex processes that generate modern big data.