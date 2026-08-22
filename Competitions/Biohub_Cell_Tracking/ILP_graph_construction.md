"" problem context

In multi-object tracking (MOT) the objective is to reconstruct the trajectory of discrete objects moving through space across a temporal sequence of frames. The tracking problem is commonly formulated as a **combinatorial optimization problem** over a network graph.
**Integer linear programing (ILP, #ILP)** Provides a rigorous mathematical framework to find the globally *optimal set of trajectories*  by casting the selection of tracking links as a binary decision variables subject to a set of linear physical and topological constraints. The sketch below shows a voxel identified as a cell at time $t=t'$ and two candidate cells at time $t=t^{''}$, in this case, the problem consist in evaluating if the identified voxels as cells are actually cells and which temporal link corresponds between cell at time $t'$ and cells at time $t^{''}$ .

![[ILP_Idea.excalidraw]]
# Graph construction

The first step is to construct a Direct acycling graph (DAG) $$
G=(\nu, \varepsilon)$$

# Nodes 
Each node $v \in \mathcal{V}$  typically represents a detected object (or segment) at a specific time frame. We can Partition the vertex set by $$
\nu=\bigcup_{t=1}^{\top} \nu_t
$$
Where $\nu_t$ Is the set of all valid detections in frame t.

## Edges $\varepsilon$

Directed edges $(u ,v)$ represent potential temporal relationships or state transitions between detection $u$ at frame $t$ and detection at frame $t'$ (where $t'>t$) 

## mathematical formulation of variables and objective function

We assign a binary decision Variable $x_e \epsilon {0,1}$  to every edge $e$ $\epsilon$ $\varepsilon$ 
$$
X_e= \begin{cases}1 & \text { if edge is optimal } \\ 0 & \end{cases}
$$
similarly, decision variables can be defined for  nodes $y_\nu \in\{0,1\}$ to indicate whether a detection is included or treated as a false positive.

### Objective function
The goal is to find a vector $\mathbf{x}=\left[x_1, x_2, \ldots x_{|\varepsilon|}\right]^{\top}$ that **minimizes a global cost function**. The cost associated with each edge, $c_e$ is typically derived from spatial distance, feature appearance similarity, or negative log-likelihoods from a probabilistic model
$$
\min _x \sum_{e \in \varepsilon} c_e x_e
$$
# exploring public notebook from Pilkwang
The notebook can be found in https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery
The following is my interpretation of Pilkwang's solution.
## Overview and Architecture Pipeline
3D multi- object tracking problem is divided into two sub-problems
1. **Detection** ($\mathcal {L} _{det}$): locating objects spatially within down sampled 3D volumetric frames using a spatio-temporal U-Net
2. **Association/tracking** $\mathcal {L}_ {det}$: linking detected nodes across adjacent frames using a Transformer architecture.

**Important** : The raw neural outputs are converted into crisp coordinates (physical micrometers), filtered, linked via a **motion relinker** (incorporating Hungarian matching), Passed through deterministic **gap repair** algorithms, and smoothed via **line-fitting**.

**Note:** this solution incorporates post-processing steps which are motion relinker, gap repair, and line-fitting.

## Mathematical definition of Variables

- $V_ {t: t+w-1}$ : temporal window of 3D volumetric frames from time $t$ to $t+w-1$ 
- $h_\theta$ : The TemporaI UNet 3D neural network parameterized by Weights $\theta$ 
- $F_ {t} (r)$ : high dimensional feature vector extracted at spatial voxel coordinate $r= (x, y, z)$ in frame $t$ 
- $a_{t}(r)$ : the raw Unnormalizedd scalar logit Output representing the probability of a node existing at voxel $r$ 
- $P_ {t} (r)$ The sigmold probability map derived from $a_{t}(r)$ Via the logistic function $\sigma (r)$ 
- $y_{ t}(r)$ the ground-truth binary mask indicating true object centroids (1) versus background voxels (0)
- $w ^{ \pm}(b)$ Normalization weights balancing positive and negative classes across batch sample b
- $N_{(b)} ^+$ , $N_{b}^-$ the total count of positive (true) and negative (false) voxels in batch b
- $g_{\phi}$ bidirectional cross-attention transformer parameterIzed by weights $\phi$ that evaluates edge connectivity
- $\psi(r_{i},t)$  Extra spatial or geometry feature concatenated with the UNET representation for node i
- $l_{ i,j }$ The scalar edge logit representing the compatibility of a directed connection from node i ( at frame t) to node j ( at frame$t+1$) 
- $P_{i,j}$ the normallted edge probability compreted using a softmax function over all candidate parent sources for target node J
- $M_{i,j}$ A binary indicator mask restricting the edge loss calculation to rows/columns touching ground-truth annotations
- $\tau=0.985$ hard probability threshold applied to local maxima to isolate valid point detections at inference time
- $\hat{ r }_{i,t+1}$ dynamically extrapolated spatial position of node i en frame $t+1$ based on its prior velocity vector
## Training Objective: Detection Loss
The volumetric window is fed into A 3D espatio-temporal UNET. At frame $t$, the output feature map $F_{t}(r)$   is projected down to a scalar field via a 1×1x 1 convolution $$
a_t(r)=q_\theta\left(F_t(r)\right), P_t(r)=\sigma\left(a_t(r)\right)
$$
**Class Imbalancing** Because cells are rare events, the majority of voxels are categorized as False. To prevent class unbalancing and that the network learns to Predict only False, the Binary Cross-Entropy (BCE) loss is weighted per batch sample b $$
w_{(b)}^{+}=\frac{1}{N_{(b)}^{+}}, w_{(b)}^{-}=\frac{\alpha}{N_{(b)}^{-}} \quad(with \alpha=0.01)
$$
the global detection loss sums these weighted components across batch size B and voxels r
$$L_{det} =\dfrac{1}{B}\sum^{B}_{b=1}\sum _{r}W_{b}\left( r\right) \cdot BCE Logit\left( a_{b}\left( r\right) ,y_{b}\left( r\right) \right)$$

## Training Objective : Edge Loss
Once potential nodes have been identified via local maxima of $P_{t}$, the system construct edges between consecutive  frames $t$ and $t+1$
### Transformer Edge Scoring
For source node i at frame $t$ and target node j at frame $t+1$ , features and relative spatial displacements ( $r_j - r_ i$) are processed by the transformer $g_{\phi}$ 
$$li,j=g_{\phi }\left( \left[ F_{t}\left( r_{i}\right) ,\psi \left( r_{i},t\right) \right] ,\left[ F_{t+1}\left( r_{j}\right) ,\psi \left( r_{j},t+1\right) \right] ,r_{j}-r_{i}\right) $$
### Softmax Normalization
To enforce structural rules (such as allowing a parent cell to divide into two daughters while preventing many competing parents claim the same daughter ), the edge logits are normalized over all alternative incoming paths into target node j $$P_{i,j}=\dfrac{\exp \left( l_{i,J}\right) }{\sum _{k}\exp \left( l_{k,J}\right) }$$
Where $k$ are all other nodes at $t$ possibly linking to node j at $t+1$
### Masked Focal Edge Loss
Because annotations are sparse, loss calculations are restricted via the indicador mask $M_{i,j}$, act are only where ground-truth paths exist.$$M_{i,J}=\begin{cases}1,if\sum _{J}y_{i,J} >0 or\sum _{i}y_{i,j} >0\\
0, otherwise\end{cases}$$
The objective uses a focal - style weighting factor $\left( 1-p_{i,j}^{\ast }\right) ^{2}$ to down -weight easy background pairs and force the model to focus on hard tracking boundaries
$$L_{edge}=mean_{\left( i,J\right) :M_{i,j}=1}\left[ \left( 1-P_{i,j}^{\ast }\right) ^{2}\cdot BCE\left( P_{i,j},y_{i,J}\right) \right] $$
where $P_{i,j}^{\ast }= P_{i,j}$ if $y_{i,j}=1$, and $P_{i,j}^{\ast }=1-P_{i,j}$ Otherwise. 
The overall multi-task training loss combines both components with $\lambda_{det}=1$ $$L=L_{edge}+\lambda _{\det }L_{\det }$$
Where "edge" is related to linking nodes at two different timesteps and "det" is related to correctly identify a voxel as cell
## Submission Graph Construction and Inference Post -Processing
At test time, row neural outputs undergo a strictic multi-stage deterministic refinement workflow:

### 1. Thresholding and Metric Calibration
• **Detections**: extracted where local maxima exceed $\tau=0.985$ 
• **Physical Distance Conversion**: voxel index shifts are scaled into real-world micrometers using physical microscope calibration factors :$$d_{um}\left( i,J\right) =\sqrt{\left( 1.625\Delta z\right) ^{2}+\left( 0.40625\Delta y\right) ^{2}+\left( 0.40625\Delta x\right) ^{2}}$$
### 2. Motion Relinking and Hungarian Assignment
To correct minor missed links from the neural network, a velocity- extrapolated position is predicted :$$\widehat{r}_{i,t+1}=r_{i,t}+\lambda _{v}\left( r_{i,t}-r_{i,t-1}\right) ,\left( \lambda \nu =0.52\right) $$
* *interpretation* : the predicted position ($\widehat{r}_{i,t+1}$ ) is evaluated against the current position of the node. This helps to check consistency
A cost matrix combining physical motion distance, raw spatial features, and learned edge probabilities ($\beta=0.78$) is solved using the **Hungarian algorithm** $$C_{i,j} = d_{motion}+0.05d_{raw}(i,j)-\beta P_{i,j}^{learned}$$
• *interpretation:* The Hungarian algorithm solves a minimization problem. According to this, the first 2 terms in $c_{ i, J }$ reward nodes with a lower distance or displacement with respect to its position at the previous time step, while select nodes with a higher predicted probability
#### Notes about the Hungarian algorithm
This is a combinatorial optimization algorithm that solves **the linear assignment problem** in polynomial time $O(n^3 )$.
• **The problem:** Given a cost matrix $C$ of size $N \times M$  representing the cost of pairing every source node in $t$ to every target node in $t+1$,  find the optimal one-to-one matching that minimizes the global cost
• **context:** The cost matrix $C$ contains the cost for all possible links between nodes $i$ and $j$, the Hungarian algorithm globally solves the assignment problem finding the lower cost and assuring that no two tracks claim the same detection
### Deterministic Gap Repair
• **one frame gap repair** : If a track terminates at frame $t$ and another starts at $t+2$, a missing intermediate node is bridge If $d_{\mu m } (i, J) \leq 2g$ (where g=5.9 $\mu m$) 
• **Two-missing-Frame recovery** : Governed by strict radial bounds ( $R_2=9.7 \mu m$, $s_2=4.05  \mu m$, $\rho_2= 0.0032$ ) to stich together longer dropouts without creating false structures
### Line-fit smoothing:
To eliminate jitter along continuous paths, final spatial coordinates are adjusted via a local linear neighborhood regression parameter (w = 0.74 ) without altering the underlying graph topology
$$ r_i'= (1- w ) r_ i + w \cdot LineFit _{ N(I)} ( t_i)$$
#### Notes on LineFit
$r_i$ is the position vector of a voxel identified as a node. these detections can suffer from noise across consecutive frames. Line-fit smoothing takes those established graph nodes and shafts their spatial coordinate ( $r _i'$) to lie in a smooth geometric line.$N(i)$ is the local temporal neighborhood and represents a subsect of connected nodes along a single trajectory centered around node $i$ across a small temporal window ( e.g., nodes in the window frame $t+ K$ to $t-K$ ).  the LineFit is a localized linear regression calculated through the spatial coordinates of the nodes in neighborhood $N(i)$, evaluated specifically at the time $t_i$ of node $i$



