# Biohub - Cell Tracking During Development
"Detect and track zebrafish cells through 3D space and time"
This Kaggle competition can be found in https://www.kaggle.com/competitions/biohub-cell-tracking-during-development

## Goal
Develop algorithms to *detect, track and link cells across time in 3D microscopy data*, including accurate *identification of cell divisions and lineage reconstruction.*
The algorithms features are:
1. Use of real microscopy input datasets
2. Handle dense cell population
3. Handle noise
4. Handle complex biological structures

## Tasks
"Your task is to detect cells, associate them across frames, and identify division events to reconstruct accurate cell lineages"

## Evaluation
Information about the evaluation can be found in the [git of the competition](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md). The used metric for this competition is a combined tracking metric that **measures both:**
- edge accuracy: how well cells are linked across time
- division detection: how well cell mitosis events are identified.
The combined score is 
$$\text{score} = \text{adjusted\_edge\_jaccard} +0.1 \times \text{division\_jaccard} \tag{1}$$
#voxel
**Edge Jaccard:** Predicted nodes are matched to ground-truth nodes per timepoint via optimal bipartite assignment on scaled centroid distance (max 7.0 µm, physical scale z=1.625, y=x=0.40625 µm/voxel). A predicted edge is a true positive when both endpoints match ground-truth nodes connected by a ground-truth edge. The edge Jaccard is $$\frac{TP} {(TP + FP + FN)} \tag{2}$$adjusted by a penalty on over-predicting the total number of nodes. Where TP is true positive, FP is false positive, and FN is false negative. With the best value being 1 and worse value being 0.

==**Interpretation**:== Instead of evaluating pixel-by-pixel, this metric checks how accurately my model reconstructed the connections (edges) between objects (nodes) compared to the ground truth. When can understand this algorithm dividing it into 3 steps:
1. x,y, and z voxel distances are converted to physical distances. After that an optitmal algorithm (**Bipartite assigment**, e.g. Hungarian/Munkres) pairs each predicted node with the nearest ground-truth node using the defined *cutoff threshold*
2. Once the nodes are paired, the algorithm checks the connections between them. Defining:
	1. True Positive (TP): The predicted edge between node A and node B matches with the ground truth edge between nodes A' and B' (ground truth nodes)
	2. False Positive (FP): The model predicts an edge where none exists in ground truth or connects wrong nodes
	3. False Negative (FN): The ground truth has an edge but the model missed it.
3. Now that each node and edge have been identified and labeled (TP, FP, FN) the Eq 1 is used to compute the metric.

**Division Jaccard:** "A cell division is a node with two or more outgoing edges". The algorithm checks that the predicted graph has a connected component that covers pre-split stage and touches both daughter lineages, and it is compared with ground-truth.

==**Calculation of final score**== __"Per-sample adjusted edge Jaccards are weight-averaged by (TP + FP + FN); division Jaccards are micro-averaged across all samples."__ This line tells how final score are computed, it applies to different approach to each metric, i.e. edge and division jaccards.

1. Egde Jaccards: It consists of calculating edge Jaccard by using Eq. 2 and apply the *node over-prediction penalty* to get that sample's *Adjusted Edge Jaccard.*
2. Computation of the *weight-average*: For example, if you have two videos (samples) one with 100 edges and the other with 10.000, the second video should have 100 times more influence on the overall edge score because it represents 99% of the model data. For this reason, the *weighted-average* takes into account the total number of edges of each sample: $$\text{W}_i = \text{TP}_i + \text{FP}_i + \text{FN}_i \tag{3}$$ where $\text{W}_i$ is the weight. The final edge Jaccard is $$\text{Final Edge Jaccard} = \frac{\sum \text{Ajusted Edge Jaccard}_i \times \text{W}_i}{\sum \text{W}_i} \tag{4}$$
3. "division Jaccards are micro-averaged across all samples" *Micro-averaging* ignores sample boundaries during the final calculation. For this it computes TP, FP and FN across all sample videos and after that it computes one single Jaccard score at the very end. $$\text{Final Division Jaccard} = \frac{\text{Total TP}}{\text{Total TP}+\text{Total FP}+\text{Total FN}} \tag{5}$$ where $\text{Total X} = \sum X_i$ and $X$ can be TP, FP and FN. *Micro-averaging is done because* it is robust against 0 divisions and treats every single division event equally, regardless of which video it happened in. In the context of cell division, these events happen lest frequently so it is possible that one video presents only 1 ground-truth cell division, if the model doesn't predict this, it will gives a score $0/(0+0+1)= 0$, and if average vide-by-video is done (*macro-average*) that single missed division would drag the entire score as heavily as falling 50 divisions in a huge video.

**Note about _Division Jaccard_:** 
```
"
The exact timepoint at which a cell visibly splits is somewhat subjective, so the division metric uses a local window around each GT split:

grandparent → dividing parent → children → grandchildren

This window permits a predicted fork one timepoint before or after the GT split without using graph-wide reachability. Predicted nodes are matched independently against each GT division window with the same timepoint-aware, 7 µm optimal assignment used by the edge metric
"
```
*Interpretation:* If the ground-truth split is marked at $t=10$, but my model predicts the split at $t=9$, it is still counted as a TP, provided the local parent-to-daughter connectivity is intact.
**Note:** In graph theory, *reachability* checks if _any path_ connects node A to node B across the entire lineage tree (==even 20 timepoints later==)
- _Global reachability:_ A model could split a cell at time frame 5 (instead of frame 10), and as long as the descendants eventually match 20 frames (or any amount of frames) later, it might be a TP.
- _WITHOUT_ global reachability: The metric demands that the ==split happens locally in time==+
- Quick summary checklist for a valid division

| Condition  | Requirement                                                              |
| ---------- | ------------------------------------------------------------------------ |
| Structure  | The predicted node must split into at least two distinct branches        |
| Depth      | Each branch must extend to cover the child and its immediate descendants |
| Uniqueness | GT branch A and GT branch B must match two separate predicted branches   |
| Timing     | Both branch matches must fall within the allowed local time window       |

## Data

### Data Format
- `.zarr`directories
- All data is located at `0/`
- Data shape `(T,Z,Y,X)` where T is time and Z,Y,X are cartesian dimensions. Typically `(100,64,256,256)` 
- Chunks are one timepoint each `(1,64,256,256)` 
- The location is each chunk is `0/c/{t}/0/0/0`
- The physical voxel scale is z=1.625, y=0.40625, x=0.40625 µm/voxel.

# Model Baseline
```
End-to-end detection + linking, trained jointly:

1. **Detection**: a 3D U-Net with temporal attention (`TemporalUNet3D`) produces per-voxel features and a single-channel detection map; cell centres are recovered with local-max suppression.
   
2. **Linking**: per-node features from the U-Net are pooled at the detected centres and fed to a cross-attention transformer (`SimpleNodeTransformer`), which scores every (t, t+1) node pair.

3. **Sparse supervision**: only edges with ground truth are used for backpropagation — background detections and unannotated cells are ignored during training.
```

==**Interpretation**== 
**High-Level Concept: "End-to-end detection + linking, trained jointly":** the pipeline takes raw 3D video frames as input and directly outputs connected trajectories without requiring hardcoded, manual rules between stages. Additionally, the *detection module* and *linking module* ==update their weights together during training==.

**Detection stage:** _"a 3D U-Net with temporal attention (TemporalUNet3D) produces per-voxel features and a single-channel detection map; cell centres are recovered with local-max suppression_
- A Unet is an encoder-decoder neural network (NN) commonly used for image segmentation. Because microscopy video has 3D volumes, the *3D U-Net* processes these volumes. Additionally, our dataset has a time dimension, *temporal attention* helps the network focus on changes between frames.
- Per-voxel features: For* every voxel* in the 3D grid, the U-Net calculates a *rich vector of numbers* (features) describing it.
- Single-channel detection map: The *U-Net outputs* a *probability heatmap* of the *same 3D shape*, where higher values indicate a high likelihood that a cell center is present at that voxel.
- Local-max suppression: Probability heatmaps create "blobs" of high values near a cell.* Local-maximum suppression* suppresses all neighbor voxels around a peek, *leaving only the single highest voxel coordinate* to represent the exact center of each *detected cell*.

**Linking stage:** *"per-node features from the U-Net are pooled at the detected centres and fed to a cross-attention transformer (SimpleNodeTransformer), which scores every (t, t+1) node pair"*
- Per-node features pooled at detected centers: Each detected center becomes a "node". The model extracts (pools) the spatial feature vectors (created during the U-Net step) at those exact coordinate locations.
- Cross-attention transformer (`SimpleNodeTransformer`): A transformer NN uses *attention mechanisms* to *evaluate relationships between items*. Cross-attention lets the model compare the feature representation of a cell at frame t with candidate cells in frame t+1.
- Scoring every (t, t+1) node pair: For *every detected cell at time t*, the transformer calculates a *connection score against* *every candidate cell at time t+1*. A *high score* indicated a *high probability* that both detections belong to the *exact same physical cell* *or* they have a *parent-child relationship.*

**Training strategy: Sparse supervision:** *"only edges with ground truth are used for backpropagation — background detections and unannotated cells are ignored during training"*
- Ground truth edges: "Edges" refer to the tracked links connecting a cell at time t to its position at time t+1
- backpropagation on ground truth only: Fully annotating every single cell across thousands of 3D frames is practically impossible. *During training (backpropagation)*, the *loss function* only measures *error* on the *specific links that human experts explicitly labeled.*
- Ignoring unannotated cells: If the model detects a cell or link that has no label in the dataset, it does not penalize the model. This keeps incomplete human annotations from falsely penalizing the AI for finding real cells that the human annotator missed. **But** there is a correction term accounting for the fact that the model could try to predict more nodes, so if the amount of predicted nodes is larger than the amount of labeled nodes the loss metric in increased.



