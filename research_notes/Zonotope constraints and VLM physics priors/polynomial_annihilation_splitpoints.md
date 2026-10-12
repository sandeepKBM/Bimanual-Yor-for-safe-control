# Polynomial annihilation and split-point detection for zonotope decomposition (geometry and safe sets), with VLM part proposals

Researched 2026-10-08, web only. Labels:
- **[VERIFIED]**: I read the primary source text myself (raw paper text, or the arXiv/journal abstract page or project page fetched directly).
- **[SUMMARY]**: the claim comes from search-engine snippets, listing pages or secondary summaries, and I did not check it against the full paper.

No DOIs or arXiv IDs were invented. Every ID below appeared in a fetched URL or a search-result URL.

## Q1. What is polynomial annihilation (PA), and how does it detect jump discontinuities from scattered data? (original papers and multi-D extensions)

### Takeaway
PA is a local, high-order, divided-difference edge detector for scalar functions sampled at irregular points (Archibald, Gelb and Yoon, SIAM J. Numer. Anal. 2005). It uses a stencil of m+1 nearby samples and weights that annihilate every polynomial of degree below m. The output is therefore O(h^m)-small where the function is smooth and close to the jump height [f] at a discontinuity. A minmod over several orders suppresses oscillations and helps tell real jumps from steep gradients. Later extensions detect kinks (jumps in the derivative or gradient) and handle higher dimensions, but in higher dimensions the method is applied as 1D detection along lines or tensor stencils, so its cost grows quickly with dimension.

### Cited Findings
- [SUMMARY] Original paper: R. Archibald, A. Gelb, J. Yoon, "Polynomial Fitting for Edge Detection in Irregularly Sampled Signals and Images," *SIAM J. Numer. Anal.* 43(1):259–279, 2005. The method rests on "a local polynomial annihilation technique," converges to zero away from discontinuities, is cheap, and is "entirely independent of any specific shape or complexity of boundaries." — [ORNL listing](https://www.ornl.gov/publication/polynomial-fitting-edge-detection-irregularly-sampled-signals-and-images); [Ewha listing](https://pure.ewha.ac.kr/en/publications/polynomial-fitting-for-edge-detection-in-irregularly-sampled-sign/)
- [SUMMARY] In the same paper, applying the minmod function across PA detectors of several orders "ensures a high rate of convergence away from the discontinuities while reducing the inherent oscillations near the discontinuities." It also lets the method distinguish jump discontinuities from steep gradients, "even in instances where only sparse nonuniform data is available." — [ORNL impact listing](https://impact.ornl.gov/en/publications/polynomial-fitting-for-edge-detection-in-irregularly-sampled-sign/)
- [VERIFIED] The exact operator, as restated by Glaubitz and Gelb:
  - Definition: L_m[u](ξ) = (1/q_m(ξ)) Σ_{x_j∈S_ξ} c_j(ξ) u(x_j). It approximates the jump [u](ξ) = u(ξ⁺) − u(ξ⁻).
  - Stencil: S_ξ is a set of m+1 local points.
  - Weights: the c_j solve Σ c_j(ξ) p_l(x_j) = p_l^(m)(ξ) for a polynomial basis {p_l}, l = 0..m. In closed form, c_j(ξ) = m! / ∏_{i≠j}(x_j − x_i), i.e. Newton divided-difference weights.
  - Normalisation: q_m(ξ) = Σ_{x_j∈S_ξ⁺} c_j(ξ), summed over the stencil points to the right of ξ (x_j ≥ ξ).
  - Accuracy: at a jump, L_m[u] = [u] + O(h). In smooth regions it is O(h^{min(m,k)}) for u ∈ C^k.
  - Stated weaknesses: oscillations near discontinuities grow as m increases, steep gradients can give false positives, and the order m must be chosen with care.
  — [Glaubitz & Gelb, arXiv:1903.03844](https://ar5iv.arxiv.org/html/1903.03844)
- [SUMMARY] Kink detection: Archibald, Gelb and Yoon, "Determining the locations and discontinuities in the derivatives of functions" (title confirmed on listing pages), extends PA to discontinuities in derivatives. — [ORNL listing](https://www.ornl.gov/publication/determining-locations-and-discontinuities-derivatives-functions); [Ewha listing](https://pure.ewha.ac.kr/en/publications/determining-the-locations-and-discontinuities-in-the-derivatives-/)
- [SUMMARY] A CICP paper, "A High Order Method for Determining the Edges in the Gradient of a Function," extends the PA detector "to locate discontinuities in the gradient given irregularly sampled point values of a continuous function." I did not check its author list. — [CICP / Global-Sci](https://ojs.global-sci.org/index.php/cicp/article/view/5629); [Global-Sci article page](https://global-sci.org/intro/article_detail/cicp/7758.html)
- [SUMMARY] Multivariate extension for stochastic simulations: "Discontinuity detection in multivariate space for stochastic simulations" (J. Comput. Phys. 2009; Archibald et al., per the PDF filename "Archibald_JCP2009"). — [ORNL listing](https://ornl.gov/publication/discontinuity-detection-multivariate-space-stochastic-simulations); [ORNL impact listing](https://impact.ornl.gov/en/publications/discontinuity-detection-in-multivariate-space-for-stochastic-simu/)
- [VERIFIED] Scaling limitation, stated by later authors: the PA approach "originally developed for one and two-dimensional edge detection, was extended to solve problems in higher dimensions. However, such methods rely on the evaluation of the target function based on a set of local tensor-product grids, so that the number of function evaluations grows exponentially as the dimension increases." — [Zhang, Webster, Gunzburger, Burkardt, hyper-spherical adaptive sparse-grid preprint (2015)](https://people.math.sc.edu/Burkardt/publications/zwgb_2015.pdf)
- [SUMMARY] A "polynomial annihilation transform" also exists for ℓ1 regularisation. It annihilates the function wherever it is locally a polynomial of degree below the order. The order-1 transform T_1 is equivalent to total variation (TV), but higher orders model smooth regions as low-degree polynomials instead of constants, which avoids TV staircasing. It was solved with Split Bregman for image reconstruction from undersampled Fourier data (Archibald, Gelb, Platte). — [ORNL listing](https://www.ornl.gov/publication/image-reconstruction-under-sampled-fourier-data-using-polynomial-annihilation-transform); [HOTV parameter-selection paper, arXiv:1608.04819](https://arxiv.org/pdf/1608.04819)
- [SUMMARY] A separate notion, *annihilation relations* and annihilating filters (finite rate of innovation), models a curve or surface as the zero-level set of a bandlimited function ψ, with points on the surface satisfying ψ(x) = 0. This has been used to recover point clouds on surfaces. Despite the shared word, it is not the Archibald–Gelb local PA operator. — [arXiv:1801.00886, "Recovery of Point Clouds on Surfaces: Application to Image Reconstruction"](https://arxiv.org/pdf/1801.00886)

### Inferences
- PA is in effect a high-order, irregular-grid finite difference, normalised so its output is the jump height. Order m = 1 reduces to a first difference (the TV-like case). The extra value of PA over a plain gradient threshold comes from (a) the multi-order minmod, which separates jumps from steep gradients, and (b) exactness on irregular samples.
- Applied to a derivative (the kink extensions), PA finds where linear pieces meet. That is exactly the split-point question for piecewise-linear or piecewise-affine approximation, so PA is a reasonable fit for the user's "find split points" goal in 1D or along lines.
- Higher-order divided differences amplify noise roughly like h^(−m). For noisy depth-camera points, only low orders (m ≈ 1–2) are likely to be usable unless the data is smoothed first. The PA papers frame this as the oscillation and steep-gradient issue rather than as a measured noise limit.

### Gaps
- I could not fetch the full 2005 SINUM paper (the sci.utah.edu PDF link for the 2009 JCP paper returned 404, and the ORNL PDF timed out). Its 2D/image construction (how stencils are chosen on scattered 2D points) is therefore described only at summary level.
- I found no published quantitative noise-robustness benchmark for PA on scattered data, for example detection rate versus SNR.

## Q2. Where has PA been used: UQ surrogates, classification boundaries, PDE shock capturing, imaging? Any use on reachable or safe-set boundaries, 3D point clouds or robotics?

### Takeaway
PA's main uses are: (1) uncertainty-quantification surrogates, to locate discontinuity hypersurfaces in parameter space (Archibald et al. 2009; Jakeman, Archibald and Xiu on adaptive sparse grids; Gorodetsky and Marzouk, where PA seeds the labels and an SVM refines the boundary); (2) troubled-cell sensors for shock capturing in discontinuous Galerkin solvers; and (3) imaging, as a sparsifying regulariser. I found **no** use of PA on 3D point clouds, on robot or object geometry, or on reachable or safe sets. The closest thing to a safe-set use is PA-seeded classification-boundary learning in UQ (Gorodetsky and Marzouk), which carries over almost directly to learning a safe/unsafe boundary from simulator samples.

### Cited Findings
- [SUMMARY] Jakeman, Archibald and Xiu, "Characterization of Discontinuities in High-Dimensional Stochastic Problems on Adaptive Sparse Grids" (J. Comput. Phys.), extends PA and relies on adaptive sparse-grid refinement. When the discontinuity lives in a small subset of the dimensions, the method is "optimal": the number of function evaluations grows linearly with dimension. — [ORNL listing](https://www.ornl.gov/publication/characterization-discontinuities-high-dimensional-stochastic-problems-adaptive-sparse); [ORNL impact listing](https://impact.ornl.gov/en/publications/characterization-of-discontinuities-in-high-dimensional-stochasti/)
- [VERIFIED] Gorodetsky and Marzouk, "Efficient Localization of Discontinuities in Complex Computational Simulations," SIAM J. Sci. Comput. 2014 (arXiv:1402.2845; the epubs URL carries 10.1137/140953137). The goal is to localise discontinuities in parameter space so the model output can be approximated as piecewise smooth. Two phases:
  - Initialisation: PA assigns function values to different regions and seeds an automated labelling procedure.
  - Refinement: active learning updates a kernel SVM representation of the separating surface.

  The method avoids structured grids, exploits simple boundary geometry, and was tested up to 11 dimensions (algebraic, ODE and PDE models), with better scaling than other localisation methods. — [arXiv abstract](https://arxiv.org/abs/1402.2845); [SIAM page](https://epubs.siam.org/doi/10.1137/140953137)
- [VERIFIED] The hyper-spherical adaptive sparse-grid (HS-AHSG) method models the discontinuity hypersurface directly. "Under a mild assumption about its geometry," the surface becomes a radius function of N−1 angles. Each radius value comes from a 1D discontinuity detection along a ray, which "many existing techniques," including PA or root-finding for characteristic functions, can solve. The paper motivates this by noting that plain adaptive sparse grids become dense near an N−1-dimensional discontinuity. — [Zhang et al. preprint](https://people.math.sc.edu/Burkardt/publications/zwgb_2015.pdf)
- [VERIFIED] Shock capturing: Glaubitz and Gelb build a discontinuous-Galerkin troubled-cell sensor S = S_3/S_1 from PA operators of orders 3 and 1, where S_m is the maximum of |L_m| over element points. Flagged elements switch on ℓ1 regularisation. — [arXiv:1903.03844](https://ar5iv.arxiv.org/html/1903.03844)
- [SUMMARY] A related search for PA-based shock capturing returned Glaubitz et al., "Smooth and compactly supported viscous sub-cell shock capturing for Discontinuous Galerkin methods." I did not check whether it uses a PA sensor. — [arXiv:1810.02152](https://arxiv.org/pdf/1810.02152)
- [SUMMARY] Imaging: the PA transform was used as an ℓ1 regulariser for image reconstruction from undersampled and noisy Fourier data. — [ORNL listing](https://www.ornl.gov/publication/image-reconstruction-under-sampled-fourier-data-using-polynomial-annihilation-transform); [ORNL listing ("sparsity of edges")](https://www.ornl.gov/publication/image-reconstruction-fourier-data-using-sparsity-edges-polynomial-annihilation)
- [SUMMARY] A targeted search ("polynomial annihilation" with point cloud, surface, robotics or classification boundary) found point-cloud boundary and edge papers based on ordinary polynomial curve fitting or FFT. The only "annihilation" hit was the separate annihilating-filter work (arXiv:1801.00886), not Archibald–Gelb PA. — [Strathclyde point-cloud boundary/edge paper](https://strathprints.strath.ac.uk/63217); [arXiv:1801.00886](https://arxiv.org/pdf/1801.00886)

### Inferences
- **Safe-set boundaries are where PA transfers most directly.** A safe set given by a black-box simulator or label (collision or no collision, spill or no spill) is an indicator function with a jump on the boundary. That is exactly the UQ discontinuity-detection setting, and the Gorodetsky–Marzouk recipe (PA labels, then SVM active learning) is a ready-made template for active boundary learning.
- **Constraint functions with kinks.** Many safety functions are piecewise smooth with kinks, for example h(q) = min over link pairs of d_ij(q) − d_safe, or an HJ value function. PA applied to ∇h along lines (the kink extensions) would locate *active-set switching surfaces*. Those surfaces are natural split points for piecewise-affine or zonotope approximations of the safe set, because on each side a single smooth constraint is active and linearisation error stays small. This is my inference; I found no paper that does it.
- In the dimensions that matter here (6–12 joint DoF per arm pair), raw tensor-stencil PA is too expensive (exponential in dimension, per Zhang et al.). PA would have to be used along rays or lines (hyper-spherical style), on sparse grids (Jakeman et al.), or only as a seed for an SVM or other classifier (Gorodetsky and Marzouk).

### Gaps
- I found no PA application to reachability analysis, Hamilton–Jacobi reachable sets, control-barrier-function boundaries or robotics. It may exist in grey literature that the searches missed, but none surfaced.
- I found no PA application to 3D point clouds or meshes.

## Q3. Other ways to find split points for piecewise-linear or convex decomposition (curvature and normal segmentation, CoACD concavity, PWA and hinging hyperplanes, zonotope splitting in reachability, zonotope fitting)

### Takeaway
The established split criteria come in four families:
- **Differential geometry:** normal-angle and curvature thresholds with region growing.
- **Convexity:** CoACD's collision-aware concavity metric plus plane cuts chosen by Monte Carlo tree search (MCTS). Newer variants are learned (feature-field convex decomposition, CVPR 2026) or interactive (Empart, with region-specific tolerances).
- **Function fitting:** piecewise-affine regression and hinging hyperplanes, where clustering or mixed-integer programming picks the regions.
- **Error-driven splitting in reachability:** Althoff splits a zonotope along the generator whose halves minimise a Lagrange-remainder performance index.

For zonotopes specifically, the correct split signal is how much a piece's zonotope over-approximation is reduced (concavity *and* lack of central symmetry), not edge strength.

### Cited Findings
**Curvature and normal-discontinuity segmentation**
- [SUMMARY] Region growing is a common point-cloud segmentation algorithm. It "greatly depends on the accuracy of points' normals and requires tuning two thresholds," namely the curvature increment and the normal angle. Normals usually come from local PCA, as the smallest-eigenvalue eigenvector of the k-NN covariance. — [IEEE Access 2023 (DOAJ)](https://doaj.org/article/656f4bcf130b4fb4a05b1e4f6e1083f2)
- [SUMMARY] Estimating normals, principal curvatures and sharp edges from a noisy point cloud is a standard geometry-processing problem, with Voronoi-based estimators among the robust options. I did not check the exact title or venue of this paper. — [Voronoi-based feature estimation paper (M. Ovsjanikov's page)](https://www.lix.polytechnique.fr/~maks/papers/voronoi.pdf)

**Approximate convex decomposition**
- [SUMMARY] CoACD (SIGGRAPH 2022) makes three changes to approximate convex decomposition:
  - a collision-aware concavity metric that measures the distance between a shape and its convex hull "from both the boundary and the interior";
  - direct mesh cutting with 3D planes, which avoids voxelisation errors and keeps the hulls intersection-free;
  - multi-step MCTS over cutting planes instead of a one-step greedy choice, which avoids unnecessary cuts.

  It ships as a C++ library and a pip package and has a recent "real metric" mode. — [CoACD GitHub](https://github.com/sarahweiii/CoACD); [arXiv:2205.02961 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2205.02961)
- [VERIFIED] "Learning Convex Decomposition via Feature Fields" (Yang, Huang, Uy, Sharp; CVPR 2026 oral; arXiv:2603.09285) is billed as the first feed-forward model for open-world convex decomposition:
  - It learns a continuous feature field, self-supervised from a convexity objective (contrastive "convex pairs," i.e. pairs whose connecting segment lies inside the shape).
  - Clustering the field gives the pieces, with adjustable thresholds controlling granularity.
  - It is purely geometric: no semantic or foundation-model prior.
  - It works on meshes, CAD models and Gaussian splats.
  — [arXiv abstract](https://arxiv.org/abs/2603.09285); [NVIDIA project page](https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/)
- [VERIFIED] Empart (Vu, Ganguly, Joshi; arXiv:2509.22847, Sept 2025) is an interactive convex-decomposition tool in which users "specify different simplification tolerances for selected regions of a mesh," calling existing decomposition algorithms as a subroutine. In robot pick-and-place it "reduced the overall simulation time by 69% compared to a uniform decomposition," and it needs fewer convex parts than V-HACD at a fixed error threshold. — [arXiv abstract](https://arxiv.org/abs/2509.22847)

**Learned primitives**
- [SUMMARY] SuperDec (arXiv:2504.00992) decomposes point clouds of arbitrary objects into compact sets of superquadrics for downstream uses including robot manipulation. — [arXiv:2504.00992](https://arxiv.org/pdf/2504.00992)

**Piecewise-affine (PWA) regression and hinging hyperplanes**
- [SUMMARY] Ferrari-Trecate et al. identify PWA hybrid systems by combining clustering in a feature space, linear identification and pattern recognition. This recovers both the affine submodels and the polyhedral partition. Most PWA regression methods (Bemporad et al. 2005; Ferrari-Trecate et al. 2003) work in two steps: cluster the data, then separate the clusters into polyhedral regions. Hinging hyperplanes (Breiman 1993) have been fitted to data with mixed-integer programming. — [Template-Based PWA Regression, arXiv:2305.08686](https://arxiv.org/pdf/2305.08686); [U. Pavia record](https://iris.unipv.it/handle/11571/139495)

**Zonotope splitting in reachability (Althoff)**
- [VERIFIED] Althoff, Stursberg, Buss, "Reachability Analysis of Nonlinear Systems with Uncertain Parameters using Conservative Linearization," IEEE CDC 2008. I read the raw text. The method:
  - Linearises locally and bounds the linearisation error with the Lagrange remainder, added as an uncertain input.
  - Computes, per time step, the set L of linearisation errors and an admissible set L̄. If L ⊄ L̄, it splits the initial reachable set and repeats.
  - Splits a zonotope by halving its j-th generator (Proposition 3). The two halves overlap in Z*, the zonotope with that generator removed. Higher-order zonotopes overlap more, so splitting them helps less, and the authors suggest reducing the order first. This trades overlap against over-approximation.
  - Chooses the generator with the lowest performance index ρ_j = max(L̂_{1,j}/L̄) · max(L̂_{2,j}/L̄), computed from the remainder bounds of the two candidate halves. Sets are split again recursively if needed.
  - Notes that splitting the state set is usually more effective than splitting input or parameter sets, because the linearisation error is mostly driven by state uncertainty. The number of sets grows with the number of splits.
  — [CDC 2008 PDF](https://skoge.folk.ntnu.no/prost/proceedings/cdc-2008/data/papers/0299.pdf); [TUM archive copy](https://archive.air.in.tum.de/Main/Publications/Althoff2008c.pdf)
- [SUMMARY] Linearisation-based reachability needs more splits when the abstraction becomes inaccurate, which can grow the number of sets exponentially. This motivated polynomial zonotopes as an alternative. Separately, a paper titled "Representation of Polytopes as Polynomial Zonotopes" exists (title only checked). — [Althoff 2008 TUM listing](https://portal.fis.tum.de/en/publications/reachability-analysis-of-nonlinear-systems-with-uncertain-paramet/); [arXiv:1910.07271](https://arxiv.org/pdf/1910.07271)

**Fitting zonotopes to geometry**
- [SUMMARY] Guibas, Nguyen and Zhang, "Zonotopes as bounding volumes" (2003), construct optimal or approximately optimal zonotopes around point sets or other geometry and test collisions directly from the generators, without building the polytope. — [Stanford PDF](https://geometry.stanford.edu/lgl_2024/papers/gnz-zbv-2003/gnz-zbv-2003.pdf)
- [SUMMARY] A search snippet describes over-approximating a V-polytope, such as a convex hull of data, by a zonotope through a linear program over user-chosen generator directions. I am not sure which paper the snippet came from. — [arXiv:2501.18426](https://arxiv.org/pdf/2501.18426)
- [SUMMARY] ARMTD uses zonotopes to over-approximate workspace obstacles and notes that any convex, bounded object can be over-approximated by a zonotope. — [ARMTD, arXiv:2002.01591](https://arxiv.org/pdf/2002.01591)

### Inferences
- **Edges and concavity point in different directions.** A box has sharp edges where PA or curvature detectors fire, yet it is a perfect zonotope that needs no split. A smooth bowl or torus has no edges, so detectors stay silent, yet it is strongly non-convex and must be split. An edge or kink detector on raw geometry therefore errs both ways. It is only useful if it is (a) filtered to *concave* creases (negative dihedral angle) or (b) applied to a convexity or fit-error signal instead of to the geometry itself.
- **Zonotopes add a symmetry requirement.** Zonotopes are centrally symmetric, so a convex but lopsided piece (a wedge, cone or triangle) still wastes volume. A good split score for zonotope decomposition should measure over-approximation directly, for example vol(Z_i) − vol(piece_i), or excess clearance in the directions the MPC constraint cares about. This is Althoff's ρ_j logic with "linearisation error" replaced by "fit conservatism," and it could reuse his choose-the-best-halving-generator machinery.
- **A clean way to use PA here: 1D profiles along candidate cut axes.** For each candidate cut direction n (CoACD-style plane cuts), compute a 1D signal along the axis s, such as cross-section area A(s), width w(s), hull defect d(s), or the zonotope-fit residual of the slab below s. Run low-order PA on the signal or its derivative to find kinks; these are candidate plane offsets. This turns plane search into cheap 1D kink detection, which is PA's best-understood regime (irregular samples, multi-order minmod against noise and steep gradients), and it could prune CoACD's MCTS candidates.
- **Splitting in state space.** Althoff's split is a halving in generator space, not an arbitrary plane, and the two halves overlap (Z*). For decomposing geometry, plane cuts give disjoint pieces whose individual zonotope fits may overlap; total conservatism, not piece count, should be the metric.

### Gaps
- I did not fetch CoACD's exact concavity formula (it is a Hausdorff-style distance combining boundary and interior), so it is given here at summary level only.
- I found no paper that uses zonotope-fit conservatism as the split criterion for decomposing object geometry. Guibas et al. fit zonotopes but do not decompose. That combination looks open.
- I did not survey adaptive splitting in neural-network verification (input-set partitioning), which is related.

## Q4. VLM and foundation-model 3D part segmentation that could name parts and propose split boundaries; work combining semantic proposals with geometric split criteria

### Takeaway
Zero- and few-shot 3D part segmenters now work on open-world shapes:
- PartSLIP (CVPR 2023) and PartSLIP++ (SAM plus EM-style lifting);
- SAMPart3D (2024: 2D-to-3D distillation, scale-conditioned granularity, multimodal LLM naming);
- PartField (ICCV 2025: a feed-forward feature field clustered into a part hierarchy).

They produce *semantic* or *perceptual* parts with fuzzy boundaries and no convexity guarantee. Convex-decomposition methods (CoACD, feature-field convex decomposition) are purely geometric. The pieces that would connect the two already exist: both newest methods are "feature field, then cluster," and Empart exposes per-region tolerances. I found no published method that refines semantic parts with a geometric split criterion to produce convex pieces or zonotopes.

### Cited Findings
- [SUMMARY] PartSLIP (CVPR 2023) does zero- and few-shot part segmentation of 3D point clouds, including part-level instance segmentation, using pretrained image-language models. — [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PartSLIP_Low-Shot_Part_Segmentation_for_3D_Point_Clouds_via_Pretrained_CVPR_2023_paper.pdf)
- [SUMMARY] PartSLIP++ uses SAM for pixel-wise 2D segmentations and replaces PartSLIP's heuristic 2D-to-3D conversion with a modified expectation-maximisation algorithm. — [arXiv:2312.03015](https://arxiv.org/pdf/2312.03015)
- [SUMMARY] SAMPart3D is a zero-shot framework that segments 3D objects into semantic parts at multiple granularities without predefined labels or text prompts. It uses 2D-to-3D feature distillation, scale-conditioned grouping, and multimodal LLMs to label parts, and it introduces the PartObjaverse-Tiny benchmark. — [arXiv:2411.07184](https://arxiv.org/pdf/2411.07184)
- [SUMMARY] PartField (NVIDIA / U. Toronto / Vector / UCSD; ICCV 2025; arXiv:2504.11451) is feed-forward. It learns a continuous part feature field, distilled contrastively from 2D and 3D part proposals, and clusters it into a hierarchical part decomposition with no templates or text names. It is reported as up to 20% more accurate and often orders of magnitude faster than recent class-agnostic part segmenters, and it supports co-segmentation and correspondence. — [arXiv:2504.11451](https://arxiv.org/pdf/2504.11451); [ICCV 2025 page](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_PartField_Learning_3D_Feature_Fields_for_Part_Segmentation_and_Beyond_ICCV_2025_paper.html); [project page](https://research.nvidia.com/labs/toronto-ai/partfield-release)
- [VERIFIED] The feature-field convex decomposition (CVPR 2026) shares PartField's "feature field, then cluster" design but uses a self-supervised convexity objective and has no semantic prior. — [NVIDIA project page](https://research.nvidia.com/labs/sil/projects/learning-convex-decomp/); [arXiv:2603.09285](https://arxiv.org/abs/2603.09285)
- [VERIFIED] Empart's per-region tolerance interface is human-driven; there is no VLM in the loop. — [arXiv:2509.22847](https://arxiv.org/abs/2509.22847)
- [SUMMARY] A 2019 learned convex-decomposition paper (CvxNet) argues that part-based models give a natural semantic decomposition useful for reasoning about extent, mass, contact and motion planning, while convex decomposition is the basis of real-time physics collision detection. — [CvxNet, arXiv:1909.05736](https://arxiv.org/pdf/1909.05736v2)
- [SUMMARY] VLMs are increasingly used to turn images and language into safety-constraint geometry. Examples: online updates of robot safety representations from natural-language feedback, contextual safety reasoning grounded into spatially defined safe sets, and AEGIS, a vision-language-safe-action layer that uses VLM-grounded control barrier functions (reported 77.85% average collision avoidance; I am not sure of the snippet's source page). — [arXiv:2409.14580](https://arxiv.org/pdf/2409.14580); [arXiv:2602.19983](https://arxiv.org/pdf/2602.19983); [alphaxiv 2512.11891](https://alphaxiv.org/overview/2512.11891v1)

### Inferences
- **What the VLM is good for.** It knows which parts matter functionally: the handle is graspable, the rim is spill-relevant, the body is the main collision volume. It can therefore set *where precision matters* (Empart-style per-region tolerance or piece budget) and give coarse region seeds.
- **What the VLM is bad at.** Boundaries come from multi-view 2D-to-3D lifting and are imprecise, and semantic parts are often non-convex (a mug body is a hollow cylinder). Each semantic part may still need several zonotopes, or a cavity-aware representation.
- **Natural division of labour.** The VLM or part field proposes regions and a per-region tolerance. A geometric detector (concave-crease filter, CoACD concavity, PA on 1D profiles or fit residuals) snaps split boundaries to geometric evidence within a band around each proposed boundary and subdivides inside parts until the zonotope conservatism is below that part's tolerance.
- Because PartField and the feature-field convex decomposition share an architecture, a joint field (a semantic part loss plus a convexity loss), clustered and then fitted with zonotopes, is a plausible learned version of the user's idea. I found no paper that does this.

### Gaps
- I did not check which image-language model each part segmenter uses internally, or their accuracy on real partial RGB-D scans (for example single-view ZED clouds) as opposed to complete synthetic meshes. Behaviour on partial views is an open practical question.
- I found no benchmark of part-segmentation boundary accuracy relevant to collision or zonotope conservatism.

## Q5. Is "VLM proposes parts + PA- or curvature-like detector refines split points + each piece becomes a zonotope" done anywhere? What is closest, and how good is the user's idea?

### Takeaway
I found no published work combining all three steps (VLM parts, PA or curvature refinement, per-piece zonotopes). The closest work covers the steps separately:
- **Semantic to safety sets:** VLM/LLM outputs become inflated obstacle sets plus a zonotope reachability gate (Dwedar et al. 2026), and language parse trees become composed constrained polynomial zonotopes (CompCPZ 2026). Neither decomposes object geometry.
- **Geometry to convex pieces:** CoACD, feature-field convex decomposition, and Empart (human-set per-region tolerances).
- **Zonotope fitting and splitting:** Guibas et al. 2003; Althoff's error-driven generator splitting.
- **PA:** UQ discontinuity localisation (Gorodetsky and Marzouk).

The idea looks new as a combination. However, PA's specific advantages (high order, irregular samples, telling jumps from steep gradients) pay off most for **safe-set or constraint boundaries in state or parameter space** and for **1D kink detection on profiles along candidate cuts**. On raw noisy 3D point clouds they add little over concavity metrics.

### Cited Findings
- [VERIFIED] "Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability Analysis" (Dwedar, Hafez, Jesser, Alanwar; arXiv:2609.27816). The VLM's semantic output "is mapped into conservative metric geometric constraints, including inflated obstacle sets, safe corridors, and goal regions," checked by a "robot-specific zonotope reachability gate." The abstract says nothing about decomposing objects into parts or choosing set splits. The fetched page gave a submission date of 18 Aug 2026, which conflicts with a 2609 arXiv ID (Sept 2026), so treat the date as uncertain. — [arXiv abstract](https://arxiv.org/abs/2609.27816)
- [VERIFIED] "CompCPZ: Preserving Multi-Modal Intent in Language-Guided Robot Manipulation" (Zhang, Hafez, Xie, Huang, Wu, Alanwar; arXiv:2608.17717, 18 Aug 2026) composes "per-primitive constrained polynomial zonotope enclosures along the language parse tree," with distribution-free conformal coverage. It keeps disjunctive intent ("near the red plate or the blue plate") as separate connected components. Its splits follow language structure, not geometry. — [arXiv abstract](https://arxiv.org/abs/2608.17717)
- [SUMMARY] "Safe LLM-Controlled Robots with Formal Guarantees via Reachability Analysis" uses set-based reachability with zonotopes and constrained zonotopes, chosen for cheap Minkowski sums. — [arXiv:2503.03911](https://arxiv.org/pdf/2503.03911)
- [VERIFIED] Empart is the closest geometry-side analogue of "semantics decides where to be precise": per-region tolerances, but set by a human. — [arXiv:2509.22847](https://arxiv.org/abs/2509.22847)
- [VERIFIED] Gorodetsky and Marzouk is the closest PA-side analogue of "detector seeds, then a learner refines a boundary." — [arXiv:1402.2845](https://arxiv.org/abs/1402.2845)
- [VERIFIED] Althoff's ρ_j generator-choice rule is the closest zonotope-side analogue of an error-driven split criterion. — [CDC 2008 PDF](https://skoge.folk.ntnu.no/prost/proceedings/cdc-2008/data/papers/0299.pdf)

### Inferences
Assessment of the idea "use PA-like methods to guide the VLM to produce better zonotopes by finding split points well":

1. **Sound core intuition.** Split points should sit where the object or function stops being well approximated by one simple piece. PA, especially the derivative and kink variants, is a principled, high-order, scattered-data detector of exactly "where one polynomial or affine piece ends." That matches piecewise-affine or zonotope approximation.
2. **Main pitfall: the wrong signal.** On raw geometry, kink detectors fire on the convex edges of shapes that are already zonotope-like and stay silent on smooth concavities. The detector must run on a *conservatism* signal: concavity or hull defect, zonotope-fit residual, or a concave-only dihedral angle. With that change, PA is just one way to find kinks in that signal, and the choice of signal matters more than the choice of detector.
3. **Where PA could clearly add value.**
   - (a) *1D profiles along candidate cut planes*, to propose plane offsets that prune CoACD-style search.
   - (b) *Safe-set and constraint boundaries in joint or state space.* Kinks in min-distance clearance functions or HJ value functions mark active-set switches, which are natural places to split a zonotope or PWA approximation of the safe set. Spill/no-spill indicator boundaries from simulation fit the Gorodetsky–Marzouk recipe (PA labels plus SVM active learning).
   - (c) *Black-box dynamics or learned models.* There Althoff's analytic Lagrange remainder is not available, and a sampled kink or discontinuity detector could choose which generator or direction to split.
4. **Where PA adds little.** On single-view, noisy ZED point clouds, high-order PA amplifies noise. Low-order PA is roughly a normal or curvature difference, which region growing, CoACD and learned convex decomposition already capture. Expect small gains there unless the cloud is first converted into a smooth implicit field or a 1D profile.
5. **VLM's role.** The VLM should set *where precision matters* (per-part tolerance or piece budget: tight near the gripper, rim and handle; loose on the far side) and provide coarse seeds. The geometric or PA layer should place exact boundaries and split recursively by conservatism. Letting the VLM output split points or zonotopes directly is unlikely to be precise. This is my inference; I found no measurements either way.
6. **A falsifiable test for the project.** At equal piece count K, compare zonotope decompositions from (a) CoACD hulls then zonotope fits, (b) VLM or PartField parts then fits, and (c) VLM parts plus PA/concavity refinement plus recursive conservatism splitting. Metrics: excess volume, false-positive collision or clearance violations in the FL-MPC arm–object constraints, and QP solve time. Run separately on safe-set boundary approximation (kink-guided versus uniform splitting of the clearance constraint).

### Gaps
- I read the 2026 VLM-to-zonotope papers (Dwedar et al., CompCPZ) only at abstract level. Their full texts might contain geometry-decomposition details.
- I found no work on PA-guided splitting in reachability or zonotope-based verification. Other researchers on this topic may want to check CORA documentation or the polynomial-zonotope literature for adaptive split heuristics that use sampled nonlinearity measures.
- I found no evidence on how robust VLM part boundaries are on real partial point clouds from a torso-mounted camera, which is what this project would use.
