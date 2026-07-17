# Kepler v0.1: a world encoder that learns physics across a robot's senses
*Ishneet Sukhvinder Singh, Alex Nguyen, Nicole Zhu, Linus Seah and Yip Jia Qi*
*[Read the paper](https://arxiv.org/html/2607.13522v1) · [Code](https://github.com/menloresearch/world-encoder/tree/user/jiaqi-paper-edits)*

Press your hand flat on a table, then hover it a millimeter above the surface. You can tell the difference instantly, and not because your eyes are sharp. You know what contact is because you have spent a lifetime perceiving the world through many senses at once: sight, touch, sound, and the felt position of your own body, each one shaped by the others.

For a robot, this trivially human distinction is nearly invisible. Most robotic perception today is built around vision as the primary input, with everything else the robot senses, its joint positions, its wrist's sense of force and touch, attached downstream as auxiliary signals.⁽1⁾ Asking such a system to understand contact is like asking you to feel a texture by looking at it, or to taste a dish by touching it: at best a vague approximation, at worst confidently wrong. Pressing and hovering look identical on camera, and much of physics, force, contact, weight, slip, never shows up in pixels at all.

> **[FIGURE 1]**
> *A robot's senses do not share a clock. Force/torque arrives above 100Hz, proprioception near 100Hz, vision at roughly 10Hz effective, and some robots lack entire sensors. These are the operating conditions Kepler is designed for. v0.1 addresses fusion and missing sensors; the temporal problem is the explicit next step.*

Our thesis is that robots fail in the real world because their own multi-modal data is treated as a side thought rather than a first-class input. We believe the place to fix that is the encoder: the layer that turns everything a robot senses into the representation the rest of its intelligence runs on. Get the encoder right, and every model built on top of it inherits a richer physical world.

Kepler is our research project toward a large physics model built to test this hypothesis, and today we are sharing its first piece: Kepler-Encoder v0.1, a world encoder, with a technical report. In it we present evidence for a property we call sensory entanglement: a robot's senses can be made to teach each other. When vision, body position, and force are trained together, each sense learning to predict the others, the vision pathway develops something it never had on its own: the ability to recover force and body state from pixels alone, at test time, with every other sense switched off. The encoder is fully self-supervised, fuses vision, proprioception, and force/torque into one shared representation, and is built for the conditions of real deployment, where embodiments vary, sensors differ, and data arrives noisy and asynchronous.

> **[FIGURE 2]**
> *The Kepler program. v0.1 is the encoder stage, trained on vision, proprioception, and force/torque; additional senses and the predictive models shown here are roadmap, not current results.*

> **[FIGURE 3]**
> *Physical intelligence as we operationalize it: information compression plus sensory entanglement.*

Kepler-Encoder v0.1 makes four claims.

1. **An architecture that entangles the senses and scales linearly.** A small set of learned queries compresses everything a robot senses into one fixed-size latent. Adding sensors grows cost linearly rather than quadratically, and the compression itself is what forces the senses to become entangled.
2. **Trained by cross-prediction, one sense learns to recover another.** Each sense is trained to predict the others. The result: from the camera alone, at test time, the encoder recovers force and body state that pixels by themselves do not contain.
3. **The encoder transfers across robot bodies.** One encoder spans four embodiments, matching per-robot specialists on their home robot and far exceeding them elsewhere, and the breadth comes from body diversity, not data volume.
4. **A safety monitor comes built in.** The encoder is measurably surprised when its senses disagree, which flags invalid robot state with zero additional training.

## Entangling the senses, scaling linearly

Everything the robot senses becomes a token in one shared vocabulary: patches from a frozen vision backbone, a token per joint and gripper, and a block for the wrist's force/torque and pose.⁽2⁾ A set of 8 learned queries attends over all of them, in the style of the Perceiver, and pools whatever arrives into a single 256-dimensional latent. Because the queries do the looking, cost grows linearly as sensors are added, not quadratically, and the output keeps the same size whatever the sensor suite. A robot missing a sensor simply masks the empty slots: the Franka, which has no force sensor, trains and evaluates like every other robot, so one architecture serves heterogeneous hardware with no per-robot variants and no retraining.

> **[FIGURE 4]**
> *From the paper (Figure 1): the cross-attention bottleneck at matrix level. All 217 sensor tokens are projected once to keys and values; the 8 learned queries score them as an 8 × 217 attention map instead of the 217 × 217 square that self-attention would fill, so cost grows linearly as senses are added. The output pools to one 256-d latent.*

The 8 learned queries, a matrix of 8 by 256 numbers, are the entire prior of the model: at every timestep, everything the robot senses is forced through them into one 256-dimensional vector. That bottleneck is where entanglement comes from, because the only way to compress three senses into one small code is to exploit their redundancy, letting what the camera sees stand in for part of what the joints report and the wrist feels. It is also a gamble. Compression this aggressive is lossy, and the latent might simply not keep enough. Whether real information survives the bottleneck is not something we assume; it is the question every experiment below is designed to test.

> **[FIGURE 5]**
> *The v0.1 encoder. Frozen ViT-B/16 patches, an 8-slot motor grid, and a 13-slot end-effector block are fused by 8 learned queries over roughly 217 tokens into one 256-dimensional latent.*

## One sense learns to predict another

Training is fully self-supervised and follows the JEPA recipe: hide part of the input and predict its representation rather than its raw values. Our move is what gets hidden. In image JEPAs the hidden part is a patch of the picture; here it is an entire sense. At each step one modality is masked, the rest are fused, and the model predicts the hidden modality's embedding under a slow-moving copy of the encoder, with the SIGReg regularizer preventing collapse. The objective asks each sense to predict the others, never to agree with them, which entangles the senses while preserving what makes each one different. We train on RH20T, a real-robot manipulation corpus of roughly 54 million frames across four embodiments, with roughly 30 percent held out so evaluation scenes are genuinely unseen.⁽3⁾

> **[FIGURE 6]**
> *From the paper (Figure 2): the training objective as a joint-embedding predictive architecture. Left: the generic JEPA template. Right: the Kepler instantiation, where the hidden view is an entire sense. Visible modalities are fused into z, and a per-modality head predicts the held-out sense's embedding under a slow EMA target; the held-out sense rotates across vision, motor, and end-effector. The loss is prediction error plus SIGReg.*

Does anything survive the bottleneck? Consider force, the sense we opened with. A single image simply does not contain it, and raw features from a strong pretrained vision model read contact force at an R² at or below 0.10, essentially nothing. We ran experiments to test whether training can change that. After training we froze the encoder, fed it only the camera image with every state input masked, and fit a linear probe, the simplest possible readout, from the latent to held-out robot state. If a linear map can read the state, the information is not just present but well organized, and the credit belongs to the representation.

> **[FIGURE 7]**
> *The protocol behind every number below. Training fuses all available senses; at evaluation the state inputs are masked, only the camera enters the frozen encoder, and a linear probe reads robot state out of the vision-only latent.*

On motor state, which the camera can largely see, the fused latent ties the strongest vision baselines: fusion adds no tax where vision is already sufficient. On end-effector state, where the camera is weakest, it leads every baseline on every sensored robot. Force is the sharpest case, moving from near zero to an R² of 0.05 to 0.19 depending on the robot. These absolute numbers are modest, and a single frame fundamentally bounds them, since much of what force means unfolds over time. The direction is the point: training vision against touch changes what pixels can tell you. The paper rules out the mundane explanations, in-domain training, dimensionality reduction, and data volume, with matched controls, and reports one negative result worth knowing before you build on this encoder: probes fit with all sensors present do not transfer to vision-only use, so readouts must be fitted for the sensor configuration they will see at deployment.

| feature (all 256-d) | motor R² | end-effector R² |
| --- | --- | --- |
| **Kepler-Encoder v0.1, vision-only latent** | 0.304 ± 0.019 | **0.282 ± 0.026** |
| compute-matched vision-only control | 0.198 ± 0.003 | 0.142 ± 0.003 |
| frozen ViT + finetuned linear head | 0.279 ± 0.008 | 0.206 ± 0.008 |
| frozen ViT, PCA-256 compression | **0.308 ± 0.010** | 0.234 ± 0.009 |

*Linear-probe R² on held-out scenes, averaged over the three sensored embodiments. For trained encoders the interval is the standard deviation over five training seeds; for deterministic baselines it is a cluster-bootstrap standard error. Motor is a statistical tie; end-effector is not.*

The latent also passes a generative check: a diffusion decoder reconstructs a recognizable camera frame from the 256 numbers alone, blurry by design, preserving scene and body configuration while discarding texture. The gamble of the bottleneck holds: aggressive compression, but the world-state survives. The decoder is a post-hoc probe only and plays no role in training.

> **[FIGURE 8]**
> *Camera frames decoded from the frozen 256-dimensional latent (top rows: real, bottom rows: decoded). Scene layout and body configuration survive the compression; texture does not, by design.*

## One encoder, four robots

Robot learning has an embodiment problem: models tuned to one body tend to fail on another. We test this directly by training specialist encoders on each robot and probing all models on all robots. The specialists behave as expected, strong at home and collapsing elsewhere. The unified encoder matches each specialist on its home robot and remains strong across the other three.

The obvious confound is data volume, since the unified model trains on the union of all four corpora. A budget-matched control settles it: given the same total number of frames, an encoder that spends them across diverse embodiments outperforms one that concentrates them on a single robot, when evaluated off that robot's home turf. Embodiment diversity, not frame count, buys the generality. We read this as early evidence for a position we hold more broadly: for physical intelligence, the binding constraint is not raw frame count but the diversity and sensory completeness of the data.

We want to be precise about the boundary of this claim. It holds for embodiments in the training mixture. Zero-shot transfer to a robot never seen in training currently trails the raw vision baseline, which tells us the encoder's body-awareness is learned per-embodiment rather than abstracted, at least at a single timestep. Closing that gap, including on our own hardware platforms, is an explicit target for the next version, and we report the failure now so progress against it is measurable.

## A safety monitor, for free

Because each sense is trained to predict the others, the encoder carries an intrinsic measure of cross-modal consistency: its own prediction error. When the robot's reported state does not match what the camera sees, that error spikes. With zero additional training, no labels and no finetuning, this signal cleanly separates valid from out-of-range robot states, and still flags harder scene-swapped cases, where a plausible state is paired with the wrong scene, though less strongly. The same frozen encoder also supports direct state recovery: a standard linear probe decodes joint angles from the vision-only latent across robots, shown alongside.

> **[FIGURE 9]**
> *Cross-modal surprise on the frozen encoder. Prediction error separates valid robot states from mismatched and out-of-range corruptions, with no additional training.*

> **[FIGURE 10]**
> *Joint angles decoded from the frozen vision-only latent by the standard ridge probe, predicted against actual on each robot's held-out scenes, all joints overlaid. Panel titles give each robot's overall motor R² (flexiv 0.27, ur5 0.34, kuka 0.36); the ur5 panel shows six joints, since a 6-DOF arm leaves the grid's seventh joint row unpopulated. The same frozen representation that flags anomalies also supports direct state recovery.*

Safety monitoring in deployed robotics is usually a separately engineered layer. Here, an anomaly detector falls out of the representation itself, at the encoder stage, before any policy acts. We treat that as a requirement rather than a bonus: when a language model harbors an undetected internal inconsistency, the failure is bad text, but when a robot does, the failure moves mass in the physical world. Safety should be a property of the representation, not only of the layers above it, and building it in at this stage is deliberate.

This is the operational payoff of sensory entanglement, the property we opened with: when each sense carries information about the others, disagreement between senses becomes visible in the representation itself. Alongside compression, learning what to keep and what to ignore, we regard sensory entanglement as one of the two capabilities a physical world model has to get right.

## Limitations and next steps

Everything above operates at a single timestep. The encoder represents no velocity, acceleration, or contact dynamics, which both bounds force recovery and excludes event detection, such as the moment a grasp slips. The learned structure is correlational, not causal: nothing in v0.1 predicts the consequences of actions. And as noted, embodiment generality is demonstrated within the training mixture, not beyond it.

These limitations define the roadmap. The next version targets native-rate temporal fusion: force/torque streams arrive at 100Hz or more while cameras deliver roughly 10Hz, and the standard fix of resampling to a common clock destroys exactly the high-frequency structure that makes force informative. Our bet is continuous-time embeddings that let each sensor contribute at its natural rate. On top of a temporal latent, we plan action-conditioned forward prediction, moving from correlation toward a model that can be used for control, and evaluation on additional hardware, including our own platforms. v0.1 trains entirely on a public corpus, which keeps every claim reproducible. The data the temporal version needs, high-rate force streams, contact events, and long-horizon recordings, largely does not exist in public corpora; it comes from instrumented hardware operating in the field, and supplying it is precisely why we build and deploy our own.

We are sharing this at v0.1, with its negative results and open problems stated, because we think the field moves faster when foundations are inspectable. The encoder, evaluation protocol, and controls are documented in the technical report, and we would genuinely value replication, criticism, and extension from the community. If entangled, compressed multimodal representations are the right substrate for physical intelligence, the evidence should survive contact with other labs' robots.

*[Read the technical report](https://arxiv.org/html/2607.13522v1) · [Discuss in the forum](https://forum.menlo.ai/)*

---

1. Recent vision-language-action systems such as OpenVLA, Pi-Zero, and GR00T N1 incorporate proprioception alongside vision and language, and the field is increasingly treating robot state as more than an auxiliary input. Visual embeddings nonetheless remain the primary perceptual representation in these systems.
2. The vision backbone is a frozen ViT-B/16; only the roughly 2M-parameter fusion module trains. Token counts: 196 vision patches, 8 motor slots (seven joints plus gripper), 13 end-effector slots covering force/torque and tool pose, roughly 217 tokens per timestep. We keep the backbone frozen because finetuning it on robot video degraded state recovery in our ablations.
3. 12,776 scenes across seven hardware configurations and four robot bodies (UR5, Franka, KUKA, Flexiv). The held-out set is split by task and operator group with near-duplicate filtering, so evaluation scenes are genuinely unseen.
