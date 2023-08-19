# GitHub Repository README

---

## Deployment 
is documented in the Dockerfile located at ./deployment/Dockerfile
Python requirements are listed at ./deployment/requirements.txt

## **Figure 3**: Interpolation between Two Prompts

Transform the following prompt:

`beautiful mountain landscape, lake, snow, oil painting 8 k hd` -> 
`a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic scale, insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert bierstadt, james gurney, brian froud,`


- **Seed:** 824331

- **Execution:** To run, execute `python3 ./interpolation/embedding_interpolation.py`

- **Output:** Created images are located in /output/beautiful mount_a beautiful an/

---

## Section 2.4: Prompt Datasets
- 
- list of prompts located in ./metric_based_optimization/datasets 
- prompts.txt: 
  - list of 150 randomly selected prompts from the diffusiondb (https://huggingface.co/datasets/poloclub/diffusiondb)
  - selected subsets: large_random_100k, large_random_1k
  - utilized to evaluate the metric optimization
- LAION-Aesthetic-V2-prompts.txt
  - prompts used to create Figure 8

---

## Section 3.1: Metric-Based Optimization
Please run download script for aesthetic predictor weights: *.sh


- **Source Code:** Code for this section is contained in ...
- ./metric_based_optimization/utils/aesthetic_metric_generalization.py (Figure 9)
- ./metric_based_optimization/full_pipeline_descent.py (Figure 7, 8)
- Result: /output/metric_generalization/highly detailed photoreal eldritch biomechani/


- **Examples:** See below (Figures 7, 8, 9) for execution examples.

---

## Section 3.2: Iterative Human Feedback

Here, we explore the iterative human feedback mechanisms.

- **User Interface:** The UI, as illustrated in Figure 4, can be executed using `python3 ./iterative_human_feedback/userinteraction.py`
- **Results:** See below (Figure 10) for the results.

---

## Figure 5: Images generated with different random seeds

**Prompt:** `Single Color Ball`
**Seeds:** 683395, 417016, 23916, 871288, 383124

**Instructions:** To run, execute ...

---

## Figure 6: Traversing the prompt embedding space
- **Execution:** To run, execute `python3 ./seed_invariant_embeddings/prompt_embedding_space_traversal.py`
- **Output:** /output/universal_embeddings/embedding_space_traversal.pdf
- **Remark:** The values list in ./seed_invariant_embeddings/prompt_embedding_space_traversal.py corresponds to the interpolation values \[ \alpha \] and
\[ \beta \], which can be obained by running `python3 ./seed_invariant_embeddings/utils/universal_embeddings_slerp.py`. The pythonlist values can than be found in the last print statement


---

## Figure 7: Optimizing blurriness and sharpness
- In order to óptimize the blurriness/sharpness metric the methods increase_blurriness() and increase_sharpness() must be executed.
- Therefore select the methods in main and execute: ./metric_based_optimization/full_pipeline_descent.py
- Output: 
  - Blurriness: ./output/metric_optimization/Blurriness/
  - Sharpness : ./output/metric_optimization/Sharpness/

---

## Figure 8: Optimizing the aesthetics metric
- For the optimization of the aesthetics metric the increase_aesthetic_score() method has to be executed located within ./metric_based_optimization/full_pipeline_descent.py
- Output: ./output/metric_optimization/LAION-Aesthetics V2/

---

## Figure 9: Aesthetic metric for different seeds
- **Execution:** To run, execute `python3 ./metric_based_optimization/aesthetic_metric_generalization.py`
- **Output:** /output/metric_generalization/highly detailed photoreal eldritch biomechani/

---

## Section 4.2: Iterative Human Feedback

This section delves deeper into the iterative human feedback mechanisms.

- **Prompt Engineering UI:** The user interface for the prompt engineering reference baseline method can be deployed using...
- **Seed-Invariance Software:** The software to test the seed-invariance of the generated images is available at ...
- **User Study Questionnaires:** The questionnaires used in our user study can be accessed ...

---

## Figure 10: Images created in our user study

**Resources:** Images and prompt embeddings related to this study can be found in ./iterative_human_feedback/user_study

---

## Figure 11: Unguided seed-invariant prompt embedding method

**Instructions:** To run, execute `python3 ./seed_invariant_embeddings/universal_embeddings.py`
**Results:** All results can be located in ./output/universal_embeddings

---

We appreciate your interest in our work. For further information or any clarifications, please check the respective sections or feel free to raise an issue on this repository.

