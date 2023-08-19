# GitHub Repository README

Welcome to our GitHub repository! This README provides a guide to the various sections, figures, and functionalities in our project.

## Deployment 
- **Docker Configuration:** Refer to the Dockerfile at `./deployment/Dockerfile`.
- **Python Dependencies:** All required Python packages are mentioned in `./deployment/requirements.txt`.

## **Figure 3**: Interpolation between Two Prompts
Transform the given prompt:
`beautiful mountain landscape, lake, snow, oil painting 8 k hd` 
to 
`a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic scale, insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert bierstadt, james gurney, brian froud,`.

- **Seed:** 824331
- **Execution:** Run using `python3 ./interpolation/embedding_interpolation.py`
- **Output:** Images can be found at `/output/beautiful mount_a beautiful an/`.

## Section 2.4: Prompt Datasets
- Prompt list can be found in `./metric_based_optimization/datasets`.
- `prompts.txt`: 
  - A selection of 150 prompts from the diffusiondb: [Link](https://huggingface.co/datasets/poloclub/diffusiondb)
  - Selected subsets: `large_random_100k`, `large_random_1k`
  - Used for evaluating metric optimization.
- `LAION-Aesthetic-V2-prompts.txt`: Prompts utilized for Figure 8.

## Section 3.1: Metric-Based Optimization
To begin, run the download script for aesthetic predictor weights.

- **Source Code:** Find the code in:
  - `./metric_based_optimization/utils/aesthetic_metric_generalization.py` (Figure 9)
  - `./metric_based_optimization/full_pipeline_descent.py` (Figures 7 & 8)
- **Result:** `/output/metric_generalization/highly detailed photoreal eldritch biomechani/`
- **Examples:** Refer to Figures 7, 8, and 9.

## Section 3.2: Iterative Human Feedback
- **User Interface:** The interface is showcased in Figure 4. Run with `python3 ./iterative_human_feedback/userinteraction.py`
- **Results:** See Figure 10.

## Figure 5: Images with Varying Seeds
- **Prompt:** `Single Color Ball`
- **Seeds:** 683395, 417016, 23916, 871288, 383124
- **Instructions:** To execute, run ...

## Figure 6: Traversing the Prompt Embedding Space
- **Execution:** Use `python3 ./seed_invariant_embeddings/prompt_embedding_space_traversal.py`
- **Output:** `/output/universal_embeddings/embedding_space_traversal.pdf`
- **Notes:** Interpolation values `[alpha]` and `[beta]` can be obtained by executing `python3 ./seed_invariant_embeddings/utils/universal_embeddings_slerp.py`. Check the final print statement for specific values.

## Figure 7: Optimizing Blurriness and Sharpness
- Execute the methods `increase_blurriness()` and `increase_sharpness()` found in `./metric_based_optimization/full_pipeline_descent.py`.
- **Output:** 
  - Blurriness: `./output/metric_optimization/Blurriness/`
  - Sharpness: `./output/metric_optimization/Sharpness/`

## Figure 8: Aesthetic Metric Optimization
- **Execution:** Run `increase_aesthetic_score()` in `./metric_based_optimization/full_pipeline_descent.py`
- **Output:** `./output/metric_optimization/LAION-Aesthetics V2/`

## Figure 9: Aesthetic Metrics across Different Seeds
- **Execution:** Use `python3 ./metric_based_optimization/aesthetic_metric_generalization.py`
- **Output:** `/output/metric_generalization/highly detailed photoreal eldritch biomechani/`

## Section 4.2: Advanced Iterative Human Feedback
- **Prompt Engineering UI:** Launch the user interface for the prompt engineering reference method with ...
- **Seed-Invariance Software:** Access the software for seed-invariance testing at ...
- **User Study Questionnaires:** View the questionnaires for our user study ...

## Figure 10: User Study Image Showcase
- **Resources:** Find images and prompt embeddings for this study in `./iterative_human_feedback/user_study`.

## Figure 11: Unguided Seed-Invariant Embedding Method
- **Execution:** Run using `python3 ./seed_invariant_embeddings/universal_embeddings.py`
- **Results:** Check `./output/universal_embeddings`.

Thank you for exploring our work. If you need additional details or clarifications, please refer to the mentioned sections or open an issue on this repository.
