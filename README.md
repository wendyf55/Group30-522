# Predicting Abalone Ages Based on Individual Characteristics - Regression Problem

## Project Summary

This project builds and compares regression models to predict the age of individual abalones from simple physical measurements. Using the Abalone dataset from the UCI Machine Learning Repository, we model the relationship between age (approximated from shell rings) and features such as sex, shell length, diameter, height, and several weight measurements.

We fit three supervised learning models in Python:

- **Baseline linear regression**
- **Random Forest regressor**
- **Support Vector Regressor (SVR) with an RBF kernel**

Our linear regression model explains about half of the variance in ring count (R² ≈ 0.44, RMSE ≈ 5.48), while the non-linear Random Forest and SVR models both perform better (R² ≈ 0.52, RMSE ≈ 4.76–4.70). Whole weight is a strong positive predictor of age, whereas shucked weight has a strong negative coefficient, suggesting collinearity among weight variables. Overall, these results indicate that flexible, non-linear models using carefully chosen features are better suited for predicting abalone age from physical measurements.

## Contributors

Group 30 in DSCI 522  

- Mehmet Imga  
- Wendy Frankel  
- Claudia Liauw  
- Serene Zha

## How to Run the Analysis

Follow these steps to set up the environment and reproduce our analysis.

### Step 1: Clone the repository

```bash
git clone https://github.com/wendyf55/Group30-522.git
cd Group30-522
```

### Step 2: Create and activate the conda environment

We provide an environment.yml file that pins the versions of Python and required packages.

```bash
conda env create -f environment.yaml
conda activate 522
```

### Step 3: Run the analysis

1. Launch Jupyter Lab or Jupyter Notebook from within the activated environment:

```bash
jupyter lab
```

2. Open the abalone_rings.ipynb notebook in the repository.

3. Run all cells from top to bottom.
This will:
- Fetch and load the Abalone dataset,
- Perform basic data cleaning and train–test splitting,
- Conduct exploratory data analysis (EDA),
- Fit and evaluate the linear regression, Random Forest, and SVR models,
- Produce summary tables and visualizations comparing model performance and feature importance.

## Dataset

### Abalone Dataset

- Source: UCI Machine Learning Repository
- Instances: 4,177 abalones
- Features: 8 predictor variables
- Target: Rings – number of shell rings

## Dependencies

All dependencies (with versions) are specified in environment.yaml. Key libraries include:
- pandas
- numpy
- scikit-learn
- altair
- ucimlrepo
- ipykernel

Install these via the conda environment described above to ensure a reproducible computational environment.

## Adding a new dependency

- Add the dependency to the environment.yml file on a new branch.

- Run conda-lock -k explicit --file environment.yml -p linux-64 to update the conda-linux-64.lock file.

- Re-build the Docker image locally to ensure it builds and runs properly.

- Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

- Update the docker-compose.yml file on your branch to use the new container image (make sure to update the tag specifically).

- Send a pull request to merge the changes into the main branch.

## Computational environment (Docker)

This project uses a reproducible Docker environment located at:

- **Docker Hub image:** `wfrankel55/group30-522`
- **Compose file:** `docker-compose.yml`
- **Build definition:** `Dockerfile` and `conda-linux-64.lock`

The Docker image is automatically built and pushed to Docker Hub by the
GitHub Actions workflow `.github/workflows/docker-publish.yml`. 

---

### How to start the environment (recommended: Docker Compose)

1. Install **Docker Desktop** (which includes Docker Compose).

2. Clone this repository:

   ```bash
   git clone https://github.com/wendyf55/Group30-522.git
   cd Group30-522
   ```

3. Start the container with Docker Compose:

```bash
docker compose up
```

This command:

- builds/pulls the image defined in docker-compose.yml

- starts a container called dockerlock

- maps port 8888 on your machine to 8888 inside the container
(so Jupyter will be available at http://localhost:8888)

4. To stop the container, press Ctrl + C in the terminal where
docker compose up is running, then clean up with:

```bash
docker compose down
``` 

## License

The code and analysis in this repository are licensed under the MIT License.
See LICENSE.md for the full license text.

## References

[1] Dua, D., & Graff, C. (2019). UCI Machine Learning Repository: Abalone Data Set. University of California, Irvine, School of Information and Computer Science. Retrieved from the UCI Machine Learning Repository.

[2] Nash, W. J., Sellers, T. L., Talbot, S. R., Cawthorn, A. J., & Ford, W. B. (1994). The population biology of abalone (Haliotis species) in Tasmania. I. Blacklip abalone (H. rubra) from the north coast and islands of Bass Strait. Sea Fisheries Division Technical Report No. 48.

[3] Guney, S., Kilinc, I., Hameed, A.A., Jamil, A. (2022). Abalone Age Prediction Using Machine Learning. In: Djeddi, C., Siddiqi, I., Jamil, A., Ali Hameed, A., Kucuk, İ. (eds) Pattern Recognition and Artificial Intelligence. MedPRAI 2021. Communications in Computer and Information Science, vol 1543. Springer, Cham. https://doi.org/10.1007/978-3-031-04112-9_25

[4] Python Core Team. Python: A dynamic, open source programming language. Python Software Foundation, 2019. Python version 2.7. URL: https://www.python.org/.