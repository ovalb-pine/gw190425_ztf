# ZTF Candidate Pipeline

Python tools for searching Zwicky Transient Facility (ZTF) image metadata,
downloading science/reference/difference-image cutouts, measuring candidate SNR,
building aligned image triplets, and ranking candidates with the BRAAI
real/bogus classifier.

This repository contains the analysis code and small synthetic tests. The ZTF
cutouts, astronomy catalogs, trained model weights, and pipeline output folders
are local or generated data and are intentionally excluded from Git.

## What It Does

The main pipeline follows these stages:

1. Index local ZTF difference images.
2. Measure quadratic-centroid aperture SNR and empirical off-source significance.
3. Build science/reference/difference triplets around a fixed sky position.
4. Score triplets with BRAAI.
5. Check pre-trigger and post-trigger photometry.
6. Apply host-galaxy and image-quality filters.
7. Save candidate tables and diagnostic plots.

## Repository Layout

```text
.
├── pipeline/
│   ├── april_candidate_pipeline.py   # Main end-to-end pipeline
│   ├── config.yaml                    # Example ZTF22aabjpxh configuration
│   ├── config_gw.yaml                 # Example GW190425 configuration
│   ├── braai_batch.py                 # BRAAI model loading and inference
│   ├── scripts/                       # Small pipeline utilities
│   ├── tests/                         # Synthetic tests; no astronomy data needed
│   ├── image_download.ipynb            # ZTF download workflow notebook
│   ├── candidate_search.ipynb          # Candidate-search exploration
│   └── ztf_downloads/                 # ZTF search, download, and SNR code
├── galaxies/
│   └── galaxy_list.ipynb               # Galaxy-catalog workflow notebook
├── requirements.txt
└── README.md
```

## Requirements

- Windows, Linux, or macOS
- Python 3.10 or newer
- A working C/C++ build environment may be needed by some scientific packages
- Access to ZTF data if downloading new cutouts
- The BRAAI package and a compatible model file for stage 2 inference

BRAAI is an upstream GitHub project and is not included in this repository.
Install or clone it separately, and keep its model files outside this Git
repository.

The current `requirements.txt` records the working environment, including
Jupyter and TensorFlow. For a smaller deployment environment, split it into
runtime, test, and notebook requirements before publishing a release.

## Installation

From the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Input Data

Do not commit the following files to GitHub:

- ZTF FITS/FITS.FZ cutouts
- Large FITS catalogs and exported CSV catalogs
- BRAAI `.h5` model files
- Pipeline output directories, logs, progress files, and generated plots

Keep them in a separate data directory or use GitHub Releases, Git LFS, Zenodo,
or an institutional data store. Update the paths in a copied configuration file
to point to those local files. The example configurations currently assume the
data directories are adjacent to `pipeline/`:

```yaml
diff_root: "../ztf_diff_cutout_imgs_catalog2"
sci_root: "../ztf_sci_cutout_imgs"
ref_root: "../ztf_ref_cutout_imgs"
model_path: "../braai/models/braai_d6_m9.h5"
```

The exact data products and access conditions should be documented in a future
data README, including source survey, query dates, filters, and any catalog
licenses or acknowledgements.

## Run the Pipeline

Run from the `pipeline` directory so the relative paths in the example configs
resolve as intended:

```powershell
Set-Location pipeline
python april_candidate_pipeline.py --config config.yaml --force
```

For the GW190425 configuration:

```powershell
python april_candidate_pipeline.py --config config_gw.yaml --force
```

Use `--resume` or the configuration's `resume: true` setting to reuse completed
stage outputs. Generated tables and logs are written below the configured
`out_root` directory.

## Run Tests

From the repository root:

```powershell
python -m pytest pipeline/tests
```

The SNR test creates a synthetic FITS image in a temporary directory, so it does
not require the local ZTF dataset.

## Stage 1 Random-Spot Diagnostic

The repository includes a diagnostic that compares stage 1 SNR at the source
position with reproducible random noise positions and writes a ZScale PNG with
numbered crosshairs to `pipeline/ztf_downloads/`:

```powershell
python pipeline/ztf_downloads/test_stage1_random_spots.py --n-spots 20
```

To use a specific image and save a CSV of measurements:

```powershell
python pipeline/ztf_downloads/test_stage1_random_spots.py `
    path\to\difference_image.fits.fz `
    --n-spots 30 `
    --output pipeline\ztf_downloads\random_spot_snr.csv
```

## Publishing to GitHub

1. Review the `.gitignore` and confirm that FITS files, model weights, outputs,
   virtual environments, and bytecode are ignored.
2. Inspect what would be committed:

   ```powershell
   git status --short
   git check-ignore -v ztf_diff_cutout_imgs_catalog2\some_file.fits.fz
   ```

3. Remove accidental generated files from the index if any were previously
   staged. Do not delete the local files:

   ```powershell
   git restore --staged -- .
   ```

4. Add only the source and documentation:

   ```powershell
   git add .gitignore README.md requirements.txt pipeline galaxies
   git status
   ```

5. Commit and push to a new GitHub repository:

   ```powershell
   git commit -m "Prepare ZTF candidate pipeline for publication"
   git remote add origin https://github.com/USERNAME/REPOSITORY.git
   git push -u origin main
   ```

Replace `USERNAME/REPOSITORY` with the actual repository name. Never commit
credentials, private catalog exports, or data whose redistribution is not
permitted.

## Before Making the Repository Public

- Document the upstream BRAAI repository, version, and model download location.
- Add a top-level license for the new pipeline code.
- Add a citation file or publication reference if this work supports a paper.
- Document the data source, data-release versions, and reproducibility paths.
- Replace machine-specific paths and remove stale experimental scripts/configs.
- Add continuous integration for `python -m pytest pipeline/tests`.
- Consider splitting `requirements.txt` into `requirements-runtime.txt`,
  `requirements-test.txt`, and `requirements-notebooks.txt`.
- Use Git LFS or an external archive for large project-owned artifacts only after checking storage limits and
  redistribution rights.

## License

The project license has not yet been selected. Add a top-level `LICENSE` file
before publishing. BRAAI is an external upstream project and should retain its
own license in its own repository.