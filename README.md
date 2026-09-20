# ZTF Candidate Pipeline

An astronomy pipeline for processing Zwicky Transient Facility (ZTF) image
cutouts, measuring candidate significance, building aligned image triplets, and
ranking transient candidates with the BRAAI real/bogus classifier.

## Pipeline

The end-to-end workflow is organized into four stages:

1. Index difference images and measure quadratic-centroid SNR.
2. Build science, reference, and difference-image triplets.
3. Score triplets with BRAAI and analyze pre-trigger/post-trigger photometry.
4. Apply host-galaxy and image-quality filters and write candidate results.

The SNR implementation also computes an empirical off-source significance using
the same peak-selection procedure as the source measurement.

## Repository Layout

```text
.
├── pipeline/
│   ├── april_candidate_pipeline.py       # Main pipeline
│   ├── april_candidate_pipeline_2.py     # Pipeline variant
│   ├── config*.yaml                      # Example configurations
│   ├── braai_batch.py                    # BRAAI inference helpers
│   └── ztf_downloads/                    # ZTF search, download, and SNR code
├── galaxies/
│   ├── galaxy_list.ipynb                 # Galaxy-catalog workflow
│   └── util/                             # Catalog utilities
├── pipeline/image_download.ipynb         # ZTF image-download workflow
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- Dependencies listed in `requirements.txt`
- ZTF image metadata and cutouts
- Galaxy catalog data used by the selected configuration
- BRAAI installed separately, with a compatible model file

The data products and BRAAI model weights are intentionally not stored in this
repository. Keep them in local directories and point the configuration file to
their locations.

## Installation

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy or adapt one of the YAML files in `pipeline/`. A configuration specifies
the difference-image, science-image, and reference-image roots, the galaxy
catalog, the BRAAI model, thresholds, and the output directory. For example:

```yaml
diff_root: "../ztf_diff_cutout_imgs_catalog2"
sci_root: "../ztf_sci_cutout_imgs"
ref_root: "../ztf_ref_cutout_imgs"
model_path: "../braai/models/braai_d6_m9.h5"
```

The paths are resolved for execution from the `pipeline/` directory.

## Running

```powershell
Set-Location pipeline
python april_candidate_pipeline.py --config config.yaml
```

For the GW190425 setup:

```powershell
python april_candidate_pipeline.py --config config_gw.yaml
```

Use `--resume` to reuse existing stage outputs or `--force` to recompute them.
Results are written below the configured `out_root` directory.

## Data and Licensing

Large FITS files, catalogs, trained model weights, generated outputs, local
environments, and obsolete experimental files are excluded from Git. Users
must obtain the relevant ZTF and catalog data from their original providers and
follow those providers' attribution and redistribution terms.

BRAAI is an external upstream project and is not included here. Its code and
model are subject to their own repository's license and terms.

The license for the pipeline code has not yet been selected.
