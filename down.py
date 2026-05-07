try:
    import kaggle
except ImportError as exc:
    raise ImportError(
        "The 'kaggle' package is not installed. "
        "Install it with 'pip install kaggle' and configure your Kaggle API credentials."
    ) from exc

print("Downloading dataset...")
kaggle.api.dataset_download_cli('crarojas/smdg-19', unzip=True)
print("Download and extraction complete!")