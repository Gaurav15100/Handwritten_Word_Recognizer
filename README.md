# Handwritten Word Recognizer

Handwritten Word Recognizer is a Python project for recognizing handwritten words using CNN models and segmentation preprocessing. The repository includes preprocessing scripts, model code, and sample data organization.

Quick local steps
- Create a Python virtual environment and install dependencies.

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Publish to GitHub (recommended steps)
1. Initialize the local repository (already done if you see a `.git` folder).
2. Create a remote repository on GitHub (via website or `gh` CLI).
   - With `gh`: `gh repo create <OWNER>/<REPO> --public --source=. --remote=origin --push`
   - Or create via github.com and then: `git remote add origin https://github.com/<OWNER>/<REPO>.git` and `git push -u origin main`.

Notes
- Large data and model files are excluded via `.gitignore`. To share models, consider attaching them to releases or using a storage service.
