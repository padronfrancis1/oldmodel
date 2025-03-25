name: Canary ML Model CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Upgrade pip, setuptools, and wheel
      run: |
        python -m pip install --upgrade pip setuptools wheel

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run scoring script
      run: |
        python score.py
