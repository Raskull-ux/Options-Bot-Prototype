name: Entitlements check

on:
  workflow_dispatch:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install requests
      - name: Run entitlements check
        env:
          APCA_API_KEY_ID: ${{ secrets.APCA_API_KEY_ID }}
          APCA_API_SECRET_KEY: ${{ secrets.APCA_API_SECRET_KEY }}
          FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
        run: python check_entitlements.py
