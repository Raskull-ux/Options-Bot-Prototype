name: Feature lab

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  lab:
    runs-on: ubuntu-latest
    timeout-minutes: 150
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - name: Feature lab (discover on 2025, confirm on 2026)
        env:
          APCA_API_KEY_ID: ${{ secrets.APCA_API_KEY_ID }}
          APCA_API_SECRET_KEY: ${{ secrets.APCA_API_SECRET_KEY }}
        run: python backtest_feature_lab.py
      - name: Save results
        if: always()
        run: |
          git config user.name "options-bot"
          git config user.email "options-bot@users.noreply.github.com"
          git add data/feature_lab_rows.csv.gz data/feature_lab_screen.csv || true
          if git diff --cached --quiet; then echo "No changes."; else
            git commit -m "Feature lab: $(date -u +%Y-%m-%d)"; git pull --rebase --autostash; git push; fi
