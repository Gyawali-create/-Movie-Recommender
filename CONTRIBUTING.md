# Contributing

1. Start from `dev`: `git checkout dev && git pull`
2. Create a branch: `git checkout -b feature/<short-name>`
3. Commit small and often with the format `type: message` (`feat`, `fix`, `docs`, `refactor`, `test`, `chore`)
4. Run `pytest` before pushing
5. Open a Pull Request into `dev` and ask one teammate to review
6. After review, merge. `dev` → `test` → `main` happens only when tests pass

Rules: never commit `.env`, `__pycache__/`, `mlruns/`, `mlflow.db`, datasets or `model.pkl`.
