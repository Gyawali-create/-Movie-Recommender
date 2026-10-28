# Branching and commit rules

| Branch | Purpose | Who merges into it |
|---|---|---|
| `main` | Final, stable version | from `test` only |
| `test` | Testing before release | from `dev` |
| `dev` | Daily integration | from `feature/*` by pull request |
| `feature/*` | One task each, e.g. `feature/data-cleaning` | one person |

Commit format: `type: short message`

| Type | Use for |
|---|---|
| feat | new feature |
| fix | bug fix |
| docs | documentation |
| refactor | code cleanup without behaviour change |
| test | tests |
| chore | setup, config, dependencies |
