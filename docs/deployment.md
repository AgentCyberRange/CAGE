# Docs Deployment

CAGE docs are built with VitePress from the `docs/` directory.

## Local Preview

```bash
cd docs
npm install
npm run dev
```

Build static output:

```bash
cd docs
npm run build
```

Preview the static build:

```bash
cd docs
npm run preview
```

## GitHub Actions

The workflow is:

```text
.github/workflows/docs.yml
```

It runs on changes to:

- `docs/**`;
- `examples/*/README.md`;
- root `README.md`;
- the workflow file itself.

For pull requests and private-repo pushes, the workflow builds the docs and
stops there. For public-repo pushes to `main`, it also uploads the Pages
artifact and deploys.

## Public URL

The public documentation site is currently published through a dedicated Pages
repository:

```text
https://agentcyberrange.github.io/cage-org.github.io/
```

That repository is:

```text
AgentCyberRange/cage-org.github.io
```

Because it is served as a project Pages site under that path, build the source
docs with the matching base before publishing there:

```bash
cd docs
DOCS_BASE=/cage-org.github.io/ npm run build
```

Then publish the contents of:

```text
docs/.vitepress/dist/
```

to the root of the `AgentCyberRange/cage-org.github.io` repository (Pages source
is its `main` branch, path `/`).

After the source repository (`AgentCyberRange/CAGE`) is public, the
`docs.yml` workflow can also deploy it as a project Pages site directly. The
expected URL for that mode is:

```text
https://agentcyberrange.github.io/CAGE/
```

The VitePress `base` is configured for that project-page path in CI when no
override is provided:

```ts
base: process.env.DOCS_BASE || (process.env.GITHUB_ACTIONS ? '/CAGE/' : '/')
```

Override it for any other static host:

```bash
DOCS_BASE=/docs/ npm run build
```

## Current Private-Repo Behavior

If the repository is private and the GitHub plan does not support Pages for
private repositories, GitHub returns:

```text
Your current plan does not support GitHub Pages for this repository.
```

In that state, keep the workflow in build-only mode. The current workflow does
that automatically by checking:

```yaml
github.repository_visibility == 'public'
```

When the repo is opened publicly, the same workflow will deploy without needing
to change the docs source.

## Manual Publishing Fallback

If GitHub Pages is not available, publish the static output elsewhere:

```bash
cd docs
npm ci
npm run build
```

The static site is emitted to:

```text
docs/.vitepress/dist/
```

Upload that directory to any static host.
