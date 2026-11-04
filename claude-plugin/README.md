# Pictograph for Claude Code

Real-time vision tools for agents and developers. This plugin teaches Claude Code the
[Pictograph](https://pictograph.io) CLI and Python SDK: upload and label image datasets, draft
labels with SAM3, train RF-DETR, YOLOX and Segformer models, deploy them as GPU endpoints,
stream live RTSP or WebRTC video to them, and run trained models as tools.

## Install

```text
/plugin install pictograph --marketplace pictograph-io/pictograph-sdk
```

The skill drives the `pictograph` CLI, so install it too:

```bash
pip install "pictograph[cli]"
```

When you enable the plugin, Claude Code asks for your Pictograph API key (app.pictograph.io,
Settings, API Keys) and keeps it in your system's credential store. A session hook sets it as
`PICTOGRAPH_API_KEY` for the commands Claude runs. Leave it empty to use your own
`PICTOGRAPH_API_KEY` or `pictograph login` instead.

## Where your data goes

The skill works through the `pictograph` CLI and SDK, which talk only to Pictograph:

- **Pictograph's API** (`api.pictograph.io`): the images and videos you upload, their
  annotations, and your requests to train, export, deploy and run models, authenticated with your
  API key. Model weights you download come back from Pictograph.
- **Your deployments and workflow endpoints** (`workflows.pictograph.io`): the images and video
  frames you send them for inference.
- **An import from V7 or Roboflow**: the source account's API key you give is sent to Pictograph's
  API, which reads the datasets from that provider.

An image URL you give a command is fetched on your machine. Webhook endpoints you register receive
events from Pictograph's servers, not from this plugin. The plugin and the CLI send no analytics or
telemetry. What you upload stays in your Pictograph workspace until you delete it; see the
[privacy policy](https://pictograph.io/privacy).

## What it allows

The skill pre-approves only `pictograph ...` commands and reading files. Anything else, including
its helper scripts and any file it writes, asks for your permission first.

Docs: [pictograph.io/docs/agents](https://pictograph.io/docs/agents). Support:
[GitHub issues](https://github.com/pictograph-io/pictograph-sdk/issues) or support@pictograph.io.
