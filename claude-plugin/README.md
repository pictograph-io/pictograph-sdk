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
`PICTOGRAPH_API_KEY` or `pictograph login` instead; a key already in your environment always wins.

## What it allows

The skill pre-approves only `pictograph ...` commands and reading files. Anything else, including
its helper scripts and any file it writes, asks for your permission first.

Docs: [pictograph.io/docs/agents](https://pictograph.io/docs/agents). Support:
[GitHub issues](https://github.com/pictograph-io/pictograph-sdk/issues) or support@pictograph.io.
