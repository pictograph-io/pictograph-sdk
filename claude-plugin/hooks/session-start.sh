#!/bin/sh
# Hands the API key entered when the plugin was enabled (the plugin's sensitive `api_key`
# option, which Claude Code passes to hooks as CLAUDE_PLUGIN_OPTION_API_KEY) to the session's
# shell commands as PICTOGRAPH_API_KEY, the variable the pictograph CLI and SDK read. It prints
# nothing: a SessionStart hook's output reaches the model.
#
# - No key entered, or no CLAUDE_ENV_FILE: nothing to do.
# - Only a value made of the characters a Pictograph key uses ([A-Za-z0-9_-]) is written, so
#   nothing in it can be read by the shell as code.
key="${CLAUDE_PLUGIN_OPTION_API_KEY:-}"
if [ -z "$key" ] || [ -z "${CLAUDE_ENV_FILE:-}" ]; then
  exit 0
fi
case "$key" in
  *[!A-Za-z0-9_-]*) exit 0 ;;
esac
printf 'export PICTOGRAPH_API_KEY=%s\n' "$key" >> "$CLAUDE_ENV_FILE"
exit 0
