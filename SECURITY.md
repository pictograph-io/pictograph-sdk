# Security policy

## Reporting a vulnerability

**Do not open a public issue.**

Use GitHub's private vulnerability reporting - the **Security** tab, then
**Report a vulnerability**. That opens an advisory only the maintainers can see.
If it is unavailable to you, email **support@pictograph.io** with `SECURITY` in
the subject line.

Useful reports include:

- what the vulnerability is and what an attacker gains,
- the SDK version and Python version,
- steps to reproduce, ideally a minimal proof of concept.

We aim to acknowledge within a few business days and will keep you posted through
remediation. We are glad to credit you in the advisory unless you would rather stay
anonymous.

**Redact your API key** before sending anything. A key pasted into a report should
be rotated in Settings regardless of who saw it.

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/pictograph/) receives
fixes. Upgrade before reporting.

## Scope

This policy covers the SDK in this repository. Report an issue in the hosted API or
the web app through the same private channel and we will route it.

## What this SDK does with your key

Worth knowing when you assess a report:

- The key is read from `PICTOGRAPH_API_KEY` or `~/.pictograph/config.toml`, which
  the CLI writes with `0600` permissions. It is never written to the repository.
- It is sent only to the configured host. A server-supplied redirect or URL cannot
  move it to another host, and redirects are not followed by default.
- It is kept out of exception messages, `repr()` output and CLI error rendering.
- The default endpoint is HTTPS and certificate verification is always on. Nothing
  in the SDK can turn verification off. An explicit `http://` `base_url` is honoured
  for local development, so point that only at a host you control.
