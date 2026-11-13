---
layout: ../../layouts/DocsLayout.astro
title: "Enterprise Setup"
---

# Enterprise Setup

Run ChunkHound across a fleet from one internal object store. Each machine reads its configuration from the store at the start of every run, and buffers usage events locally for upload when analytics is enabled.

A single machine does not need this setup.

Envelope schema, analytics fields, and precedence: [Configuration](/docs/configuration/#remote-configuration).

## How the parts connect

<figure class="d-figure" role="group" aria-label="A fleet of ChunkHound instances around one IT-owned object store">
  <figcaption>One bucket in the middle: config down, analytics up</figcaption>
  <span class="sr-only">ChunkHound runs as a fleet of instances against one IT-owned object store. Each instance fetches its configuration envelope from the config prefix at the start of every run, applies the envelope's rules to its global config, and keeps running on the saved config when the fetch fails. Analytics is opt-in: each command appends one event to a local buffer, and the client uploads batches to the analytics prefix. The instances never coordinate with each other. The organization's own ETL pipeline reads the analytics prefix into its warehouse.</span>
  <section class="d-band d-row" data-cols="2">
    <header class="d-row-rail">
      <svg class="d-row-icon" width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
        <path d="M2.75 8.4 10 3.6l7.25 4.8V16a.75.75 0 0 1-.75.75H3.5A.75.75 0 0 1 2.75 16Z" />
        <path d="M7.75 16.75v-4h4.5v4" />
      </svg>
      <span class="d-band-label d-row-label">Your warehouse</span>
      <span class="d-band-caption d-row-caption">your own analytics stack</span>
    </header>
    <div class="d-row-lane">
      <div class="d-node">
        <span class="d-node-title">Standard ETL pipeline</span>
        <span class="d-node-detail">Loads usage analytics into your warehouse.</span>
      </div>
    </div>
  </section>
  <div class="d-row d-row-link" data-cols="2">
    <div class="d-row-lane">
      <span class="d-row-arrow" data-dir="up" aria-hidden="true"></span>
    </div>
  </div>
  <section class="d-band d-row" data-tone="accent" data-cols="2">
    <header class="d-row-rail">
      <svg class="d-row-icon" width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
        <path d="M3.4 7h13.2" />
        <path d="M4.1 7h11.8l-1.3 8.6a1.1 1.1 0 0 1-1.1.9H6.5a1.1 1.1 0 0 1-1.1-.9Z" />
      </svg>
      <span class="d-band-label d-row-label">IT-owned bucket</span>
      <span class="d-band-caption d-row-caption">the only shared component</span>
    </header>
    <div class="d-row-lane">
      <div class="d-node">
        <span class="d-node-title">Remote config</span>
        <span class="d-node-detail"><code>config/<wbr>&lt;repo&gt;/<wbr>&lt;ring&gt;/<wbr>envelope.json</code></span>
      </div>
      <div class="d-node">
        <span class="d-node-title">Usage analytics</span>
        <span class="d-node-detail"><code>analytics/<wbr>&lt;repo&gt;/<wbr>&lt;identity&gt;/<wbr>YYYY/<wbr>MM/<wbr>DD/*</code></span>
      </div>
    </div>
  </section>
  <div class="d-row d-row-link" data-cols="2">
    <div class="d-row-lane">
      <span class="d-row-arrow" data-dir="down" aria-hidden="true"></span>
      <span class="d-row-arrow" data-dir="up" aria-hidden="true"></span>
    </div>
  </div>
  <section class="d-band d-row" data-cols="2">
    <header class="d-row-rail">
      <span class="d-row-mark" aria-hidden="true">
        <img src="/logo.svg" alt="" class="d-row-mark-light" />
        <img src="/logo-light.svg" alt="" class="d-row-mark-dark" />
      </span>
      <span class="d-band-label d-row-label">ChunkHound fleet</span>
      <span class="d-band-caption d-row-caption">thousands, no coordination between them</span>
    </header>
    <div class="d-row-lane">
      <div class="d-node">
        <span class="d-node-title">Reads its envelope</span>
        <span class="d-node-detail">On every start, before work begins.</span>
      </div>
      <div class="d-node">
        <span class="d-node-title">Pushes usage events</span>
        <span class="d-node-detail">Buffered locally, shipped in batches.</span>
      </div>
    </div>
  </section>
  <p class="d-note">Config is fetched before every run and never cached; a failed fetch falls back to the config on disk.</p>
</figure>

The only stateful component is the object store, and it is the only thing the fleet shares. Each machine reads centralized configuration from the store before every run and writes usage analytics back to it, so no machine needs to know about any other. Adding machines adds clients, not ChunkHound services: only the store must scale with the fleet.

## What the store must provide

Config lives at any HTTPS URL you choose, so its key layout is yours. The `config/` in the examples is a convention.

Analytics keys are fixed; [Usage Analytics](/docs/configuration/#usage-analytics)
shows the key and how the endpoint places it.

<aside class="docs-callout">
  <p class="docs-callout-title"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M10 2.5 4 4.8v4.4c0 3.4 2.4 6.4 6 7.6 3.6-1.2 6-4.2 6-7.6V4.8Z" /><path d="m7.4 9.6 1.9 1.9 3.4-3.6" /></svg>Least privilege</p>
  <div class="docs-callout-body">
    <p>ChunkHound needs read access to the config URL and write-only access to the analytics keys: <code>PutObject</code> only, with no read, list, or delete.</p>
  </div>
</aside>

## Fetch the configuration

`remote_config.url` is any HTTPS URL that returns the envelope JSON. The client sends a `GET` with one optional `Authorization` header, taken from `remote_config.auth_header`.

Set both in the global config file or with `CHUNKHOUND_REMOTE_CONFIG__URL` and `CHUNKHOUND_REMOTE_CONFIG__AUTH_HEADER`; file discovery, precedence, and backups: [Configuration](/docs/configuration/#remote-configuration).

### Pull directly

When the store serves the envelope readably, the URL is the whole configuration:

```json
{
  "remote_config": {
    "url": "https://objects.corp.internal/chunkhound/config/<repo>/prod/envelope.json"
  }
}
```

Leave `auth_header` unset: some stores reject an unexpected `Authorization` header.

### Pull through a proxy

The proxy holds the store credentials and reads the envelope with its own identity, so it can sign requests or present a client certificate when the store requires it. It validates the bearer token and maps it to a repo and a ring:

```json
{
  "remote_config": {
    "url": "https://config.corp.internal/chunkhound/envelope.json",
    "auth_header": "Bearer ${CHUNKHOUND_CONFIG_TOKEN}"
  }
}
```

Redirects stay on the same origin: the client drops `Authorization` on a cross-origin redirect. The proxy access log is your pull audit.

Open-source proxies for this:

- [Envoy](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/jwt_authn_filter): token checks and store request signing in one process.
- [Caddy](https://github.com/greenpau/caddy-security) with `caddy-security`: JWT and PASETO checks plus ACLs; add [aws-sigv4-proxy](https://github.com/awslabs/aws-sigv4-proxy) when the store needs SigV4.
- [oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/behaviour/) behind Traefik or Nginx: JWT bearer checks; add a signer for the store.

The token-to-ring map lives in the proxy config or a small service.

## Envelope rollout model

An envelope is a JSON list of changes that each machine applies at startup. Keep it in Git, validate it in CI, and publish it at the ring's pull URL.

```json
{
  "version": 1,
  "min_chunkhound_version": "5.3.0",
  "rules": [
    { "id": "embedding.provider", "op": "set", "value": "openai" }
  ]
}
```

- Each rule changes one setting. Rules run in order, and the last rule at a path wins.
- An envelope changes only the settings it names. To undo a setting, publish a rule that sets the old value or removes it.
- A bad rule is skipped and the rest still apply. If the result would break the config, the client discards the envelope.
- `min_chunkhound_version` keeps a new envelope off an old binary.
- `when.os` and `when.existing` are the only client-side selectors; use separate URLs or tokens for groups.
- Values are written to the global config in plaintext, so keep secrets in the environment.

Schema, rule ids, predicates, and the delta gate: [Envelope format](/docs/configuration/#envelope-format), [Rules](/docs/configuration/#rules).

## Analytics

Analytics is off by default. When enabled, each command appends to a local
buffer, and a flush uploads one `application/x-ndjson` object. Key layout, the
buffer cap, and the upload failure model: [Usage Analytics](/docs/configuration/#usage-analytics).

- Credentials are static: the environment pair or the `analytics.*` keys. There is no STS token, role discovery, or refresh.
- Set `anonymize` to `hashed` or `anonymous` before rollout; `full` records the OS username.

## Operational constraints

Every command fetches remote config at startup. Size the store and the endpoint
for the whole fleet and keep them reachable; the fetch budget and failure model
live in [Fetch behavior](/docs/configuration/#fetch-behavior-and-failure-model).

Envelopes are not signed, so TLS and review in Git are the integrity boundary.

Fleet targeting is limited to URLs, tokens, `when.os`, and `when.existing`. There is no push channel or dashboard.

To verify a rollout, run `chunkhound index --simulate --verbose`, then stop the endpoint and confirm a command still runs from the last saved config. Fetch and upload failure behavior: [Fetch behavior](/docs/configuration/#fetch-behavior-and-failure-model), [Analytics failure behavior](/docs/configuration/#failure-behavior).

## Where to next

- [Configuration precedence](/docs/configuration/#configuration-precedence)
- [Architecture](/docs/architecture/)
