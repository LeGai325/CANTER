# Reviewing an anonymous release

The public artifact should contain source, tests, sample configuration, documentation
and small verification records. Keep the owner-facing preparation report and raw
backups outside it.

Before sharing an anonymous mirror:

1. Check file contents and names for author names, affiliations, contact addresses,
   institutional routes, private URLs, credentials and personal filesystem paths.
   Inspect images, notebooks, PDFs and archives separately if adding any later.
2. Keep `.env`, `configs/*.local.json`, datasets, result directories, caches and Git
   history out of the exported archive. The supplied `.gitignore` is a convenience;
   inspect the actual upload list, including already-tracked files.
3. Preserve third-party attribution. It is not an author identity to redact.
   Have the authors select a project license before claiming an open-source license.
4. Retain the scope distinction in [EXPERIMENTS.md](EXPERIMENTS.md). Any future final
   configuration or result addition should state its source and validation status.
5. Download the archive served by the anonymous platform and run the README commands
   again: automatic text substitutions can affect code, paths and hashes.

A checksum manifest applies to the exact distributed bytes. If an anonymizer or an
author intentionally changes those bytes, regenerate the manifest for that export.
The anonymous view should provide enough information to use the source without
linking to an identifying owner account. Platform ownership/access settings are
managed separately from these files.
