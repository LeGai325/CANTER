# Packaging changes

Original source and prior exports are retained outside this anonymous release.
The changes below concern portability, installation, documentation and local
review commands:

- machine-specific project/config/data paths were replaced by relative paths;
- the model endpoint is an example address, with an ignored local configuration
  and environment-variable template;
- both network launchers now invoke the active Python executable instead of a
  hardcoded machine-specific Conda interpreter;
- the experiment lock defaults to the checkout, supports an explicit shared path,
  and is propagated to frozen-source workers;
- install metadata, direct dependency pins, an analysis extra and a source manifest
  were added; the installed wheel retains the upstream scorer's provenance JSON;
  optional Linux experiment requirements are documented;
- private historical notes were replaced with reviewer-facing English documentation;
- an offline controller demonstration, review test runner, integrity verifier and
  configuration index were supplied;
- source provenance wording reflects the authors' confirmation of method reuse
  in larger-scale experiments.

The admission/estimator semantics, budgets, scoring, protocol guards and original
experimental configuration counts were retained. No 3,000-question run, frozen
cohort or result table was manufactured. The original Git history, model outputs,
datasets, credentials and private preparation records are outside this package.
